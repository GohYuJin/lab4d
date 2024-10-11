import sys, os
from absl import app, flags
import numpy as np
import cv2
import torch
import glob
from skimage.metrics import structural_similarity
from scipy.ndimage import binary_erosion

sys.path.insert(0, os.getcwd())
from projects.csim.scripts import lpips_models
from lab4d.utils.vis_utils import img2color
from lab4d.render import get_config, construct_batch_from_opts, render_batch
from projects.diffgs.trainer import GSplatTrainer as Trainer


def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def compute_depth_acc_at_10cm(dph_gt, dph, conf_gt, mask=None, dep_scale = 0.2):
    """
    from totalrecon
    """
    # INPUTS:
    # 1. dph_gt:            Ground truth depth image (scaled by "dep_scale"):                                       numpy array of shape = (H, W)
    # 2. dph:               Rendered depth image     (scaled by "dep_scale"):                                       numpy array of shape = (H, W)
    # 3. conf_gt:           Ground truth depth-confidence image:                                                    numpy array of shape = (H, W)
    # 4. mask:              Binary spatial mask over which to compute the metric:                                   numpy array of shape = (H, W)
    # 5. dep_scale:         Scale used to scale the ground truth depth during training
    #
    # RETURNS:
    # 1. depth accuracy at 0.1m:  Computes the number of test rays estimated with 0.1m of their ground truth
    depth_diff = (dph - dph_gt) / dep_scale                                             # depth difference in meters

    if mask is None:
        mask = np.ones_like(conf_gt)                                                    # shape = (H, W)

    # compute depth accuracy @ 0.1m over pixels that 1) have high confidence value (conf_gt > 1.5) and 2) a mask value of 1
    is_depth_accurate = (np.abs(depth_diff) < 0.1)
    depth_acc_at_10cm = np.mean(is_depth_accurate[(conf_gt > 1.5) & (mask == 1.)])

    return depth_acc_at_10cm, is_depth_accurate

def compute_lpips(rgb_gt, rgb, lpips_model, mask=None):
    # IMPORTANT!!! Both rgb_gt and rgb need to be in range [0, 1]
    # INPUTS:
    # 1. rgb_gt:            Ground truth image:                                         numpy array of shape = (H, W, 3)         
    # 2. rgb:               Rendered image:                                             numpy array of shape = (H, W, 3)
    # 3. lpips_model:       torch lpips_model (instantiate once in the main script, and feed as input to this method):      "lpips_model = lpips_models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True,version=0.1)""
    # 4. mask:              Binary spatial mask over which to compute the metric:       numpy array of shape = (H, W)
    #
    # # OUTPUTS:
    # 1. lpips

    rgb_gt_0 = im2tensor(rgb_gt).cuda()                                                     # torch tensor of shape = (H, W, 3)
    rgb_0 = im2tensor(rgb).cuda()                                                           # torch tensor of shape = (H, W, 3)
    
    if mask is not None:
        mask_rgb = np.repeat(mask[..., np.newaxis], 3, axis=-1)                     # shape = (H, W, 3)
    else:
        mask_rgb = np.ones_like(rgb)                                                # shape = (H, W, 3)
    
    mask_0 = torch.Tensor(mask_rgb[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))            # torch tensor of shape = (1, 3, H, W)
    lpips = lpips_model.forward(rgb_gt_0, rgb_0, mask_0).item()
    return lpips

def compute_psnr(rgb_gt, rgb, mask=None):
    if rgb_gt.dtype == np.uint8 and rgb.dtype == np.uint8:
        max_pixel = 255.0
    elif (rgb_gt <= 1.0).all() and (rgb <= 1.0).all():
        max_pixel = 1.0
    else:
        raise TypeError("unsupported datatype in images")
    
    if mask is not None:
        mse = np.mean((rgb_gt[mask] - rgb[mask]) ** 2) 
    else:
        mse = np.mean((rgb_gt - rgb) ** 2) 

    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) 
    return psnr 

def compute_ssim(rgb_gt, rgb, mask=None, channel_axis=2, win_size=7):
    ssim_score, S = structural_similarity(rgb_gt, rgb, 
                                          channel_axis=channel_axis, 
                                          full=True, 
                                          win_size=win_size)
    pad = (win_size - 1) // 2
    if mask is not None:
        # to avoid edge effects will ignore filter radius strip around edges
        # In skimage library they crop the image by the padding factor
        # as we are using a mask we will shrink the mask instead
        mask = binary_erosion(mask, structure=np.ones((pad,pad)))
        return S[mask].mean()
    else:
        return ssim_score 

def subsample(batch, skip_idx):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v[::skip_idx] # every 10-th frame
        else:
            subsample(v, skip_idx)

def construct_batch_from_opts(opts, model, data_info):
    from lab4d.utils.camera_utils import (
        construct_batch,
        get_fixed_cam,
        get_object_to_camera_matrix,
        get_orbit_camera,
        get_rotating_cam,
        create_field2cam,
        get_bev_cam,
    )
    from lab4d.utils.geom_utils import K2inv, K2mat, mat2K
    from lab4d.dataloader import data_utils

    device = "cuda"
    # data info
    if "motion_id" in opts:
        video_id = opts["motion_id"]
    else:
        video_id = opts["inst_id"]
    # ref video size
    raw_size = data_info["raw_size"][video_id]  # full range of pixels
    # ref video length
    vid_length = data_utils.get_vid_length(video_id, data_info)

    # whether to freeze a frame
    if opts["freeze_id"] == -1:
        if opts["noskip"]:
            # render all frames
            frameid_sub = np.arange(vid_length)
            render_length = vid_length
        else:
            # render filtered frames
            frame_mapping = data_info["frame_info"]["frame_mapping"]
            frame_offset = data_info["frame_info"]["frame_offset"]
            frameid = frame_mapping[frame_offset[video_id] : frame_offset[video_id + 1]]

            frameid_start = data_info["frame_info"]["frame_offset_raw"][video_id]
            frameid_sub = frameid - frameid_start
            render_length = len(frameid)
        # remove last frame to be consistent with flow
        frameid_sub = frameid_sub[:-1]
        render_length = render_length - 1
    elif opts["freeze_id"] >= 0 and opts["freeze_id"] < vid_length:
        if opts["num_frames"] <= 0:
            num_frames = vid_length
        else:
            num_frames = opts["num_frames"]
        frameid_sub = np.asarray([opts["freeze_id"]] * num_frames)
    else:
        raise ValueError("frame id %d out of range" % opts["freeze_id"])
    print("rendering frames: %s from video %d" % (str(frameid_sub), video_id))
    frameid = frameid_sub + data_info["frame_info"]["frame_offset_raw"][video_id]

    # get cameras wrt each field
    with torch.no_grad():
        frameid = torch.tensor(frameid, device=device)
        field2cam_fr = model.get_cameras(frame_id=frameid)
        intrinsics_fr = model.get_intrinsics(
            frameid_sub + data_info["frame_info"]["frame_offset_raw"][video_id]
        )
        aabb = model.get_aabb(inst_id=opts["inst_id"])
    # convert to numpy
    for k, v in field2cam_fr.items():
        if torch.is_tensor(v):
            field2cam_fr[k] = v.cpu().numpy()
        if torch.is_tensor(aabb[k]):
            aabb[k] = aabb[k].cpu().numpy()
    if torch.is_tensor(intrinsics_fr):
        intrinsics_fr = intrinsics_fr.cpu().numpy()

    # construct batch from user input
    if opts["viewpoint"] == "ref":
        # rotate around viewpoint
        field2cam = None

        # camera_int = None
        crop2raw = np.zeros((len(frameid_sub), 4))
        crop2raw[:, 0] = raw_size[1] / opts["render_res"]
        crop2raw[:, 1] = raw_size[0] / opts["render_res"]
        camera_int = mat2K(K2inv(crop2raw) @ K2mat(intrinsics_fr))
        crop2raw = None
    elif opts["viewpoint"].startswith("rot"):
        # rotate around field, format: rot-evelvation-degree
        elev, max_angle = [int(val) for val in opts["viewpoint"].split("-")[1:]]

        # bg_to_cam
        obj_size = (aabb["fg"][0, 1, :] - aabb["fg"][0, 0, :]).max()
        cam_traj = get_rotating_cam(
            len(frameid_sub), distance=obj_size * 2.5, max_angle=max_angle
        )
        cam_elev = get_object_to_camera_matrix(elev, [1, 0, 0], 0)[None]
        cam_traj = cam_traj @ cam_elev
        field2cam = create_field2cam(cam_traj, field2cam_fr.keys())

        camera_int = np.zeros((len(frameid_sub), 4))

        # focal length = img height * distance / obj height
        camera_int[:, :2] = opts["render_res"] * 2 * 0.8  # zoom out a bit
        camera_int[:, 2:] = opts["render_res"] / 2
        raw_size = (640, 640)  # full range of pixels
        crop2raw = None
    elif opts["viewpoint"].startswith("bev"):
        elev = int(opts["viewpoint"].split("-")[1])
        # render bird's eye view
        if "bg" in field2cam_fr.keys():
            # get bev wrt first frame image
            # center_to_bev = centered_to_camt0 x centered_to_rotated x camt0_to_centered x bg_to_camt0
            center_to_bev = get_object_to_camera_matrix(elev, [1, 0, 0], 0)[None]
            camt0_to_center = np.eye(4)
            camt0_to_center[2, 3] = -field2cam_fr["bg"][0, 2, 3]
            camt0_to_bev = (
                np.linalg.inv(camt0_to_center) @ center_to_bev @ camt0_to_center
            )
            bg2bev = camt0_to_bev @ field2cam_fr["bg"][:1]
            # push cameras away
            bg2bev[..., 2, 3] *= 3
            field2cam = {"bg": np.tile(bg2bev, (render_length, 1, 1))}
            if "fg" in field2cam_fr.keys():
                # if both fg and bg
                camt2bg = np.linalg.inv(field2cam_fr["bg"])
                fg2camt = field2cam_fr["fg"]
                field2cam["fg"] = field2cam["bg"] @ camt2bg @ fg2camt
        elif "fg" in field2cam_fr.keys():
            # if only fg
            field2cam = {"fg": get_bev_cam(field2cam_fr["fg"], elev=elev)}
        else:
            raise NotImplementedError

        camera_int = np.zeros((len(frameid_sub), 4))
        camera_int[:, :2] = opts["render_res"] * 2
        camera_int[:, 2:] = opts["render_res"] / 2
        raw_size = (640, 640)  # full range of pixels
        crop2raw = None
    else:
        raise ValueError("Unknown viewpoint type %s" % opts.viewpoint)

    batch = construct_batch(
        inst_id=opts["inst_id"],
        frameid_sub=frameid_sub,
        eval_res=opts["render_res"],
        field2cam=field2cam,
        camera_int=camera_int,
        crop2raw=crop2raw,
        device=device,
    )
    return batch, raw_size

def main(_):
    render_res = 512
    inst_id = 0
    seqname1 = "eagle-d-{0:04}".format(inst_id)
    skip_idx = 1

    # rgb_pred_files = "projects/diffgs/scripts/eval/temp/rgb/*.png"
    # depth_pred_files = "projects/diffgs/scripts/eval/temp/depth/*.npy"

    rgb_gt_files = "database/processed/JPEGImages/Full-Resolution/{0}/*.jpg".format(seqname1) 
    depth_gt_files = "database/processed/Depth/Full-Resolution/{0}/*.npy".format(seqname1)
    mask_gt_files = "database/processed/Annotations/Full-Resolution/{0}/*.npy".format(seqname1)

    opts = get_config()
    opts["render_res"] = render_res
    opts["inst_id"] = inst_id
    opts["load_suffix"] = "latest"
    opts["n_depth"] = 256
    opts["logroot"] = sys.argv[1].split("=")[1].rsplit("/", 2)[0]
    model, data_info, ref_dict = Trainer.construct_test_model(opts, force_reload=False, return_refs=False)
    
    opts["seqname"] = "eagle-d"
    opts["logname"] = "diffgs-fs-view_n0123_fg-b4-bob-r120-const"
    model2, data_info2, _ = Trainer.construct_test_model(opts, force_reload=False, return_refs=False)

    batch, raw_size = construct_batch_from_opts(opts, model2, data_info2)
    from lab4d.utils.quat_transform import se3_to_quaternion_translation
    from lab4d.utils.geom_utils import K2inv, K2mat, mat2K
    from lab4d.render import batch_to_flow_batch

    batch = batch_to_flow_batch(batch)
    model2.process_frameid(batch)
    model2.reshape_batch_inv(batch)
    crop_size = model2.config["render_res"]
    Kmat, w2c = model2.compute_camera_samples(batch, crop_size)
    frameid = batch["frameid"]
    batch["dataid"] = batch["dataid"] * 0
    frameid = frameid % (data_info2["frame_info"]["frame_offset"][inst_id + 1] - 
                                    data_info2["frame_info"]["frame_offset"][inst_id])

    model2.gaussians.update_trajectory(frameid)
    model.gaussians.update_trajectory(frameid)

    rendered, _ = model.render_pair(crop_size, Kmat, w2c=w2c, frameid=frameid)
    for k, v in rendered.items():
        rendered[k] = v[:, 0].detach()
    rendered = model.rendered_to_output(rendered, return_numpy=True)

    rendered_pair, _ = model2.render_pair(crop_size, Kmat, w2c=w2c, frameid=frameid)
    for k, v in rendered_pair.items():
        rendered_pair[k] = v[:, 0].detach()
    rendered_pair = model.rendered_to_output(rendered_pair, return_numpy=True)
    cv2.imwrite("tmp/other_sample_rgb_pred.png", 
                cv2.cvtColor((rendered_pair["rgb"][-1]*255).astype(np.uint8), cv2.COLOR_RGB2BGR))


    lpips_model = lpips_models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True,version=0.1)

    lpips_list = []
    lpips_bg_list = []
    lpips_fg_list = []
    psnr_list = []
    psnr_bg_list = []
    psnr_fg_list = []
    ssim_list = []
    ssim_bg_list = []
    ssim_fg_list = []
    depth_acc_list = []
    depth_acc_bg_list = []
    depth_acc_fg_list = []
    for it, files_paths in enumerate(zip(sorted(glob.glob(rgb_gt_files))[::skip_idx], 
                                         sorted(glob.glob(depth_gt_files))[::skip_idx], 
                                         sorted(glob.glob(mask_gt_files))[::skip_idx],
                                         rendered["rgb"][::skip_idx],
                                         rendered["depth"][::skip_idx])):
        rgb_gt_file, depth_gt_file, mask_gt_file, rgb_pred_file, depth_pred_file = files_paths
        
        mask = np.load(mask_gt_file) > 0
        rgb_gt = cv2.imread(rgb_gt_file)[...,::-1]/255.
        rgb_pred = rgb_pred_file #cv2.imread(rgb_pred_file)[...,::-1]/255.
        depth_gt = cv2.resize(np.load(depth_gt_file), raw_size[::-1])
        depth_pred = depth_pred_file[:,:,0] #cv2.resize(np.load(depth_pred_file), raw_size[::-1])

        depth_acc, depth_err = compute_depth_acc_at_10cm(depth_gt, depth_pred, np.ones_like(depth_gt) * 2, mask=None, dep_scale = 1)
        depth_acc_list.append(depth_acc)
        depth_acc_fg, _ = compute_depth_acc_at_10cm(depth_gt, depth_pred, np.ones_like(depth_gt) * 2, mask=mask, dep_scale = 1)
        depth_acc_fg_list.append(depth_acc_fg)
        depth_acc_bg, _ = compute_depth_acc_at_10cm(depth_gt, depth_pred, np.ones_like(depth_gt) * 2, mask=~mask, dep_scale = 1)
        depth_acc_bg_list.append(depth_acc_bg)

        lpips = compute_lpips(rgb_gt, rgb_pred, lpips_model, mask=None)
        lpips_list.append(lpips)
        lpips_fg = compute_lpips(rgb_gt, rgb_pred, lpips_model, mask=mask)
        lpips_fg_list.append(lpips_fg)
        lpips_bg = compute_lpips(rgb_gt, rgb_pred, lpips_model, mask=~mask)
        lpips_bg_list.append(lpips_bg)

        psnr = compute_psnr(rgb_gt, rgb_pred, mask=None)
        psnr_list.append(psnr)
        psnr_fg = compute_psnr(rgb_gt, rgb_pred, mask=mask)
        psnr_fg_list.append(psnr_fg)
        psnr_bg = compute_psnr(rgb_gt, rgb_pred, mask=~mask)
        psnr_bg_list.append(psnr_bg)

        ssim = compute_ssim(rgb_gt, rgb_pred, mask=None)
        ssim_list.append(ssim)
        ssim_fg = compute_ssim(rgb_gt, rgb_pred, mask=mask)
        ssim_fg_list.append(ssim_fg)
        ssim_bg = compute_ssim(rgb_gt, rgb_pred, mask=~mask)
        ssim_bg_list.append(ssim_bg)
        
        # depth_vis = img2color("depth", np.concatenate([depth_gt, depth_pred, depth_err], axis=0)[...,None])
        # cv2.imwrite("tmp/%05d-depth.jpg"%it, depth_vis[...,::-1]*255)
        # cv2.imwrite("tmp/%05d-rgb.jpg"%it, np.concatenate([rgb_gt, rgb_pred], axis=0)[...,::-1]*255)
    
    depth_acc_list = np.stack(depth_acc_list, 0)
    depth_acc_fg_list = np.stack(depth_acc_fg_list, 0)
    depth_acc_bg_list = np.stack(depth_acc_bg_list, 0)
    lpips_list = np.stack(lpips_list, 0)
    lpips_fg_list = np.stack(lpips_fg_list, 0)
    lpips_bg_list = np.stack(lpips_bg_list, 0)
    psnr_list = np.stack(psnr_list, 0)
    psnr_fg_list = np.stack(psnr_fg_list, 0)
    psnr_bg_list = np.stack(psnr_bg_list, 0)
    ssim_list = np.stack(ssim_list, 0)
    ssim_fg_list = np.stack(ssim_fg_list, 0)
    ssim_bg_list = np.stack(ssim_bg_list, 0)
    

    print("depth acc: %.3f" % depth_acc_list.mean())
    print("depth-fg acc: %.3f" % depth_acc_fg_list.mean())
    print("depth-bg acc: %.3f" % depth_acc_bg_list.mean())
    print("lpips: %.3f" % lpips_list.mean())
    print("lpips-fg: %.3f" % lpips_fg_list.mean())
    print("lpips-bg: %.3f" % lpips_bg_list.mean())
    print("psnr: %.3f" % psnr_list.mean())
    print("psnr-fg: %.3f" % psnr_fg_list.mean())
    print("psnr-bg: %.3f" % psnr_bg_list.mean())
    print("ssim: %.3f" % ssim_list.mean())
    print("ssim-fg: %.3f" % ssim_fg_list.mean())
    print("ssim-bg: %.3f" % ssim_bg_list.mean())

    cv2.imwrite("tmp/sample_rgb_pred.png", 
                cv2.cvtColor((rgb_pred*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    app.run(main)