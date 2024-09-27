
import sys, os
import numpy as np
import cv2
import torch
import glob
from skimage.metrics import structural_similarity
from scipy.ndimage import binary_erosion

sys.path.insert(0, os.getcwd())
from projects.csim.scripts import lpips_models
from lab4d.utils.vis_utils import img2color


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



def main():
    seqname1 = "eagle-d-0000"
    skip_idx = 1

    rgb_pred_files = "projects/diffgs/scripts/eval/temp/*.png"
    depth_pred_files = "database/processed/Depth/Full-Resolution/eagle-d-0000/*.npy"

    rgb_gt_files = "database/processed/JPEGImages/Full-Resolution/{0}/*.jpg".format(seqname1) 
    depth_gt_files = "database/processed/Depth/Full-Resolution/{0}/*.npy".format(seqname1)
    mask_gt_files = "database/processed/Annotations/Full-Resolution/{0}/*.npy".format(seqname1)

    lpips_model = lpips_models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True,version=0.1)
    
    raw_size = (512, 512)

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
                                         sorted(glob.glob(rgb_pred_files))[::skip_idx],
                                         sorted(glob.glob(depth_pred_files))[::skip_idx])):
        rgb_gt_file, depth_gt_file, mask_gt_file, rgb_pred_file, depth_pred_file = files_paths
        
        mask = np.load(mask_gt_file) > 0
        rgb_gt = cv2.imread(rgb_gt_file)[...,::-1]/255.
        rgb_pred = cv2.imread(rgb_pred_file)[...,::-1]/255.
        depth_gt = cv2.resize(np.load(depth_gt_file), raw_size[::-1])
        depth_pred = cv2.resize(np.load(depth_pred_file), raw_size[::-1])

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
        
        depth_vis = img2color("depth", np.concatenate([depth_gt, depth_pred, depth_err], axis=0)[...,None])
        cv2.imwrite("tmp/%05d-depth.jpg"%it, depth_vis[...,::-1]*255)
        cv2.imwrite("tmp/%05d-rgb.jpg"%it, np.concatenate([rgb_gt, rgb_pred], axis=0)[...,::-1]*255)
    
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

if __name__ == "__main__":
    main()