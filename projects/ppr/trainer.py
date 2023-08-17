# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os, sys
import pdb
import torch
import numpy as np
import tqdm

from lab4d.engine.trainer import Trainer
from lab4d.engine.trainer import get_local_rank
from lab4d.engine.model import dvr_model
from ppr import config

sys.path.insert(0, "%s/ppr-diffphys" % os.path.join(os.path.dirname(__file__)))
from diffphys.dp_interface import phys_interface, query_q
from diffphys.vis import Logger
from diffphys.dp_utils import se3_loss


class dvr_phys_reg(dvr_model):
    """A model that contains a collection of static/deformable neural fields

    Args:
        config (Dict): Command-line args
        data_info (Dict): Dataset metadata from get_data_info()
    """

    @torch.no_grad()
    def copy_phys_traj(self, phys_model):
        self.steps_fr = torch.arange(phys_model.total_frames, device=self.device)
        self.phys_q = phys_model.root_pose_mlp(self.steps_fr)  # N, 7
        self.phys_ja = phys_model.joint_angle_mlp(self.steps_fr)

    def forward(self, batch):
        loss_dict = super().forward(batch)
        reg_phys = self.compute_kinemaics_phys_diff()
        reg_phys = self.config["reg_phys_wt"] * reg_phys
        loss_dict["phys_reg"] = reg_phys
        return loss_dict

    def compute_kinemaics_phys_diff(self):
        """
        compute the difference between the target kinematics and kinematics estimated by physics proxy
        """
        object_field = self.fields.field_params["fg"]
        scene_field = self.fields.field_params["bg"]
        kinematics_q = query_q(self.steps_fr, object_field, scene_field)[0]
        kinematics_ja = object_field.warp.articulation.get_vals(
            self.steps_fr, return_so3=True
        )

        loss_q = se3_loss(self.phys_q, kinematics_q).mean()
        loss_ja = (self.phys_ja - kinematics_ja).pow(2).mean()
        # print("loss_q:", loss_q)
        # print("loss_ja:", loss_ja)
        loss = 1e-2 * loss_q + loss_ja
        return loss


class PPRTrainer(Trainer):
    def __init__(self, opts):
        """Train and evaluate a Lab4D model.

        Args:
            opts (Dict): Command-line args from absl (defined in lab4d/config.py)
        """
        opts["phys_vid"] = [int(i) for i in opts["phys_vid"].split(",")]
        opts["urdf_template"] = opts["fg_motion"].split("-")[1].split("_")[0]

        super().__init__(opts)
        self.model.fields.field_params["bg"].compute_field2world()

    def trainer_init(self):
        super().trainer_init()

        opts = self.opts
        self.current_steps_phys = 0  # 0-total_steps
        self.iters_per_phys_cycle = int(
            opts["ratio_phys_cycle"] * opts["iters_per_round"]
        )
        print("# iterations per phys cycle:", self.iters_per_phys_cycle)

    def init_model(self):
        """Initialize camera transforms, geometry, articulations, and camera
        intrinsics from external priors, if this is the first run"""
        # super().init_model()
        return

    def define_model(self, model=dvr_phys_reg):
        super().define_model(model=model)
        # opts
        opts = self.opts
        # model
        model_dict = {}
        model_dict["scene_field"] = self.model.fields.field_params["bg"]
        model_dict["object_field"] = self.model.fields.field_params["fg"]
        model_dict["intrinsics"] = self.model.intrinsics

        # define phys model
        self.phys_model = phys_interface(opts, model_dict)
        self.phys_visualizer = Logger(opts)

        # move model to device
        self.device = torch.device("cuda:{}".format(get_local_rank()))
        self.phys_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.phys_model)
        self.phys_model = self.phys_model.to(self.device)

    def load_checkpoint_train(self):
        """Load a checkpoint at training time and update the current step count
        and round count
        """
        super().load_checkpoint_train()
        # training time
        if self.opts["load_path_bg"] != "":
            _ = self.load_checkpoint(self.opts["load_path_bg"], self.model)

        # reset near_far
        self.model.fields.reset_geometry_aux()

    def get_lr_dict(self):
        """Return the learning rate for each category of trainable parameters

        Returns:
            param_lr_startwith (Dict(str, float)): Learning rate for base model
            param_lr_with (Dict(str, float)): Learning rate for explicit params
        """
        # define a dict for (tensor_name, learning) pair
        param_lr_startwith, param_lr_with = super().get_lr_dict()
        opts = self.opts

        param_lr_with.update(
            {
                "module.fields.field_params.fg.basefield.": 0.0,
                "module.fields.field_params.fg.colorfield.": 0.0,
                "module.fields.field_params.fg.sdf.": 0.0,
                "module.fields.field_params.fg.rgb.": 0.0,
                "module.fields.field_params.bg.basefield.": 0.0,
                "module.fields.field_params.bg.colorfield.": 0.0,
                "module.fields.field_params.bg.sdf.": 0.0,
                "module.fields.field_params.bg.rgb.": 0.0,
            }
        )
        return param_lr_startwith, param_lr_with

    def run_one_round(self, round_count):
        if round_count == 0:
            # initialize control input of phys model to kinematics
            self.phys_model.override_states()
        # run physics cycle
        self.run_phys_cycle()
        # # transfer phys-optimized kinematics to dvr
        # self.phys_model.override_states_inv()
        # transfer hys-optimized kinematics to dvr as soft constriaints
        self.model.copy_phys_traj(self.phys_model)
        # run dr cycle
        super().run_one_round(round_count)

    def run_phys_cycle(self):
        opts = self.opts
        torch.cuda.empty_cache()

        # eval
        self.phys_model.eval()
        self.phys_model.correct_foot_position()
        if self.current_round == 0:
            self.run_phys_visualization(tag="kinematics")

        # train
        self.phys_model.train()
        # to use the same amount memory: batch * time_per_wdw = 2.4*20 = 48
        num_envs = int(48 / opts["secs_per_wdw"])
        frames_per_wdw = int(opts["secs_per_wdw"] / self.phys_model.frame_interval) + 1
        print("num_envs:", num_envs)
        print("frames_per_wdw:", frames_per_wdw)
        self.phys_model.reinit_envs(
            num_envs,
            frames_per_wdw=frames_per_wdw,
            is_eval=False,
        )
        for i in tqdm.tqdm(range(self.iters_per_phys_cycle)):
            self.phys_model.set_progress(self.current_steps_phys)
            self.run_phys_iter()
            self.current_steps_phys += 1

        # eval again
        self.phys_model.eval()
        self.run_phys_visualization(tag="phys")

    def run_phys_iter(self):
        """Run physics optimization"""
        phys_aux = self.phys_model()
        self.phys_model.backward(phys_aux["total_loss"])
        self.phys_model.update()
        if get_local_rank() == 0:
            del phys_aux["total_loss"]
            self.add_scalar(self.log, phys_aux, self.current_steps_phys)

    def run_phys_visualization(self, tag=""):
        opts = self.opts
        frame_offset_raw = self.phys_model.frame_offset_raw
        vid_frame_max = max(frame_offset_raw[1:] - frame_offset_raw[:-1])
        self.phys_model.reinit_envs(1, frames_per_wdw=vid_frame_max, is_eval=True)
        for vidid in opts["phys_vid"]:
            frame_start = torch.zeros(1) + frame_offset_raw[vidid]
            _ = self.phys_model(frame_start=frame_start.to(self.device))
            img_size = tuple(self.data_info["raw_size"][vidid][::-1])
            img_size = img_size + (0.5,)  # scale
            data = self.phys_model.query(img_size=img_size)
            self.phys_visualizer.show(
                "%s-%02d-%05d" % (tag, vidid, self.current_steps_phys),
                data,
                fps=1.0 / self.phys_model.frame_interval,
            )

    def save_checkpoint(self, round_count):
        super().save_checkpoint(round_count)
        self.phys_model.save_checkpoint(round_count)
