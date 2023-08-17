# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os

from absl import flags

opts = flags.FLAGS


class PPRConfig:
    # configs related to ppr
    flags.DEFINE_string("urdf_template", "", "whether to use predefined skeleton")
    flags.DEFINE_float("ratio_phys_cycle", 0.2, "number of iterations per round")
    flags.DEFINE_float("secs_per_wdw", 2.4, "length of the physics opt window in secs")
    flags.DEFINE_string(
        "phys_vid", "0", "whether to optimize selected videos, e.g., 0,1,2"
    )

    # weights
    flags.DEFINE_float("traj_wt", 2e-3, "weight for traj matching loss")
    flags.DEFINE_float("pos_state_wt", 0.0, "weight for position matching reg")
    flags.DEFINE_float("vel_state_wt", 0.0, "weight for velocity matching reg")

    # regs
    flags.DEFINE_float("reg_torque_wt", 0.0, "weight for torque regularization")
    flags.DEFINE_float("reg_res_f_wt", 2e-5, "weight for residual force regularization")
    flags.DEFINE_float("reg_foot_wt", 0.0, "weight for foot contact regularization")
    flags.DEFINE_float("reg_root_wt", 0.0, "weight for root pose regularization")
    flags.DEFINE_float("reg_phys_wt", 1e-2, "weight for soft physics regularization")

    # io-related
    flags.DEFINE_string("load_path_bg", "", "path to load pretrained model")
