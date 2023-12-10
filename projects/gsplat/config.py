# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os

from absl import flags

opts = flags.FLAGS


class GSplatConfig:
    # configs related to gaussian splatting
    flags.DEFINE_float("guidance_sd_wt", 1e-5, "weight for sd loss")
    flags.DEFINE_float("guidance_zero123_wt", 0.0, "wegiht for zero123 loss")
    # flags.DEFINE_float("guidance_sd_wt", 0.0, "weight for sd loss")
    # flags.DEFINE_float("guidance_zero123_wt", 1e-4, "wegiht for zero123 loss")
