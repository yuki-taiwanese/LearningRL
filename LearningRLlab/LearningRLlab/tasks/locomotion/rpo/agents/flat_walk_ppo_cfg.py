"""Separate flat-walk checkpoint family, with lower initial exploration noise."""

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg import LearningRLRPOWalkPPORunnerCfg


@configclass
class LearningRLRPOFlatWalkPPORunnerCfg(LearningRLRPOWalkPPORunnerCfg):
    experiment_name = "learningrl_rpo_flat_walk"
    wandb_project = "RPO-locomotion"
    run_name = "flat_unloaded"
    logger = "tensorboard"
    max_iterations = 20000
    save_interval = 50

    def __post_init__(self):
        self.actor.distribution_cfg.init_std = 0.5
        # The upstream mirror utility expects a URDF spawn.asset_path, whereas
        # kit-less training uses a preconverted USD. Do not silently mirror it.
        self.algorithm.symmetry_cfg = None
