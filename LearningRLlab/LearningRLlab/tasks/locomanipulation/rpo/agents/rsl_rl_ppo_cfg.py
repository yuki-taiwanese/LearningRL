"""Walking PPO baseline, without invalid left/right task augmentation."""

from isaaclab.utils.configclass import configclass

from LearningRLlab.tasks.locomotion.rpo.agents.rsl_rl_ppo_cfg import LearningRLRPOWalkPPORunnerCfg


@configclass
class RPOBoxTransferPPORunnerCfg(LearningRLRPOWalkPPORunnerCfg):
    experiment_name = "learningrl_rpo_box_transfer"
    wandb_project = "RPO-box-transfer"
    run_name = "two_tables"
    logger = "tensorboard"
    max_iterations = 20000
    save_interval = 500

    def __post_init__(self):
        self.algorithm.symmetry_cfg = None
