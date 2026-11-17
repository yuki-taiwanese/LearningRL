"""New observation/checkpoint family for phase-conditioned alternating steps."""

from isaaclab.utils.configclass import configclass

from .forward_walk_ppo_cfg import LearningRLRPOForwardWalkPPORunnerCfg


@configclass
class RPOAlternatingWalkPPORunnerCfg(LearningRLRPOForwardWalkPPORunnerCfg):
    experiment_name = "learningrl_rpo_alternating_walk"
    run_name = "alternating_3cm"
