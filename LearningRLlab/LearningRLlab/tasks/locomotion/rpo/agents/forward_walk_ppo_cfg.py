"""Checkpoint family for the first, forward-only locomotion curriculum stage."""

from isaaclab.utils.configclass import configclass

from .flat_walk_ppo_cfg import LearningRLRPOFlatWalkPPORunnerCfg


@configclass
class LearningRLRPOForwardWalkPPORunnerCfg(LearningRLRPOFlatWalkPPORunnerCfg):
    experiment_name = "learningrl_rpo_forward_walk"
    run_name = "forward_curriculum"
