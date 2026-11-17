"""Locomotion-compatible runner for the table-approach curriculum."""

from isaaclab.utils.configclass import configclass

from LearningRLlab.tasks.locomotion.rpo.agents.forward_walk_ppo_cfg import LearningRLRPOForwardWalkPPORunnerCfg


@configclass
class RPOApproachPPORunnerCfg(LearningRLRPOForwardWalkPPORunnerCfg):
    experiment_name = "learningrl_rpo_approach"
    run_name = "table_approach"
