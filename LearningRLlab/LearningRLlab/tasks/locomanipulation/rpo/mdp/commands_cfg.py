"""Config-only imports: no CommandTerm or USD on Hydra's pre-Kit import path."""

from isaaclab.managers import CommandTermCfg
from isaaclab.utils.configclass import configclass


@configclass
class TransferCommandCfg(CommandTermCfg):
    class_type: str = "LearningRLlab.tasks.locomanipulation.rpo.mdp.commands:TransferCommand"
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    debug_vis: bool = False
