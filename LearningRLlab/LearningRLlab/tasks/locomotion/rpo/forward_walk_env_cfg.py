"""Forward-only curriculum stage; no object manipulation or external assistance."""

import torch

from isaaclab.managers import RewardTermCfg
from isaaclab.utils.configclass import configclass

from .flat_walk_env_cfg import LearningRLRPOFlatWalkEnvCfg


def forward_speed_progress(env, command_name: str = "base_velocity") -> torch.Tensor:
    """Bounded signed forward speed; standing earns zero and backwards motion loses reward."""
    target = env.command_manager.get_command(command_name)[:, 0].clamp_min(0.05)
    speed = env.scene["robot"].data.root_lin_vel_b[:, 0]
    return torch.clamp(speed / target, min=-1.0, max=1.0)


@configclass
class LearningRLRPOForwardWalkEnvCfg(LearningRLRPOFlatWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 512
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.25)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.rewards.track_lin_vel_xy.weight = 3.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.12
        self.rewards.track_ang_vel_z.weight = 1.5
        self.rewards.track_ang_vel_z.params["std"] = 0.25
        self.rewards.forward_speed_progress = RewardTermCfg(func=forward_speed_progress, weight=1.0)
        self.rewards.action_rate.weight = -0.02


@configclass
class LearningRLRPOForwardWalkEnvCfg_PLAY(LearningRLRPOForwardWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 100
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.observations.policy.enable_corruption = False
