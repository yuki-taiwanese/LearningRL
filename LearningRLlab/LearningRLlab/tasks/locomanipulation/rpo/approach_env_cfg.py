"""Full two-table scene with the walking observation schema for staged control."""

import torch

from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass

from LearningRLlab.tasks.locomotion.rpo.forward_walk_env_cfg import LearningRLRPOForwardWalkEnvCfg

from .box_env_cfg import SceneCfg
from .layout import LAYOUT


def reset_approach_objects(env, env_ids):
    """Reset dynamic boxes at episode boundaries only; no object assistance during a trial."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    for name in ("box", "large_box"):
        asset = env.scene[name]
        state = asset.data.default_root_state.torch[env_ids].clone()
        state[:, :3] += env.scene.env_origins[env_ids]
        asset.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)


@configclass
class RPOApproachEnvCfg(LearningRLRPOForwardWalkEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=512, env_spacing=5.0)

    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 5.0
        # The curriculum begins aligned with the small box, one meter behind its stand point.
        stand = LAYOUT.source_stands[0]
        self.scene.robot.init_state.pos = (stand[0] - LAYOUT.approach_distance, stand[1], 0.75)
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.events.reset_base.params["pose_range"] = {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "yaw": (0.0, 0.0)}
        self.events.reset_objects = EventTermCfg(func=reset_approach_objects, mode="reset")
        self.episode_length_s = 60.0


@configclass
class RPOApproachEnvCfg_PLAY(RPOApproachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 100
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.observations.policy.enable_corruption = False
