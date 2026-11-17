"""Unloaded flat-ground walking skill; not a carrying or grasping policy."""

from robot_description.robots.cached_usd import use_cached_rpo_usd

from isaaclab.utils.configclass import configclass

from LearningRLlab.scenes import FLAT_TERRAIN_CFG

from .rpo_env_cfg import LearningRLRPOWalkEnvCfg


@configclass
class LearningRLRPOFlatWalkEnvCfg(LearningRLRPOWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_cached_rpo_usd(self.scene.robot)
        self.scene.terrain = FLAT_TERRAIN_CFG.copy()
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.num_envs = 256
        self.scene.env_spacing = 3.0
        # This 20 s training horizon is not a deadline on the two-box task.
        self.episode_length_s = 20.0
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.resampling_time_range = (6.0, 10.0)
        self.commands.base_velocity.rel_standing_envs = 0.25
        self.commands.base_velocity.ranges.lin_vel_x = (-0.35, 0.35)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.35, 0.35)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.view_middle_env = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (0.0, 0.0),
            "yaw": (-0.1, 0.1),
        }
        self.events.reset_base.params["velocity_range"] = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


@configclass
class LearningRLRPOFlatWalkEnvCfg_PLAY(LearningRLRPOFlatWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 100
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.observations.policy.enable_corruption = False
