"""RPO delivers two differently sized boxes, returning left after each delivery."""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from LearningRLlab.scenes import FLAT_TERRAIN_CFG
from LearningRLlab.tasks.locomotion.g1.g1_env_cfg import ObservationsCfg as WalkObservations
from LearningRLlab.tasks.locomotion.g1.g1_env_cfg import SceneCfg as WalkScene
from LearningRLlab.tasks.locomotion.rpo.rpo_env_cfg import LearningRLRPOWalkEnvCfg

from .assets import CompoundCfg
from .layout import LAYOUT, box_parts, table_parts
from .mdp import terms
from .mdp.commands_cfg import TransferCommandCfg


def table_cfg(y, color):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + ("LeftTable" if y > 0 else "RightTable"),
        spawn=CompoundCfg(
            parts=table_parts(),
            color=color,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(LAYOUT.table_x, y, 0)),
    )


def box_cfg(index):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + ("Box" if index == 0 else "LargeBox"),
        spawn=CompoundCfg(
            parts=box_parts(size=LAYOUT.box_sizes[index]),
            color=(0.65, 0.42, 0.20) if index == 0 else (0.82, 0.57, 0.25),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=LAYOUT.box_masses[index]),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=LAYOUT.sources[index]),
    )


@configclass
class SceneCfg(WalkScene):
    terrain = FLAT_TERRAIN_CFG
    left_table = table_cfg(LAYOUT.table_y, (0.15, 0.35, 0.65))
    right_table = table_cfg(-LAYOUT.table_y, (0.15, 0.60, 0.30))
    box = box_cfg(0)
    large_box = box_cfg(1)
    box_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Box", "{ENV_REGEX_NS}/LargeBox"],
        history_length=1,
    )
    right_support = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/RightTable"],
        history_length=1,
    )
    large_right_support = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/LargeBox",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/RightTable"],
        history_length=1,
    )


@configclass
class ObservationsCfg(WalkObservations):
    @configclass
    class PolicyCfg(WalkObservations.PolicyCfg):
        transfer_state = ObsTerm(func=terms.task_observation)

    @configclass
    class CriticCfg(WalkObservations.CriticCfg):
        transfer_state = ObsTerm(func=terms.task_observation)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RPOBoxTransferEnvCfg(LearningRLRPOWalkEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=512, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Optional kit-less training after the course URDF importer has produced
        # its inertia-corrected USD. Preserve its authored physics variant.
        cached_usd = os.environ.get("LEARNINGRL_RPO_USD")
        if cached_usd:
            if not Path(cached_usd).is_file():
                raise FileNotFoundError(f"Preconverted RPO model not found: {cached_usd}")
            source = self.scene.robot.spawn
            self.scene.robot.spawn = sim_utils.UsdFileCfg(
                usd_path=cached_usd,
                activate_contact_sensors=source.activate_contact_sensors,
                rigid_props=source.rigid_props,
                articulation_props=source.articulation_props,
                joint_drive_props=source.joint_drive_props,
                collision_props=source.collision_props,
                mass_props=source.mass_props,
            )
        self.episode_length_s = LAYOUT.episode_seconds
        self.commands.base_velocity = TransferCommandCfg()
        self.scene.box_contact.update_period = self.sim.dt
        self.scene.right_support.update_period = self.sim.dt
        self.scene.large_right_support.update_period = self.sim.dt
        self.terminations.time_out = None
        # The walk's base reset and pushes conflict with a fixed two-table task.
        self.events.reset_base = None
        self.events.reset_task = EventTerm(
            func=terms.reset_task,
            mode="reset",
            params={"approach_distance": LAYOUT.approach_distance},
        )
        self.events.push_robot = None
        self.events.view_middle_env = None
        self.events.add_base_mass = None
        self.events.physics_material = None
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.rewards.joint_deviation_arms = None
        self.rewards.alive = None
        self.rewards.feet_clearance = None
        # No recurring positive reward for waiting/walking forever in an unlimited task.
        self.rewards.gait = None
        self.rewards.track_lin_vel_xy = None
        self.rewards.track_ang_vel_z = None
        self.rewards.action_rate.weight = -0.01
        # Forearm-box contacts are intentional. Penalize torso, legs and upper arms instead.
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "rpo_termination",
            body_names=["base_link", "torso_link", ".*_thigh_.*_link", ".*_knee_link", ".*_arm_.*_link"],
        )
        self.rewards.transfer = RewTerm(func=terms.task_reward, weight=1.0)
        self.rewards.failure = RewTerm(func=terms.terminal_failure, weight=-25.0 / (self.sim.dt * self.decimation))
        self.terminations.transfer_success = DoneTerm(func=terms.task_success)
        self.terminations.transfer_failure = DoneTerm(func=terms.task_failure)
        self.viewer.eye = (2.6, -3.5, 2.5)


@configclass
class RPOBoxTransferEnvCfg_PLAY(RPOBoxTransferEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 100
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.observations.policy.enable_corruption = False
