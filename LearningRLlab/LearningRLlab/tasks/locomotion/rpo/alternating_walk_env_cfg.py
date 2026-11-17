"""Alternating-step revision, isolated from the already evaluated forward baseline."""

from isaaclab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from LearningRLlab.tasks.locomotion.g1.g1_env_cfg import ObservationsCfg as WalkObservations

from . import gait_terms
from .forward_walk_env_cfg import LearningRLRPOForwardWalkEnvCfg

FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


def feet_cfg(key):
    return SceneEntityCfg(key, body_names=FEET.copy(), preserve_order=True)


@configclass
class AlternatingObservationsCfg(WalkObservations):
    @configclass
    class PolicyCfg(WalkObservations.PolicyCfg):
        gait_clock = ObservationTermCfg(func=gait_terms.phase_clock)

    @configclass
    class CriticCfg(WalkObservations.CriticCfg):
        gait_clock = ObservationTermCfg(func=gait_terms.phase_clock)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RPOAlternatingWalkEnvCfg(LearningRLRPOForwardWalkEnvCfg):
    observations: AlternatingObservationsCfg = AlternatingObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.gait = None
        self.rewards.feet_clearance = None
        for name, weight, component in (
            ("gait_contact", 1.5, "contact"),
            ("swing_clearance", 2.0, "clearance"),
            ("wrong_support", -1.0, "wrong_support"),
            ("both_feet_air", -2.0, "flight"),
            ("excess_foot_height", -0.5, "excess_height"),
        ):
            setattr(
                self.rewards,
                name,
                RewardTermCfg(
                    func=gait_terms.gait_component,
                    weight=weight,
                    params={"component": component, "asset_cfg": feet_cfg("robot"), "sensor_cfg": feet_cfg("rpo_feet")},
                ),
            )
        self.rewards.forward_speed_progress = RewardTermCfg(
            func=gait_terms.forward_progress,
            weight=1.0,
            params={"asset_cfg": feet_cfg("robot"), "sensor_cfg": feet_cfg("rpo_feet")},
        )
        self.rewards.feet_slide.weight = -0.5
        self.rewards.base_linear_velocity.weight = -4.0


@configclass
class RPOAlternatingWalkEnvCfg_PLAY(RPOAlternatingWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 100
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.observations.policy.enable_corruption = False
