"""RPO gait terms: explicit left/right ordering and an internally generated phase clock."""

from .gait_math import GAIT_PERIOD, gait_scores, sole_centre_height, supported_forward_progress


def _torch(value):
    return value.torch if hasattr(value, "torch") else value


def phase_clock(env):
    from .gait_math import clock_observation

    return clock_observation(env.episode_length_buf * env.step_dt)


def foot_state(env, asset_cfg, sensor_cfg):
    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    contacts = _torch(sensor.data.current_contact_time)[:, sensor_cfg.body_ids] > 0
    positions = _torch(robot.data.body_pos_w)[:, asset_cfg.body_ids]
    quaternions = _torch(robot.data.body_quat_w)[:, asset_cfg.body_ids]
    heights = sole_centre_height(positions, quaternions, env.scene.env_origins[:, 2])
    return contacts, heights


def gait_component(env, component: str, asset_cfg, sensor_cfg, command_name="base_velocity"):
    contacts, heights = foot_state(env, asset_cfg, sensor_cfg)
    command = env.command_manager.get_command(command_name)
    scores = gait_scores(env.episode_length_buf * env.step_dt, contacts, heights, command[:, :2], period=GAIT_PERIOD)
    return scores[component]


def forward_progress(env, asset_cfg, sensor_cfg, command_name="base_velocity"):
    contacts, _ = foot_state(env, asset_cfg, sensor_cfg)
    return supported_forward_progress(
        _torch(env.scene[asset_cfg.name].data.root_lin_vel_b)[:, 0],
        env.command_manager.get_command(command_name)[:, 0],
        contacts,
    )
