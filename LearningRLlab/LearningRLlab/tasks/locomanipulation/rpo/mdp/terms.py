"""Observations, rewards, resets and terminations for box transfer."""

import torch

from ..layout import LAYOUT
from .core import inverse_rotate


def transfer(env):
    term = env.command_manager.get_term("base_velocity")
    term.refresh()
    return term


def task_observation(env):
    term = transfer(env)
    d = term.data
    yaw = term.yaw()
    relative_q = d["quat"].clone()
    # Expose the box rotation as its three world axes expressed in the root yaw frame.
    from .core import rotate

    axes = torch.eye(3, device=env.device).expand(env.num_envs, -1, -1)
    box_axes = rotate(relative_q[:, None].expand(-1, 3, -1), axes)
    box_axes = inverse_rotate(yaw[:, None].expand(-1, 3, -1), box_axes).flatten(1)
    return torch.cat(
        [
            inverse_rotate(yaw, d["box"] - d["root"]),
            box_axes,
            inverse_rotate(yaw, d["box_vel"]),
            inverse_rotate(yaw, d["box_ang"]),
            inverse_rotate(yaw, d["dest"] - d["root"]),
            inverse_rotate(yaw, d["source_stand"] - d["root"]),
            inverse_rotate(yaw, d["dest_stand"] - d["root"]),
            inverse_rotate(yaw[:, None].expand(-1, 2, -1), d["hands"] - d["root"][:, None]).flatten(1),
            (d["hand_force"] / 10).clamp(max=5),
            torch.nn.functional.one_hot(term.state.phase, 15).float(),
            d["size"],
            term.state.delivered.float(),
            inverse_rotate(yaw, d["home"] - d["root"]),
            inverse_rotate(yaw[:, None].expand(-1, 2, -1), d["all_boxes"] - d["root"][:, None]).flatten(1),
            term.state.hold[:, None],
            term.state.lost_contact[:, None],
            term.state.supported[:, None].float(),
        ],
        dim=-1,
    )


def task_reward(env):
    term = transfer(env)
    s = term.state
    # RewardManager multiplies rates by step_dt: event rewards are divided by dt exactly once.
    return (s.bonus + 8.0 * s.progress) / env.step_dt


def task_success(env):
    return transfer(env).state.done


def task_failure(env):
    return transfer(env).state.failed


def terminal_failure(env):
    return env.termination_manager.terminated & ~task_success(env)


def reset_task(env, env_ids, approach_distance=1.0):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    layout = LAYOUT
    robot = env.scene["robot"]
    origins = env.scene.env_origins[env_ids]
    state = robot.data.default_root_state.torch[env_ids].clone()
    state[:, :3] = origins + state.new_tensor([layout.stand_x - approach_distance, layout.table_y, 0.75])
    state[:, 3:7] = state.new_tensor([0, 0, 0, 1])
    state[:, 7:] = 0
    robot.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)
    for index, name in enumerate(("box", "large_box")):
        box = env.scene[name]
        state = box.data.default_root_state.torch[env_ids].clone()
        state[:, :3] = origins + state.new_tensor(layout.sources[index])
        state[:, 3:7] = state.new_tensor([0, 0, 0, 1])
        state[:, 7:] = 0
        box.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        box.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)
