"""Runtime command and shared step snapshot, evaluated at most once per environment step."""

import torch

from isaaclab.managers import CommandTerm

from ..layout import LAYOUT
from .core import TransferState, inverse_rotate, rotate


class TransferCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.box = env.scene["box"]
        self.hands_sensor = env.scene["box_contact"]
        names = ["left_elbow_pitch_link", "right_elbow_pitch_link"]
        self.hand_ids, _ = self.robot.find_bodies(names, preserve_order=True)
        self.contact_ids, _ = self.hands_sensor.find_bodies(names, preserve_order=True)
        self.state = TransferState(self.num_envs, self.device)
        self.last_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.data = {}
        for name in ("stage", "success", "failure", "destination_error"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        self.refresh()
        return self._command

    def _resample_command(self, env_ids):
        self.state.reset(env_ids)
        self.last_step[env_ids] = -1
        self._command[env_ids] = 0

    def _update_metrics(self):
        self.refresh()

    def _update_command(self):
        self.refresh()

    def refresh(self):
        step = self._env.common_step_counter
        active = self.last_step != step
        if not active.any():
            return
        layout = LAYOUT
        root = self.robot.data.root_pos_w.torch
        origins = self._env.scene.env_origins
        box = self.box.data.root_pos_w.torch
        quat = self.box.data.root_quat_w.torch
        hand_q = self.robot.data.body_quat_w.torch[:, self.hand_ids]
        # Centers transcribed from vendor rpo.urdf elbow_pitch collision origins.
        offsets = root.new_tensor([[0.14, -0.02, 0], [0.14, 0.02, 0]]).expand(self.num_envs, -1, -1)
        hands = self.robot.data.body_pos_w.torch[:, self.hand_ids] + rotate(hand_q, offsets)
        grasp_local = root.new_tensor(
            [[-0.03, layout.box_size[1] / 2 + 0.025, 0], [-0.03, -layout.box_size[1] / 2 - 0.025, 0]]
        )
        grasp = box[:, None] + rotate(quat[:, None].expand(-1, 2, -1), grasp_local.expand(self.num_envs, -1, -1))
        matrix = self.hands_sensor.data.force_matrix_w
        support = self._env.scene["right_support"].data.force_matrix_w
        if matrix is None or support is None:
            raise RuntimeError("Box transfer requires filtered hand-box and box-right-table contact forces.")
        forces = matrix.torch[:, self.contact_ids].sum(dim=2).norm(dim=-1)
        support_force = support.torch.sum(dim=(1, 2))[:, 2].clamp(min=0)
        self.data = {
            "root": root,
            "root_vel": self.robot.data.root_lin_vel_w.torch,
            "box": box,
            "quat": quat,
            "box_vel": self.box.data.root_lin_vel_w.torch,
            "box_ang": self.box.data.root_ang_vel_w.torch,
            "hands": hands,
            "grasp": grasp,
            "hand_force": forces,
            "support_force": support_force,
            "source_stand": origins + root.new_tensor([layout.stand_x, layout.table_y, 0]),
            "dest_stand": origins + root.new_tensor([layout.stand_x, -layout.table_y, 0]),
            "dest": origins + root.new_tensor(layout.destination),
            "table": origins + root.new_tensor([layout.table_x, -layout.table_y, 0]),
        }
        self.state.update(self.data, self._env.step_dt, active)
        self.last_step[active] = step
        p = self.state.phase
        goal = torch.where((p == 0)[:, None], self.data["source_stand"], self.data["dest_stand"])
        delta = goal - root
        yaw = self.yaw()
        delta = inverse_rotate(yaw, delta)
        self._command[:, :2] = (delta[:, :2] * 1.2).clamp(-0.35, 0.35)
        heading = torch.atan2(2 * (yaw[:, 3] * yaw[:, 2]), 1 - 2 * yaw[:, 2].square())
        self._command[:, 2] = (-heading * 1.5).clamp(-0.5, 0.5)
        self._command[(p != 0) & (p != 3)] = 0
        self.metrics["stage"][:] = p.float()
        self.metrics["success"][:] = (p == 6).float()
        self.metrics["failure"][:] = self.state.failed.float()
        self.metrics["destination_error"][:] = (box - self.data["dest"]).norm(dim=-1)

    def yaw(self):
        q = self.robot.data.root_quat_w.torch
        angle = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1].square() + q[:, 2].square()))
        out = torch.zeros_like(q)
        out[:, 2] = torch.sin(angle / 2)
        out[:, 3] = torch.cos(angle / 2)
        return out
