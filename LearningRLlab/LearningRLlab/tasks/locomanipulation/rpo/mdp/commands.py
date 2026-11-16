"""Runtime command and shared step snapshot, evaluated at most once per environment step."""

import torch

from isaaclab.managers import CommandTerm

from ..layout import LAYOUT
from .core import RoundTripState, corners, inside_table, inverse_rotate, rotate


class TransferCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.boxes = [env.scene["box"], env.scene["large_box"]]
        self.hands_sensor = env.scene["box_contact"]
        names = ["left_elbow_pitch_link", "right_elbow_pitch_link"]
        self.hand_ids, _ = self.robot.find_bodies(names, preserve_order=True)
        self.contact_ids, _ = self.hands_sensor.find_bodies(names, preserve_order=True)
        self.state = RoundTripState(self.num_envs, self.device)
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
        self.data = self.snapshot()
        previous_box = self.state.box_index.clone()
        self.state.update(self.data, self._env.step_dt, active)
        # Returning home switches the active box; observations must switch in this same step.
        if (previous_box != self.state.box_index).any():
            self.data = self.snapshot()
        self.last_step[active] = step
        p = self.state.leg.phase
        root = self.data["root"]
        goal = torch.where((p == 0)[:, None], self.data["source_stand"], self.data["dest_stand"])
        goal = torch.where((p == 6)[:, None], self.data["home"], goal)
        delta = goal - root
        yaw = self.yaw()
        delta = inverse_rotate(yaw, delta)
        self._command[:, :2] = (delta[:, :2] * 1.2).clamp(-0.35, 0.35)
        heading = torch.atan2(2 * (yaw[:, 3] * yaw[:, 2]), 1 - 2 * yaw[:, 2].square())
        self._command[:, 2] = (-heading * 1.5).clamp(-0.5, 0.5)
        self._command[((p != 0) & (p != 3) & (p != 6)) | self.state.done] = 0
        self.metrics["stage"][:] = self.state.phase.float()
        self.metrics["success"][:] = self.state.done.float()
        self.metrics["failure"][:] = self.state.failed.float()
        self.metrics["destination_error"][:] = (self.data["box"] - self.data["dest"]).norm(dim=-1)

    def snapshot(self):
        layout = LAYOUT
        root = self.robot.data.root_pos_w.torch
        origins = self._env.scene.env_origins
        rows = torch.arange(self.num_envs, device=self.device)
        index = self.state.box_index
        positions = torch.stack([box.data.root_pos_w.torch for box in self.boxes], dim=1)
        quaternions = torch.stack([box.data.root_quat_w.torch for box in self.boxes], dim=1)
        velocities = torch.stack([box.data.root_lin_vel_w.torch for box in self.boxes], dim=1)
        angular = torch.stack([box.data.root_ang_vel_w.torch for box in self.boxes], dim=1)
        sizes = root.new_tensor(layout.box_sizes)[index]
        box = positions[rows, index]
        quat = quaternions[rows, index]
        hand_q = self.robot.data.body_quat_w.torch[:, self.hand_ids]
        # Centers transcribed from vendor rpo.urdf elbow_pitch collision origins.
        offsets = root.new_tensor([[0.14, -0.02, 0], [0.14, 0.02, 0]]).expand(self.num_envs, -1, -1)
        hands = self.robot.data.body_pos_w.torch[:, self.hand_ids] + rotate(hand_q, offsets)
        grasp_local = torch.zeros(self.num_envs, 2, 3, device=self.device)
        grasp_local[:, :, 0] = -0.03
        grasp_local[:, 0, 1] = sizes[:, 1] / 2 + 0.025
        grasp_local[:, 1, 1] = -sizes[:, 1] / 2 - 0.025
        grasp = box[:, None] + rotate(quat[:, None].expand(-1, 2, -1), grasp_local)
        matrix = self.hands_sensor.data.force_matrix_w
        supports = [self._env.scene[name].data.force_matrix_w for name in ("right_support", "large_right_support")]
        if matrix is None or any(support is None for support in supports):
            raise RuntimeError("Box transfer requires filtered hand-box and box-right-table contact forces.")
        all_forces = matrix.torch[:, self.contact_ids].norm(dim=-1).transpose(1, 2)
        support_forces = torch.stack([s.torch.sum(dim=(1, 2))[:, 2].clamp(min=0) for s in supports], dim=1)
        table = origins + root.new_tensor([layout.table_x, -layout.table_y, 0])
        all_supported = []
        all_failed = []
        for i, size in enumerate(layout.box_sizes):
            points = corners(positions[:, i], quaternions[:, i], size)
            bottom = points[..., 2].min(dim=1).values
            up = rotate(quaternions[:, i], root.new_tensor([0, 0, 1]).expand_as(root))[:, 2]
            supported = inside_table(points, table, layout.table_size)
            supported &= (bottom - layout.table_size[2]).abs() < 0.025
            supported &= (up > 0.98) & (support_forces[:, i] > layout.support_threshold)
            supported &= (velocities[:, i].norm(dim=-1) < 0.12) & (angular[:, i].norm(dim=-1) < 0.30)
            all_supported.append(supported)
            all_failed.append(
                (bottom < 0.12) | (up < 0.5) | ((positions[:, i, :2] - origins[:, :2]).abs() > 3).any(dim=1)
            )
        return {
            "root": root,
            "root_vel": self.robot.data.root_lin_vel_w.torch,
            "box": box,
            "quat": quat,
            "box_vel": velocities[rows, index],
            "box_ang": angular[rows, index],
            "size": sizes,
            "all_boxes": positions,
            "all_supported": torch.stack(all_supported, dim=1),
            "any_box_failed": torch.stack(all_failed, dim=1).any(dim=1),
            "all_hand_force": all_forces,
            "hands": hands,
            "grasp": grasp,
            "hand_force": all_forces[rows, index],
            "support_force": support_forces[rows, index],
            "source_stand": origins + root.new_tensor(layout.source_stands)[index],
            "dest_stand": origins + root.new_tensor(layout.destination_stands)[index],
            "home": origins + root.new_tensor(layout.home),
            "dest": origins + root.new_tensor(layout.destinations)[index],
            "table": table,
        }

    def yaw(self):
        q = self.robot.data.root_quat_w.torch
        angle = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1].square() + q[:, 2].square()))
        out = torch.zeros_like(q)
        out[:, 2] = torch.sin(angle / 2)
        out[:, 3] = torch.cos(angle / 2)
        return out
