"""Vectorized, simulator-independent task logic; quaternions use xyzw."""

import torch

from ..layout import LAYOUT


def rotate(q, v):
    xyz = q[..., :3]
    uv = torch.cross(xyz, v, dim=-1)
    return v + 2 * (q[..., 3:] * uv + torch.cross(xyz, uv, dim=-1))


def inverse_rotate(q, v):
    return rotate(q * q.new_tensor([-1, -1, -1, 1]), v)


def corners(position, quaternion, size):
    signs = position.new_tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    local = signs * position.new_tensor(size) / 2
    return position[:, None] + rotate(quaternion[:, None].expand(-1, 8, -1), local.expand(len(position), -1, -1))


def inside_table(points, center, size, margin=0.015):
    inside = (points[..., :2] - center[:, None, :2]).abs() <= points.new_tensor(size[:2]) / 2 - margin
    return inside.all(dim=-1).all(dim=-1)


class TransferState:
    """Monotonic stages; pulses and signed progress prevent repeated milestone farming."""

    def __init__(self, n, device, layout=LAYOUT):
        self.layout = layout
        self.phase = torch.zeros(n, dtype=torch.long, device=device)
        self.hold = torch.zeros(n, device=device)
        self.lost_contact = torch.zeros(n, device=device)
        self.previous = torch.zeros(n, device=device)
        self.initialized = torch.zeros(n, dtype=torch.bool, device=device)
        self.failed = torch.zeros(n, dtype=torch.bool, device=device)
        self.bonus = torch.zeros(n, device=device)
        self.progress = torch.zeros(n, device=device)
        self.supported = torch.zeros(n, dtype=torch.bool, device=device)

    def reset(self, ids):
        for value in (
            self.phase,
            self.hold,
            self.lost_contact,
            self.previous,
            self.initialized,
            self.failed,
            self.bonus,
            self.progress,
            self.supported,
        ):
            value[ids] = 0

    def update(self, data, dt, active=None):
        layout = self.layout
        p = self.phase.clone()
        if active is None:
            active = torch.ones_like(p, dtype=torch.bool)
        points = corners(data["box"], data["quat"], layout.box_size)
        bottom = points[..., 2].min(dim=1).values
        up = rotate(data["quat"], data["box"].new_tensor([0, 0, 1]).expand_as(data["box"]))[:, 2]
        near_source = torch.linalg.vector_norm(data["root"][:, :2] - data["source_stand"][:, :2], dim=-1) < 0.10
        near_dest = torch.linalg.vector_norm(data["root"][:, :2] - data["dest_stand"][:, :2], dim=-1) < 0.16
        hands_close = torch.linalg.vector_norm(data["hands"] - data["grasp"], dim=-1).max(dim=1).values < 0.11
        bilateral = (data["hand_force"] > layout.contact_threshold).all(dim=1) & hands_close
        lifted = (bottom > layout.table_size[2] + layout.lift_clearance) & bilateral
        above_dest = torch.linalg.vector_norm(data["box"][:, :2] - data["dest"][:, :2], dim=-1) < 0.11
        slow = torch.linalg.vector_norm(data["box_vel"], dim=-1) < 0.12
        slow &= torch.linalg.vector_norm(data["box_ang"], dim=-1) < 0.30
        supported = (
            inside_table(points, data["table"], layout.table_size)
            & ((bottom - layout.table_size[2]).abs() < 0.025)
            & (up > 0.98)
            & (data["support_force"] > layout.support_threshold)
            & slow
        )
        hand_local = inverse_rotate(data["quat"][:, None].expand(-1, 2, -1), data["hands"] - data["box"][:, None])
        gap = (hand_local.abs() - data["box"].new_tensor(layout.box_size) / 2).clamp(min=0)
        released = (torch.linalg.vector_norm(gap, dim=-1) > layout.release_clearance).all(dim=1)
        released &= (data["hand_force"] < 0.1).all(dim=1)
        ready = (
            torch.stack(
                [
                    near_source & (torch.linalg.vector_norm(data["root_vel"][:, :2], dim=-1) < 0.20),
                    bilateral,
                    lifted,
                    above_dest & near_dest & bilateral,
                    supported,
                    supported & released,
                    torch.zeros_like(bilateral),
                ],
                dim=1,
            )
            .gather(1, p[:, None])
            .squeeze(1)
        )
        hold = torch.where(ready, self.hold + dt, 0.0)
        durations = hold.new_tensor([0.25, 0.15, 0.15, 0.15, 0.25, 1.0, 1e9])[p]
        advance = (hold >= durations) & (p < 6) & active & ~self.failed
        lost = torch.where(((p == 2) | (p == 3)) & ~bilateral, self.lost_contact + dt, 0.0)
        failed = (bottom < 0.12) | (up < 0.5) | (lost > 0.4)
        failed |= ((data["root"][:, :2] - data["source_stand"][:, :2]).abs() > 3.0).any(dim=1)
        failed |= (p >= 3) & (bottom < layout.table_size[2] - 0.10)
        advance &= ~failed
        self.phase[advance] += 1
        self.hold[active] = torch.where(advance, 0.0, hold)[active]
        self.lost_contact[active] = lost[active]
        self.failed[active] |= failed[active]
        self.supported[active] = supported[active]
        self.bonus[active] = torch.where(advance, hold.new_tensor([5, 5, 10, 10, 10, 50, 0])[p], 0.0)[active]
        source_d = torch.linalg.vector_norm(data["root"][:, :2] - data["source_stand"][:, :2], dim=-1)
        hand_d = torch.linalg.vector_norm(data["hands"] - data["grasp"], dim=-1).mean(dim=1)
        carry_goal = data["dest"] + data["box"].new_tensor([0, 0, layout.lift_clearance + 0.02])
        carry_d = torch.linalg.vector_norm(data["box"] - carry_goal, dim=-1)
        dest_d = torch.linalg.vector_norm(data["box"] - data["dest"], dim=-1)
        lift_d = (layout.table_size[2] + layout.lift_clearance + 0.02 - bottom).clamp(min=0)
        potentials = torch.stack(
            [
                -source_d,
                -hand_d,
                -lift_d,
                -carry_d,
                -dest_d - torch.linalg.vector_norm(data["box_vel"], dim=-1) * 0.1,
                gap.norm(dim=-1).mean(dim=1).clamp(max=0.15),
                torch.zeros_like(source_d),
            ],
            dim=1,
        )
        potential = potentials.gather(1, p[:, None]).squeeze(1)
        progress = torch.where(self.initialized & ~advance, potential - self.previous, 0.0)
        self.progress[active] = progress[active]
        self.previous[active] = potential[active]
        self.initialized[active] = ~advance[active]
        return self.phase
