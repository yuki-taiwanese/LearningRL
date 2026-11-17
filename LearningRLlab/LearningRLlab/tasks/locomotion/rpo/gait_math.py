"""Simulator-independent shaping for slow alternating walking on flat ground."""

import math

import torch

GAIT_PERIOD = 0.8
STANCE_RATIO = 0.60
SOLE_OFFSET = 0.045  # RPO foot-body origin to the nominal sole centre, metres.
SWING_CLEARANCE = 0.03  # Provisional sole-centre clearance, not ankle-link height.


def clock_observation(time: torch.Tensor, period: float = GAIT_PERIOD) -> torch.Tensor:
    angle = 2 * math.pi * time / period
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def gait_targets(time: torch.Tensor, period=GAIT_PERIOD, stance_ratio=STANCE_RATIO, clearance=SWING_CLEARANCE):
    """Half-cycle offsets with 20% double support and smooth swing-height targets."""
    phase = (time[:, None] / period + time.new_tensor([0.0, 0.5])) % 1.0
    stance = phase < stance_ratio
    swing_phase = ((phase - stance_ratio) / (1 - stance_ratio)).clamp(0, 1)
    height = clearance * torch.sin(math.pi * swing_phase).square()
    return stance, height


def sole_centre_height(foot_position, foot_quaternion_xyzw, ground_z, offset=SOLE_OFFSET):
    """Tilt-aware sole-centre proxy above a flat plane; not the lowest mesh vertex."""
    q = foot_quaternion_xyzw
    vertical_axis = 1 - 2 * (q[..., 0].square() + q[..., 1].square())
    return foot_position[..., 2] - offset * vertical_axis - ground_z[:, None]


def gait_scores(
    time,
    contacts,
    heights,
    command_xy,
    period=GAIT_PERIOD,
    stance_ratio=STANCE_RATIO,
    clearance=SWING_CLEARANCE,
    height_std=0.012,
    max_clearance=0.07,
):
    """No partial positive gait/height credit for synchronized hopping or planted feet."""
    stance, target = gait_targets(time, period, stance_ratio, clearance)
    moving = command_xy.norm(dim=-1) > 0.05
    matched = (contacts == stance).all(dim=-1)
    swing = ~stance
    height_score = torch.exp(-((heights - target) / height_std).square())
    clearance_score = (height_score * swing).sum(dim=-1)
    clearance_score *= matched & moving
    return {
        "contact": (matched & moving).float(),
        "clearance": clearance_score,
        "wrong_support": (contacts != stance).float().mean(dim=-1) * moving,
        "flight": (~contacts.any(dim=-1)).float(),
        "excess_height": (((heights - max_clearance).clamp_min(0) / clearance).clamp_max(3).square()).mean(dim=-1),
    }


def supported_forward_progress(speed_x, target_x, contacts):
    """No positive progress credit in flight; retain backwards penalties and bounded speed."""
    raw = (speed_x / target_x.clamp_min(0.05)).clamp(-1, 1)
    supports = contacts.sum(dim=-1)
    scale = torch.where(supports == 1, 1.0, torch.where(supports == 2, 0.5, 0.0))
    return torch.where(raw > 0, raw * scale, raw) * (target_x > 0.05)
