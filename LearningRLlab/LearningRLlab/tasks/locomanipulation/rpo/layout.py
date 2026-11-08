"""Designed classroom scene parameters, not measured robot specifications (SI units)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    table_size: tuple[float, float, float] = (0.70, 0.64, 0.72)
    table_x: float = 0.25
    table_y: float = 0.65
    table_thickness: float = 0.04
    box_size: tuple[float, float, float] = (0.24, 0.30, 0.20)
    wall: float = 0.012
    box_mass: float = 0.50
    stand_x: float = -0.25
    approach_distance: float = 1.0
    episode_seconds: float = 35.0
    contact_threshold: float = 0.5
    lift_clearance: float = 0.08
    release_clearance: float = 0.06
    support_threshold: float = 1.0

    @property
    def box_z(self):
        return self.table_size[2] + self.box_size[2] / 2

    @property
    def source(self):
        return (0.05, self.table_y, self.box_z)

    @property
    def destination(self):
        return (0.05, -self.table_y, self.box_z)


LAYOUT = Layout()


def box_parts(layout=LAYOUT):
    """Five non-overlapping panels; one dynamic compound body with an open top."""
    x, y, z = layout.box_size
    t = layout.wall
    return [
        ((x, y, t), (0, 0, -(z - t) / 2)),
        ((t, y, z - t), (-(x - t) / 2, 0, t / 2)),
        ((t, y, z - t), ((x - t) / 2, 0, t / 2)),
        ((x - 2 * t, t, z - t), (0, -(y - t) / 2, t / 2)),
        ((x - 2 * t, t, z - t), (0, (y - t) / 2, t / 2)),
    ]


def table_parts(layout=LAYOUT):
    x, y, h = layout.table_size
    t = layout.table_thickness
    leg = 0.05
    parts = [((x, y, t), (0, 0, h - t / 2))]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(((leg, leg, h - t), (sx * (x / 2 - leg), sy * (y / 2 - leg), (h - t) / 2)))
    return parts
