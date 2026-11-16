"""Designed classroom scene parameters, not measured robot specifications (SI units)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    table_size: tuple[float, float, float] = (0.70, 1.00, 0.72)
    table_x: float = 0.25
    table_y: float = 0.65
    table_thickness: float = 0.04
    box_size: tuple[float, float, float] = (0.24, 0.30, 0.20)
    wall: float = 0.012
    box_mass: float = 0.50
    large_box_size: tuple[float, float, float] = (0.30, 0.36, 0.24)
    large_box_mass: float = 0.70
    box_spacing: float = 0.23
    stand_x: float = -0.25
    approach_distance: float = 1.0
    # Bookkeeping horizon only. The environment disables the time-out termination.
    episode_seconds: float = 120.0
    contact_threshold: float = 0.5
    lift_clearance: float = 0.08
    release_clearance: float = 0.06
    support_threshold: float = 1.0

    @property
    def box_z(self):
        return self.table_size[2] + self.box_size[2] / 2

    @property
    def source(self):
        return self.sources[0]

    @property
    def destination(self):
        return self.destinations[0]

    @property
    def box_sizes(self):
        return (self.box_size, self.large_box_size)

    @property
    def box_masses(self):
        return (self.box_mass, self.large_box_mass)

    @property
    def sources(self):
        return tuple(
            (x, self.table_y + dy, self.table_size[2] + size[2] / 2)
            for x, dy, size in zip((0.05, 0.08), (self.box_spacing, -self.box_spacing), self.box_sizes)
        )

    @property
    def destinations(self):
        return tuple((x, y - 2 * self.table_y, z) for x, y, z in self.sources)

    @property
    def source_stands(self):
        return tuple((x - 0.30, y, 0.0) for x, y, _ in self.sources)

    @property
    def destination_stands(self):
        return tuple((x - 0.30, y, 0.0) for x, y, _ in self.destinations)

    @property
    def home(self):
        return (self.stand_x, self.table_y, 0.0)


LAYOUT = Layout()


def box_parts(layout=LAYOUT, size=None):
    """Five non-overlapping panels; one dynamic compound body with an open top."""
    x, y, z = layout.box_size if size is None else size
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
