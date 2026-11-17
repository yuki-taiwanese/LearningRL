"""Optional reuse of the course importer's inertia-corrected RPO USD, without Kit."""

import os
from pathlib import Path


def use_cached_rpo_usd(robot):
    """Preserve authored physics and actuator settings; never generate or fake a model."""
    cached_usd = os.environ.get("LEARNINGRL_RPO_USD")
    if not cached_usd:
        return
    if not Path(cached_usd).is_file():
        raise FileNotFoundError(f"Preconverted RPO model not found: {cached_usd}")

    import isaaclab.sim as sim_utils

    source = robot.spawn
    robot.spawn = sim_utils.UsdFileCfg(
        usd_path=cached_usd,
        activate_contact_sensors=source.activate_contact_sensors,
        rigid_props=source.rigid_props,
        articulation_props=source.articulation_props,
        joint_drive_props=source.joint_drive_props,
        collision_props=source.collision_props,
        mass_props=source.mass_props,
    )
