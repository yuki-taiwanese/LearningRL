"""Compound assets; USD imports are delayed until spawning, after Kit starts."""

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass


def spawn_compound(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from isaaclab.sim import schemas
    from isaaclab.sim.utils import clone, create_prim, get_current_stage

    @clone
    def spawn(path, spec, translation=None, orientation=None, **kwargs):
        stage = get_current_stage()
        create_prim(path, "Xform", translation=translation, orientation=orientation, stage=stage)
        for i, (size, position) in enumerate(spec.parts):
            panel = sim_utils.CuboidCfg(
                size=size,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec.color),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.9,
                    dynamic_friction=0.8,
                    restitution=0.0,
                ),
            )
            panel.func(f"{path}/panel_{i}", panel, translation=position)
        schemas.define_mass_properties(path, spec.mass_props, stage=stage)
        schemas.define_rigid_body_properties(path, spec.rigid_props, stage=stage)
        if spec.activate_contact_sensors:
            schemas.activate_contact_sensors(path, stage=stage)
        return stage.GetPrimAtPath(path)

    return spawn(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)


@configclass
class CompoundCfg(sim_utils.RigidObjectSpawnerCfg):
    func = spawn_compound
    parts: list = []
    color: tuple[float, float, float] = (0.65, 0.42, 0.20)
