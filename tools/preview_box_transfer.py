"""CPU-only scene preview; not a trained rollout or Isaac Lab validation."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAYOUT_FILE = ROOT / "LearningRLlab/LearningRLlab/tasks/locomanipulation/rpo/layout.py"


def load_layout():
    spec = importlib.util.spec_from_file_location("_preview_layout", LAYOUT_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_scene():
    layout = load_layout()
    scene = layout.LAYOUT
    source = ROOT / "robot_description/vendor/rpo_description/mjcf/rpo.xml"
    if not source.is_file():
        raise FileNotFoundError(
            "Robot assets missing. Run gitman install from the repository root."
        )
    tree = ET.parse(source)
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(source.parent.parent / "meshes"))
    world = root.find("worldbody")
    for name, pos, parts, color in [
        (
            "LeftTable",
            (scene.table_x, scene.table_y, 0),
            layout.table_parts(),
            "0.15 0.35 0.65 1",
        ),
        (
            "RightTable",
            (scene.table_x, -scene.table_y, 0),
            layout.table_parts(),
            "0.15 0.60 0.30 1",
        ),
        (
            "Box",
            scene.sources[0],
            layout.box_parts(size=scene.box_sizes[0]),
            "0.65 0.42 0.20 1",
        ),
        (
            "LargeBox",
            scene.sources[1],
            layout.box_parts(size=scene.box_sizes[1]),
            "0.82 0.57 0.25 1",
        ),
    ]:
        body = ET.SubElement(world, "body", name=name, pos=" ".join(map(str, pos)))
        if name in ("Box", "LargeBox"):
            ET.SubElement(body, "freejoint", name=f"{name}_free")
        total_volume = sum(math.prod(size) for size, _ in parts)
        for index, (size, position) in enumerate(parts):
            mass = (
                scene.box_masses[0 if name == "Box" else 1]
                * math.prod(size)
                / total_volume
                if name in ("Box", "LargeBox")
                else 2.0
            )
            ET.SubElement(
                body,
                "geom",
                name=f"{name}_{index}",
                type="box",
                size=" ".join(str(v / 2) for v in size),
                pos=" ".join(map(str, position)),
                rgba=color,
                mass=str(mass),
                contype="1",
                conaffinity="1",
                group="0",
            )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    base = model.joint("floating_base_joint").qposadr[0]
    data.qpos[base : base + 3] = [
        scene.stand_x - scene.approach_distance,
        scene.table_y,
        0.75,
    ]
    data.qpos[base + 3 : base + 7] = [1, 0, 0, 0]
    # Standing pose from robot_description/.../roboparty_rpo/joint_groups.py.
    nominal = {
        "thigh_pitch": -0.1,
        "knee": 0.3,
        "ankle_pitch": -0.2,
        "arm_pitch": 0.18,
        "elbow_pitch": 0.78,
    }
    for side, roll in [("left", 0.06), ("right", -0.06)]:
        for part, value in {**nominal, "arm_roll": roll}.items():
            data.qpos[model.joint(f"{side}_{part}_joint").qposadr[0]] = value
    mujoco.mj_forward(model, data)
    return model, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "box_transfer_scene.png")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Open an interactive 3D scene viewer (no policy).",
    )
    args = parser.parse_args()
    model, data = build_scene()
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [-0.25, 0, 0.6]
    camera.distance = 3.5
    camera.azimuth = 155
    camera.elevation = -28
    if args.live:
        from mujoco import viewer

        with viewer.launch_passive(model, data) as window:
            with window.lock():
                window.cam.lookat[:] = camera.lookat
                window.cam.distance = camera.distance
                window.cam.azimuth = camera.azimuth
                window.cam.elevation = camera.elevation
            window.sync()
            print(
                "SCENE_VIEWER_READY: static scene; no trained policy or physics stepping.",
                flush=True,
            )
            while window.is_running():
                window.sync()
                time.sleep(0.02)
        return
    with mujoco.Renderer(model, height=720, width=1100) as renderer:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
    # PNG encoder from the standard library keeps this preview dependency-free beyond MuJoCo/NumPy.
    import struct
    import zlib

    def chunk(name, payload):
        return (
            struct.pack("!I", len(payload))
            + name
            + payload
            + struct.pack("!I", zlib.crc32(name + payload))
        )

    height, width, _ = pixels.shape
    raw = b"".join(
        b"\x00" + row.tobytes() for row in np.asarray(pixels, dtype=np.uint8)
    )
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(png)
    print(f"Static scene preview (no trained policy): {args.output.resolve()}")


if __name__ == "__main__":
    main()
