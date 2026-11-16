"""Check nominal standing reach with bounded numeric IK; this is not a grasp policy."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from preview_box_transfer import ROOT, build_scene, load_layout


def main():
    layout = load_layout().LAYOUT
    model, data = build_scene()
    urdf = ET.parse(
        ROOT / "robot_description/vendor/rpo_description/urdf/rpo.urdf"
    ).getroot()
    limits = {
        joint.get("name"): [
            float(joint.find("limit").get(k)) for k in ("lower", "upper")
        ]
        for joint in urdf.findall("joint")
        if joint.find("limit") is not None
    }
    root = model.joint("floating_base_joint").qposadr[0]
    data.qpos[root : root + 3] = [layout.stand_x, layout.table_y, 0.75]
    rng = np.random.default_rng(42)
    worst = 0.0
    for box_index, (source, size, stand) in enumerate(
        zip(layout.sources, layout.box_sizes, layout.source_stands)
    ):
        data.qpos[root : root + 3] = [stand[0], stand[1], 0.75]
        for lift in (0.0, layout.lift_clearance + 0.02):
            for side, sign in [("left", 1), ("right", -1)]:
                joints = [
                    model.joint(f"{side}_{part}_joint")
                    for part in [
                        "arm_pitch",
                        "arm_roll",
                        "arm_yaw",
                        "elbow_pitch",
                        "elbow_yaw",
                    ]
                ]
                addresses = np.array([joint.qposadr[0] for joint in joints])
                bounds = np.array([limits[joint.name] for joint in joints])
                midpoint = bounds.mean(axis=1)
                half_range = (bounds[:, 1] - bounds[:, 0]) * 0.90 / 2
                bounds = np.column_stack([midpoint - half_range, midpoint + half_range])
                body = model.body(f"{side}_elbow_pitch_link").id
                offset = np.array([0.14, -sign * 0.02, 0])
                target = np.array(source) + [
                    -0.03,
                    sign * (size[1] / 2 + 0.025),
                    lift,
                ]

                def position(q, addresses=addresses, body=body, offset=offset):
                    data.qpos[addresses] = q
                    mujoco.mj_forward(model, data)
                    return data.xpos[body] + data.xmat[body].reshape(3, 3) @ offset

                best = math.inf
                best_q = None
                for _ in range(12):
                    q = rng.uniform(bounds[:, 0], bounds[:, 1])
                    for _ in range(150):
                        current = position(q)
                        error = target - current
                        if np.linalg.norm(error) < best:
                            best = float(np.linalg.norm(error))
                            best_q = q.copy()
                        if best < 0.002:
                            break
                        jacobian = np.column_stack(
                            [
                                (position(q + np.eye(5)[j] * 1e-4) - current) / 1e-4
                                for j in range(5)
                            ]
                        )
                        dq = jacobian.T @ np.linalg.solve(
                            jacobian @ jacobian.T + np.eye(3) * 0.002, error
                        )
                        q = np.clip(
                            q + np.clip(dq, -0.12, 0.12), bounds[:, 0], bounds[:, 1]
                        )
                    if best < 0.002:
                        break
                worst = max(worst, best)
                print(
                    f"box={box_index}, {side}, lift={lift:.2f} m: reach error={best:.4f} m; joint target={best_q}"
                )
    if worst > 0.02:
        raise SystemExit(
            "The proposed grasp points are outside nominal standing reach."
        )
    print(
        "Standing reach passed. Collision-free motion and stable grasp still require simulator validation."
    )


if __name__ == "__main__":
    main()
