"""Task logic regression tests with synthetic sensor readings; no simulator or GPU."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.static
TASK = Path(__file__).resolve().parents[1] / "LearningRLlab/tasks/locomanipulation/rpo"


def load_logic():
    # Avoid task registration and the Isaac Lab package initializer on a CPU-only computer.
    package = "_box_transfer_test"
    for name, path in ((package, TASK), (package + ".mdp", TASK / "mdp")):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    for name, path in ((package + ".layout", TASK / "layout.py"), (package + ".mdp.core", TASK / "mdp/core.py")):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[package + ".mdp.core"], sys.modules[package + ".layout"]


core, layout = load_logic()
L = layout.LAYOUT


@pytest.fixture
def data():
    def vector(value):
        return torch.tensor([value], dtype=torch.float32)

    box = vector(L.source)
    hands = box[:, None] + vector([[-0.03, L.box_size[1] / 2 + 0.025, 0], [-0.03, -L.box_size[1] / 2 - 0.025, 0]])
    return dict(
        box=box,
        quat=vector([0, 0, 0, 1]),
        root=vector([L.stand_x, L.table_y, 0.75]),
        root_vel=vector([0, 0, 0]),
        box_vel=vector([0, 0, 0]),
        box_ang=vector([0, 0, 0]),
        hands=hands,
        grasp=hands.clone(),
        hand_force=vector([1, 1]),
        support_force=torch.tensor([0.0]),
        source_stand=vector([L.stand_x, L.table_y, 0]),
        dest_stand=vector([L.stand_x, -L.table_y, 0]),
        dest=vector(L.destination),
        table=vector([L.table_x, -L.table_y, 0]),
    )


def tick(state, data, seconds):
    for _ in range(math.ceil(seconds / 0.02)):
        state.update(data, 0.02)


def move_box(data, position):
    delta = position - data["box"]
    data["box"] = position.clone()
    data["hands"] += delta[:, None]
    data["grasp"] += delta[:, None]


def test_complete_sequence_and_one_shot_success(data):
    s = core.TransferState(1, "cpu")
    tick(s, data, 0.28)
    assert s.phase.item() == 1
    tick(s, data, 0.16)
    assert s.phase.item() == 2
    move_box(data, data["box"] + torch.tensor([[0, 0, 0.11]]))
    tick(s, data, 0.16)
    assert s.phase.item() == 3
    data["root"][:, 1] = -L.table_y
    move_box(data, data["dest"] + torch.tensor([[0, 0, 0.11]]))
    tick(s, data, 0.16)
    assert s.phase.item() == 4
    move_box(data, data["dest"])
    data["support_force"][:] = L.box_mass * 9.81
    tick(s, data, 0.28)
    assert s.phase.item() == 5
    tick(s, data, 1.1)
    assert s.phase.item() == 5, "Touching the box must not count as release"
    data["hands"][:, 0, 1] += 0.15
    data["hands"][:, 1, 1] -= 0.15
    data["hand_force"][:] = 0
    tick(s, data, 0.8)
    assert s.phase.item() == 5
    tick(s, data, 0.22)
    assert s.phase.item() == 6
    assert s.bonus.item() == 50
    tick(s, data, 0.02)
    assert s.bonus.item() == 0


def test_no_lift_no_carry(data):
    s = core.TransferState(1, "cpu")
    tick(s, data, 0.5)
    assert s.phase.item() == 2
    move_box(data, data["dest"])
    tick(s, data, 1.0)
    assert s.phase.item() == 2


@pytest.mark.parametrize("force", [[0, 0], [1, 0], [0, 1]])
def test_requires_both_forearms(data, force):
    s = core.TransferState(1, "cpu")
    s.phase[:] = 1
    data["hand_force"][:] = torch.tensor(force)
    tick(s, data, 0.5)
    assert s.phase.item() == 1


def test_drop_terminates(data):
    s = core.TransferState(1, "cpu")
    s.phase[:] = 3
    data["box"][:, 2] = 0.1
    tick(s, data, 0.02)
    assert s.failed.item()


def test_lost_grip_terminates_carry(data):
    s = core.TransferState(1, "cpu")
    s.phase[:] = 3
    data["hand_force"][:] = 0
    tick(s, data, 0.44)
    assert s.failed.item()


@pytest.mark.parametrize("problem", ["hover", "overhang", "fast", "tilted"])
def test_bad_placement_never_passes(data, problem):
    s = core.TransferState(1, "cpu")
    s.phase[:] = 4
    move_box(data, data["dest"])
    data["support_force"][:] = 4.905
    if problem == "hover":
        data["support_force"][:] = 0
    elif problem == "overhang":
        move_box(data, data["box"] + torch.tensor([[-0.10, 0, 0]]))
    elif problem == "fast":
        data["box_vel"][:, 0] = 0.5
    else:
        data["quat"][:] = torch.tensor([[math.sin(0.25), 0, 0, math.cos(0.25)]])
    tick(s, data, 1.2)
    assert s.phase.item() != 5


def test_release_timer_requires_continuous_support(data):
    s = core.TransferState(1, "cpu")
    s.phase[:] = 5
    move_box(data, data["dest"])
    data["support_force"][:] = 4.905
    data["hands"] += torch.tensor([[[0, 0.2, 0], [0, -0.2, 0]]])
    data["hand_force"][:] = 0
    tick(s, data, 0.7)
    data["support_force"][:] = 0
    tick(s, data, 0.02)
    data["support_force"][:] = 4.905
    tick(s, data, 0.7)
    assert s.phase.item() == 5


def test_progress_does_not_reward_back_and_forth(data):
    s = core.TransferState(1, "cpu")
    data["root"][:, 0] = -1.4
    s.update(data, 0.02)
    data["root"][:, 0] += 0.05
    s.update(data, 0.02)
    gain = s.progress.clone()
    data["root"][:, 0] -= 0.05
    s.update(data, 0.02)
    assert torch.allclose(gain + s.progress, torch.zeros(1), atol=1e-6)


def test_selective_reset_and_inactive_environments(data):
    d = {key: value.repeat((2,) + (1,) * (value.ndim - 1)) for key, value in data.items()}
    s = core.TransferState(2, "cpu")
    for _ in range(20):
        s.update(d, 0.02, active=torch.tensor([True, False]))
    assert s.phase.tolist() == [1, 0]
    s.reset([0])
    assert s.phase.tolist() == [0, 0]
    assert not s.initialized.any()
    assert not s.bonus.any()


def test_xyzw_quaternion_geometry():
    q = torch.tensor([[0, 0, math.sqrt(0.5), math.sqrt(0.5)]])
    v = torch.tensor([[1.0, 0, 0]])
    assert torch.allclose(core.rotate(q, v), torch.tensor([[0.0, 1, 0]]), atol=1e-6)
    assert torch.allclose(core.inverse_rotate(q, core.rotate(q, v)), v, atol=1e-6)


def test_open_box_panels_have_expected_volume():
    volume = sum(math.prod(size) for size, _ in layout.box_parts())
    x, y, z = L.box_size
    t = L.wall
    assert volume == pytest.approx(x * y * z - (x - 2 * t) * (y - 2 * t) * (z - t))
