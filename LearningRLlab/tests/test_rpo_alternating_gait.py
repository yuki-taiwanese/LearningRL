"""Reward exploit regressions and phase-input checkpoint compatibility (CPU only)."""

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gait = load_file("gait_math", ROOT / "LearningRLlab/LearningRLlab/tasks/locomotion/rpo/gait_math.py")
migration = load_file("gait_migration", ROOT / "tools/migrate_gait_checkpoint.py")
pytestmark = pytest.mark.static


def test_wrong_support_cannot_earn_swing_height_credit():
    # At 0.24 s the right foot should be at the swing peak; left foot supports.
    time = torch.tensor([0.24] * 4)
    contacts = torch.tensor([[True, False], [True, True], [False, False], [False, True]])
    heights = torch.tensor([[0.0, 0.03], [0.0, 0.0], [0.03, 0.03], [0.03, 0.0]])
    scores = gait.gait_scores(time, contacts, heights, torch.tensor([[0.2, 0.0]] * 4))
    torch.testing.assert_close(scores["clearance"], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(scores["contact"], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert scores["flight"].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_double_support_and_stopping_are_allowed():
    contacts = torch.tensor([[True, True], [True, True]])
    heights = torch.zeros(2, 2)
    score = gait.gait_scores(torch.tensor([0.02, 0.24]), contacts, heights, torch.tensor([[0.2, 0], [0.0, 0]]))
    assert score["contact"].tolist() == [1.0, 0.0]
    assert score["wrong_support"].tolist() == [0.0, 0.0]
    assert score["clearance"].tolist() == [0.0, 0.0]


def test_higher_is_not_always_better_and_mirror_is_symmetric():
    time = torch.tensor([0.24, 0.24])
    contacts = torch.tensor([[True, False]] * 2)
    heights = torch.tensor([[0.0, 0.03], [0.0, 0.12]])
    command = torch.tensor([[0.2, 0.0]] * 2)
    scores = gait.gait_scores(time, contacts, heights, command)
    assert scores["clearance"][0] > scores["clearance"][1]
    assert scores["excess_height"][0] == 0 < scores["excess_height"][1]
    mirrored = gait.gait_scores(time + gait.GAIT_PERIOD / 2, contacts.flip(1), heights.flip(1), command)
    for key in scores:
        torch.testing.assert_close(scores[key], mirrored[key], atol=1e-5, rtol=1e-5)


def test_flight_and_zero_command_cannot_farm_forward_progress():
    contacts = torch.tensor([[True, False], [True, True], [False, False], [False, False], [True, False]])
    actual = torch.tensor([0.2, 0.2, 0.2, -0.2, 0.2])
    target = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.0])
    torch.testing.assert_close(
        gait.supported_forward_progress(actual, target, contacts), torch.tensor([1.0, 0.5, 0.0, -1.0, 0.0])
    )


def test_sole_proxy_uses_xyzw_and_clock_wraps():
    position = torch.tensor([[[0.0, 0.0, 0.075], [0.0, 0.0, 0.075]]])
    quaternion = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]]])
    torch.testing.assert_close(
        gait.sole_centre_height(position, quaternion, torch.zeros(1)), torch.tensor([[0.03, 0.12]])
    )
    torch.testing.assert_close(
        gait.clock_observation(torch.tensor([0.0])), gait.clock_observation(torch.tensor([0.8])), atol=1e-6, rtol=1e-6
    )


def test_warm_start_preserves_network_functions_and_resets_adam(tmp_path):
    torch.manual_seed(7)
    actor = {"mlp.0.weight": torch.randn(12, 78), "mlp.0.bias": torch.randn(12)}
    critic = {"mlp.0.weight": torch.randn(12, 81), "mlp.0.bias": torch.randn(12)}
    payload = {
        "actor_state_dict": actor,
        "critic_state_dict": critic,
        "iter": 1746,
        "optimizer_state_dict": {
            "state": {0: {"exp_avg": torch.ones(12, 78)}},
            "param_groups": [{"params": [0], "lr": 1e-5}],
        },
        "infos": None,
    }
    source, dest = tmp_path / "original.pt", tmp_path / "warm.pt"
    torch.save(payload, source)
    before = source.read_bytes()
    migration.migrate(source, dest)
    result = torch.load(dest, weights_only=True)
    for key, width in (("actor_state_dict", 78), ("critic_state_dict", 81)):
        x, clock = torch.randn(10, width), torch.randn(10, 2)
        old = torch.nn.functional.linear(x, payload[key]["mlp.0.weight"], payload[key]["mlp.0.bias"])
        new = torch.nn.functional.linear(
            torch.cat((x, clock), 1), result[key]["mlp.0.weight"], result[key]["mlp.0.bias"]
        )
        torch.testing.assert_close(old, new)
    assert result["optimizer_state_dict"]["state"] == {}
    assert result["iter"] == 0
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        migration.migrate(source, dest)
    with pytest.raises(ValueError):
        migration.convert_checkpoint(result, "already-converted")
