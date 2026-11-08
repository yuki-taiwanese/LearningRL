"""Windows extended-key decoding without requiring an interactive console."""

from types import SimpleNamespace

import pytest

from LearningRLsim.commands import keyboard

pytestmark = pytest.mark.static


@pytest.mark.parametrize(
    "keys, expected", [(["w"], "w"), (["\xe0", "H"], "w"), (["\x00", "K"], "q"), (["\xe0", "Q"], "")]
)
def test_windows_key_decoding(monkeypatch, keys, expected):
    source = iter(keys)
    monkeypatch.setattr(keyboard, "msvcrt", SimpleNamespace(getwch=lambda: next(source)), raising=False)
    assert keyboard._windows_key() == expected
