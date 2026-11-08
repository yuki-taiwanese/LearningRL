"""Run the README's local gates with the current Python; stop at the first failure."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    commands = [
        ["-m", "ruff", "check", "."],
        ["-m", "ruff", "format", "--check", "."],
        ["-m", "pytest", "-m", "static", "LearningRLlab/tests", "-q"],
        ["-m", "pytest", "robot_description/tests", "LearningRLsim/tests", "-q"],
    ]
    temp = ROOT / ".local-check-tmp"
    temp.mkdir(exist_ok=True)
    environment = {
        **os.environ,
        "TMP": str(temp),
        "TEMP": str(temp),
        "TMPDIR": str(temp),
    }
    for args in commands:
        print("Running:", sys.executable, *args, flush=True)
        result = subprocess.run(
            [sys.executable, *args], cwd=ROOT, env=environment, check=False
        )
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
