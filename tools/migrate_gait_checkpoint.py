"""Warm-start the phase-conditioned gait task from a trusted RSL-RL checkpoint.

Appends two zero-weight clock inputs, preserving the old actor/critic functions.
Adam moments and iteration count are reset. Never overwrite the original file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import torch


def convert_checkpoint(payload: dict, source_sha256: str) -> dict:
    result = copy.deepcopy(payload)
    for key, inputs in (("actor_state_dict", 78), ("critic_state_dict", 81)):
        weights = result[key]["mlp.0.weight"]
        if weights.ndim != 2 or weights.shape[1] != inputs:
            raise ValueError(
                f"Expected {key} input width {inputs}; got {tuple(weights.shape)}"
            )
        result[key]["mlp.0.weight"] = torch.cat(
            (weights, weights.new_zeros(weights.shape[0], 2)), dim=1
        )
    optimizer = result.get("optimizer_state_dict")
    if not isinstance(optimizer, dict) or "param_groups" not in optimizer:
        raise ValueError("Expected an RSL-RL optimizer_state_dict with param_groups")
    optimizer["state"] = {}
    for group in optimizer["param_groups"]:
        group["lr"] = 1.0e-3
        if "initial_lr" in group:
            group["initial_lr"] = 1.0e-3
    result["iter"] = 0
    info = result.get("infos")
    result["infos"] = info if isinstance(info, dict) else {"previous_infos": info}
    result["infos"]["gait_warm_start"] = {
        "source_sha256": source_sha256,
        "actor_inputs": [78, 80],
        "critic_inputs": [81, 83],
        "clock_weights_initially_zero": True,
        "optimizer_moments_reset": True,
        "not_a_trained_alternating_policy": True,
    }
    return result


def migrate(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if source == destination or destination.exists():
        raise FileExistsError(
            "Choose a new destination; existing checkpoints are preserved"
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = torch.load(source, map_location="cpu", weights_only=True)
    converted = convert_checkpoint(payload, digest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    torch.save(converted, partial)
    restored = torch.load(partial, map_location="cpu", weights_only=True)
    if restored["actor_state_dict"]["mlp.0.weight"].shape[1] != 80:
        raise RuntimeError("Saved actor input width failed verification")
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Source changed during conversion")
    partial.replace(destination)
    return converted["infos"]["gait_warm_start"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(migrate(args.source, args.destination))
