#!/usr/bin/env bash
# Run inside the TA's pinned Isaac Lab container, not Windows PowerShell.
set -euo pipefail
cd "${ISAACLAB_PATH:-/workspace/isaaclab}"
mode="${1:-smoke}"
viz_args=(--viz kit --enable_cameras --headless)
if [[ -n "${LEARNINGRL_RPO_USD:-}" ]]; then
  # The supplied USD must come from the course importer, including inertia fixes.
  viz_args=(--viz none)
fi
case "$mode" in
  smoke)
    ./isaaclab.sh -p -c 'import importlib,sys; importlib.import_module("LearningRLlab.tasks.locomanipulation.rpo.box_env_cfg"); offenders=[m for m in sys.modules if m == "pxr" or m.startswith("pxr.")]; print("Pre-Kit USD imports:",offenders); assert not offenders'
    ./isaaclab.sh -p source/LearningRLlab/scripts/train.py \
      --task LearningRL-RPO-BoxTransfer-v0 --num_envs 8 --max_iterations 2 \
      "${viz_args[@]}" physics=newton_mjwarp
    ;;
  train)
    ./isaaclab.sh -p source/LearningRLlab/scripts/train.py \
      --task LearningRL-RPO-BoxTransfer-v0 --num_envs 512 --max_iterations 20000 \
      "${viz_args[@]}" physics=newton_mjwarp
    ;;
  video)
    checkpoint="${2:?Supply the trained checkpoint path after video}"
    ./isaaclab.sh -p source/LearningRLlab/scripts/play.py \
      --task LearningRL-RPO-BoxTransfer-v0-PLAY --num_envs 1 --checkpoint "$checkpoint" \
      --viz kit --enable_cameras --headless --video --video_length 1750 --no-export \
      physics=newton_mjwarp
    ;;
  *) echo "Usage: bash tools/train_box_transfer.sh smoke|train|video [checkpoint]" >&2; exit 2 ;;
esac
