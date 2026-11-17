# Forward curriculum result — 2026-11-16

The forward-only policy reached the internal one-meter diagnostic gate after **1,000 additional PPO updates**, in two 500-update chunks with 512 environments (12,288,000 transitions). Training stopped at that gate rather than consuming the full optional 2,000-update allowance. The warm start was the preserved balance checkpoint `model_748.pt`; these results are not from a fresh initialization.

Task source: `36521e4a473595ccc08ad73fb99113f408ef4960`. Task: `LearningRL-RPO-ForwardWalk-v0`. Newton / MuJoCo-Warp physics on a Colab T4; task code and the physical robot were unchanged between the two chunks. All 22 new checkpoint copies matched their runtime originals by SHA-256; actor tensors and logged losses were finite.

| Frozen-policy test | Survived 10 seconds | Horizontal root displacement | Mean body-forward speed during forward phase |
|---|---:|---:|---:|
| Nominal, +0.20 m/s command, seed 20261116 | 5/5 | 1.165–1.556 m | 0.126–0.167 m/s |
| Variation, +0.18 m/s, initial yaw ±0.15 rad, seed 20261117 | 5/5 | 1.234–1.464 m | See saved evaluation JSON |

Both protocols start with a one-second zero command, then nine seconds of forward command. Scoring ends at each environment's first termination. The internal gate also checks lateral/yaw drift and velocity tracking; it is not a course grading rule. Earlier +500-update trials survived 5/5 but displaced only 0.218–1.131 m, so the additional training improved consistency.

Best checkpoint in authorized group Drive:

```text
My Drive/LearningRL/zhaoyu-lu-HMD/rounds/
  forward_chunk2_2026-11-16_231710/model_1746.pt
```

That folder contains `evaluation.json`, `variation.json`, recorded state arrays, CPU kinematic replay videos/contact sheets, configuration, training logs and checkpoint verification. The project `forward_curriculum` folder contains `Forward_Curriculum_Final_Report.json` with all intermediate outcomes and all checkpoint hashes. Kinematic replay renders the recorded Newton states; it is not an independent MuJoCo physics validation.

This is **simulated unloaded forward motion only**. Stopping at the table, turning, object pickup/place, loaded walking, the skill-switching controller and hardware deployment have not been established. The original two-box task remains intact. The next curriculum stage should teach controlled stopping before object manipulation is integrated.
