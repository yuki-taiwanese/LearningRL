# Alternating gait revision (not trained)

Task: `LearningRL-RPO-AlternatingWalk-v0` (and `-PLAY`). The evaluated
`LearningRL-RPO-ForwardWalk-v0` baseline remains available for comparison.
This revision changes training rewards, not an existing trained policy.

## Initial settings

The provisional target is a **3 cm peak sole-centre clearance**, with a smooth
rise and fall during swing. It is not a 3 cm ankle-link height. On flat ground,
the sole-centre proxy subtracts the rotated 4.5 cm foot offset from the ankle
body position. It does not measure the lowest point of the collision mesh.

The cycle is 0.8 s, with opposite left/right phases and 60% stance per foot.
This permits brief double support (20% of the cycle); it never requests flight.

| Term | Weight | Condition |
| --- | ---: | --- |
| Contact pattern | +1.5 | Both feet match the expected stance/swing pattern |
| Swing clearance | +2.0 | Smooth height target; requires the complete contact pattern to match |
| Wrong support | -1.0 | Mean contact mismatch across feet |
| Both feet airborne | -2.0 | Neither foot contacts the ground |
| Excess foot height | -0.5 | Squared excess above 7 cm, bounded |
| Forward progress | +1.0 | Bounded; full positive credit in single support, half in double support, zero in flight |
| Foot sliding | -0.5 | Existing sliding term, increased from -0.2 |
| Vertical body velocity | -4.0 | Existing term, increased from -2.0 |

Contact-pattern, swing-clearance and wrong-support terms are disabled for
stationary XY commands. Flight and excess-height costs remain active.
Existing speed tracking (+3.0), yaw tracking (+1.5), balance, body-height and
joint costs are retained. These are reward weights, not probabilities or
success guarantees. The new forward-progress term also rejects zero-command
reward farming. Higher feet do not receive ever-increasing rewards.

## Checkpoint and training compatibility

The actor and critic now receive an internal sine/cosine gait clock. This adds
two inputs: actor 78 -> 80, critic 81 -> 83. No physical sensor is added.
Do not load an old checkpoint directly into this task.

For a trusted forward checkpoint, prepare a NEW file:

```bash
python tools/migrate_gait_checkpoint.py /path/to/model_1746.pt /path/to/alternating_seed.pt
```

The converter appends zero-weight clock inputs, preserves the initial network
functions, resets Adam moments and iteration count, and leaves the source file
untouched. The converted seed is **not a trained alternating policy**. It must
be placed in the new experiment's checkpoint directory and selected using the
training runner's resume options. Use the normal README training procedure with
task `LearningRL-RPO-AlternatingWalk-v0`; the experiment family is
`learningrl_rpo_alternating_walk`. Fresh training is also possible.

The converter and reward mathematics pass CPU tests. Isaac environment startup,
real checkpoint loading and GPU training remain unverified for this revision.
The standard LearningRLsim export/replay path also needs the added clock inputs
implemented before it can replay this policy. Do not deploy it to hardware.

## Evaluation before claiming walking success

Record both foot contacts and sole-centre heights alongside root displacement.
Check alternating landings, simultaneous takeoffs, sliding and falls across
multiple seeds, including standing then walking. Distance alone does not prove
alternating steps. Compare against the unchanged forward baseline. Tune the
3 cm target only after inspecting these traces and correct-orientation video.

Earlier local replay images/videos used an incorrect xyzw-to-wxyz root
quaternion mapping. The rendering scripts have been corrected, but those old
media have not been regenerated; they are not reliable gait evidence. This
does not alter the original checkpoints or displacement measurements.

No new training was launched for this reward revision because the current
Colab account has no available compute units and GPU access is unresolved.
