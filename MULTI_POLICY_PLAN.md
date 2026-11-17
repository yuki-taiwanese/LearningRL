# Skill separation for the two-box project

## Implemented now

The two-box task and its original single whole-body policy remain available as a comparison. An independent unloaded walking task is registered as `LearningRL-RPO-FlatWalk-v0` with a `-PLAY` twin. Its checkpoint family is `learningrl_rpo_flat_walk`.

This walking policy takes the stock walking observations: angular velocity, projected gravity, commanded base velocity, joint positions/velocities and the previous action. Its actor does not use box positions or simulated object contact forces. It outputs all 23 joint targets, including the arms, which are encouraged to stay near the robot's nominal pose. Actual hardware state interfaces still need verification.

The walk skill trains on flat ground with speeds limited to +/-0.35 m/s and yaw rate +/-0.5 rad/s; 25% of command samples request standing. Startup mass/friction randomization and pushes are disabled for this initial skill. The initial policy noise is 0.5 instead of the course default 1.0. Other walking reward terms and actuator settings are inherited from the course RPO task. Mirror augmentation is disabled because its current source-URDF lookup is incompatible with the preconverted USD spawn.

The walking actor/critic inputs are expected to be 78/81, compared with 145/148 for the two-box task. They are different checkpoint formats. Do not resume one task directly from the other's checkpoint.

The 20-second episode horizon is only a locomotion training horizon. The two-box task still has no time-out termination and no per-second hurry penalty.

## Intended eventual skills (not integrated yet)

1. **Unloaded locomotion:** approach the left table and return empty handed after each delivery.
2. **Stationary manipulation:** establish a forearm grasp, lift, lower and release either size while maintaining balance.
3. **Loaded locomotion:** walk while preserving the held-arm configuration and accommodating the box load.

An unloaded walk policy is not evidence that the robot can walk while carrying a box. Loaded locomotion needs its own training or a walking policy explicitly conditioned and trained on arm poses and loads.

The eventual supervisor must choose one whole-body joint controller at a time, check a low-speed, stable transition posture, and blend bounded joint targets on handoff. Two policies must never send conflicting targets to the same joints. The current code does not switch policies during the box task: only the unloaded walking skill has been added separately.

Box grasp/place verification and the original actor's box measurements remain simulator-only. Sensor-limited hardware execution still needs an explicit estimation or constrained-task design. No extra physical sensors, grippers or robot modifications are assumed.

## Current experiment procedure

- Continue the existing two-box checkpoint as a control, rather than restarting it.
- Train the flat-walk skill in a separate run and keep its checkpoints separate.
- Copy newly completed checkpoints to the authorized group Drive while training, verify hashes, and preserve logs and configuration.
- Judge walking using falls, sustained episode duration, command tracking and independent rollouts; a higher training reward alone does not establish success.
- Do not claim a complete two-box behavior until all deliveries and both empty-handed returns have been evaluated together.
