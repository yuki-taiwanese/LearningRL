"""Two-table box transfer with the stock Roboto Origin."""

import gymnasium as gym

from . import agents

gym.register(
    id="LearningRL-RPO-Approach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.approach_env_cfg:RPOApproachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.approach_ppo_cfg:RPOApproachPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="LearningRL-RPO-Approach-v0-PLAY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.approach_env_cfg:RPOApproachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.approach_ppo_cfg:RPOApproachPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="LearningRL-RPO-BoxTransfer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.box_env_cfg:RPOBoxTransferEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RPOBoxTransferPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
gym.register(
    id="LearningRL-RPO-BoxTransfer-v0-PLAY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.box_env_cfg:RPOBoxTransferEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RPOBoxTransferPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
