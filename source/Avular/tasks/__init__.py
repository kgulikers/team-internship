import gymnasium as gym
from .robot_lidar_env import MyRobotLidarEnv, MyRobotLidarEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

# Register the custom environment and its configs so that:
#  • parse_env_cfg(..., "env_cfg_entry_point") finds MyRobotLidarEnvCfg
#  • parse_rsl_rl_cfg(..., "rsl_rl_cfg_entry_point") finds RslRlOnPolicyRunnerCfg
gym.register(
    id="MyRobotLidar-v0",
    entry_point="Avular.tasks.robot_lidar_env:MyRobotLidarEnv",
    max_episode_steps=1000,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MyRobotLidarEnvCfg,
        "rsl_rl_cfg_entry_point": RslRlOnPolicyRunnerCfg,
    },
)
