#!/usr/bin/env python3
import os, sys
from isaaclab.app import AppLauncher
import cli_args
import argparse

parser = argparse.ArgumentParser("Train an RL agent with RSL-RL.")

# 1) Your custom flags must be right here, before any parsing:
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=1000,
                    help="Number of RL training iterations")

# 2) RSL‑RL flags
cli_args.add_rsl_rl_args(parser)

# 3) Isaac Sim flags
AppLauncher.add_app_launcher_args(parser)

# DEBUG: show what flags the parser knows
parser.print_help()   # ← insert this

# 4) Now parse – this must pick up your two flags!
args_cli, extra = parser.parse_known_args()

# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 3) Launch the OmniverseKit app **before** importing any omni.* or carb.*
app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 4) Now it’s safe to pull in IsaacLab/omni modules and our Gym wrapper
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# our environment lazily imports isaaclab.sim only after Kit is up
from avular.envs.lidar_env import LidarDriveEnv
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 5) Instantiate & wrap the env
    env = LidarDriveEnv(num_envs=args_cli.num_envs)
    env = RslRlVecEnvWrapper(env, clip_actions=True)

    # 6) Runner configuration
    config = {
        "max_iterations": args_cli.max_iterations,
        "device": args_cli.device,
    }

    # 7) Create & run the OnPolicyRunner
    runner = OnPolicyRunner(
        env,
        config,
        log_dir="logs/rslrl_plain",
        device=args_cli.device
    )
    runner.learn(num_learning_iterations=config["max_iterations"])

    # 8) Clean up the env
    env.close()

if __name__ == "__main__":
    try:
        main()
    finally:
        # 9) Tear down Kit so you can `./train_rslrl.py` again cleanly
        simulation_app.close()
