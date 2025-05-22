#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import os

from scripts.env_origin_lidar import OriginLidarEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env

# ─── UPDATED IMPORT ───────────────────────────────────────────────────
from gym.wrappers import TimeLimit


def make_args():
    return argparse.Namespace(
        num_envs=1,
        gpu_collision_stack_size=130000000,
        gpu_max_rigid_patch_count=2000000,
        gpu_found_lost_pairs_capacity=2000000,
        project="test-project",
        entity=None,
        run_name="env-test",
        wheel_speed=8.0,
        yaw_rate=16.0,
        device="cpu",
        total_timesteps=5_000_000,
        save_freq=100_000,
        eval_freq=50_000,
        max_episode_steps=20000,  # how long before we force-reset
    )


def main():
    args = make_args()
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./tensorboard", exist_ok=True)

    # 1) raw Isaac Sim env
    # Use a lambda to pass args to OriginLidarEnv
    env = make_vec_env(
        lambda: TimeLimit(OriginLidarEnv(args), max_episode_steps=args.max_episode_steps),
        n_envs=args.num_envs
    )


    # 3) callbacks for checkpointing & evaluation
    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path="./models/",
        name_prefix="ppo_origin"
    )
    eval_cb = EvalCallback(
        env,
        best_model_save_path="./models/best/",
        log_path="./models/eval_logs/",
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )
    callbacks = CallbackList([checkpoint_cb, eval_cb])

    # 4) create & train PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard/",
        device=args.device,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
    )

    # 5) save final policy
    model.save("./models/final_ppo_origin")


if __name__ == "__main__":
    main()
