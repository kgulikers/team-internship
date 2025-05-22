#!/usr/bin/env python3
import argparse

from scripts.env_origin_lidar import OriginLidarEnv

def make_args():
    return argparse.Namespace(
        num_envs=1,
        gpu_collision_stack_size=130000000,
        gpu_max_rigid_patch_count=2000000,
        gpu_found_lost_pairs_capacity=2000000,
        project="test-project",
        entity=None,
        run_name="env-test",
        wheel_speed=2.0,
        yaw_rate=4.0,
        device="cpu",               # or "cuda:0" if you have a GPU
    )

def main():
    args = make_args()
    env = OriginLidarEnv(args)

    # 1) Reset
    obs = env.reset()
    print(f"[sanity] initial obs shape: {obs.shape}")

    # 2) Sample random action & step
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    print(f"[sanity] action: {action}")
    print(f"[sanity] reward: {reward:.3f}, done: {done}")
    print(f"[sanity] next obs shape: {next_obs.shape}")

    # 3) Clean up
    env.close()

if __name__ == "__main__":
    main()
