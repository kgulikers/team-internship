import numpy as np
from avular.test import AvularEnv

def main():
    start_pose = (0.0, 0.0, 0.0)
    goal_pose = (5.0, 5.0)

    env = AvularEnv(start_pose=start_pose, goal_pose=goal_pose, max_steps=100)

    obs = env.reset()
    print(f"Initial obs shape: {obs.shape}")

    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {step:2d} → action={action}, reward={reward:.2f}, done={done}")
        if done:
            print("Episode terminated early.")
            break

    env.render()

if __name__ == "__main__":
    main()
