import os
import argparse
import numpy as np
import gym
from gym import spaces

# ─────────────────────────────────────────────────────────────
# Environment config
# ─────────────────────────────────────────────────────────────
os.environ["OMNI_PHYSX_USE_GPU"] = "0"
os.environ["OMNI_RENDER_HEADLESS"] = "0"

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

# ─────────────────────────────────────────────────────────────
# USD Physics API Patch
# ─────────────────────────────────────────────────────────────
from pxr import Usd, UsdPhysics
from omni.usd import get_context

def patch_physics_apis(stage, prim_paths, verbose=False):
    def apply_apis(prim):
        if prim.IsInstance():
            prim.SetInstanceable(False)
            if verbose: print(f"Disabled instancing: {prim.GetPath()}")

        if not UsdPhysics.RigidBodyAPI.HasAPI(prim):
            UsdPhysics.RigidBodyAPI.Apply(prim)
            if verbose: print(f"Applied RigidBodyAPI to {prim.GetPath()}")

        if not UsdPhysics.CollisionAPI.HasAPI(prim):
            UsdPhysics.CollisionAPI.Apply(prim)
            if verbose: print(f"Applied CollisionAPI to {prim.GetPath()}")

        if not UsdPhysics.MassAPI.HasAPI(prim):
            UsdPhysics.MassAPI.Apply(prim)
            if verbose: print(f"Applied MassAPI to {prim.GetPath()}")

        for child in prim.GetChildren():
            apply_apis(child)

    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            apply_apis(prim)

    try:
        root = stage.GetRootLayer()
        if root.IsAnonymous():
            root.Export("/tmp/patched_stage.usd")
        else:
            root.Save()
    except Exception as e:
        print(f"Warning: Could not save stage: {e}")

# ─────────────────────────────────────────────────────────────
# Minimal AvularEnv
# ─────────────────────────────────────────────────────────────
class AvularEnv(gym.Env):
    def __init__(self, start_pose=None, goal_pose=None, max_steps=500):
        super().__init__()
        self.start_pose = start_pose
        self.goal_pose = goal_pose
        self.max_steps = max_steps
        self.sim, _ = app_launcher.launch()
        self.sim.reset()
        self.sim.play()

        stage = get_context().get_stage()
        robot_path = "/World/Robot"
        patch_physics_apis(stage, [robot_path], verbose=True)

        # Define dummy action/obs space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self):
        sim_app.update()
        return np.zeros(5, dtype=np.float32)

    def step(self, action):
        sim_app.update()
        obs = np.random.randn(5).astype(np.float32)
        reward = 0.0
        done = False
        return obs, reward, done, {}

    def render(self, mode="human"):
        self.sim.render()
