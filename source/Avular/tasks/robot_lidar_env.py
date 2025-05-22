#!/usr/bin/env python3
import os
import sys
import time
import argparse
import torch
import numpy as np
import gymnasium as gym
from dataclasses import dataclass

from isaaclab.app import AppLauncher
from isaaclab.sim import SimulationContext
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg

# ── Robot / world constants ─────────────────────────────────────────────────────
TRACK_WIDTH      = 0.6
WHEEL_RADIUS     = 0.1175
GROUND_THICKNESS = 0.10
MARGIN           = 0.005
_init_z = WHEEL_RADIUS + GROUND_THICKNESS/2 + MARGIN

# ── Configuration dataclass ────────────────────────────────────────────────────
@dataclass
class MyRobotLidarEnvCfg(DirectMARLEnvCfg):
    """
    Config for the robot+LiDAR environment.
    """
    device: str = "cpu"
    num_envs: int = 1
    env_spacing: float = 2.0
    wheel_speed: float = 2.0      # m/s
    yaw_rate: float = 4.0         # rad/s
    gpu_collision_stack_size: int = 130_000_000
    gpu_max_rigid_patch_count: int = 2_000_000
    gpu_found_lost_pairs_capacity: int = 2_000_000

    def __post_init__(self):
        # Physics settings
        self.physx = PhysxCfg(
            solver_type=0,
            enable_enhanced_determinism=True,
            gpu_collision_stack_size=self.gpu_collision_stack_size,
            gpu_max_rigid_patch_count=self.gpu_max_rigid_patch_count,
            gpu_found_lost_pairs_capacity=self.gpu_found_lost_pairs_capacity,
            min_position_iteration_count=64,
            max_position_iteration_count=64,
            min_velocity_iteration_count=8,
            max_velocity_iteration_count=8,
            enable_ccd=True,
        )
        # Simulation settings
        self.sim = SimulationCfg(
            dt=1.0/120.0,
            device=self.device,
            physx=self.physx,
            gravity=(0.0, 0.0, -9.81),
        )
        super().__post_init__()


# ── Environment class ──────────────────────────────────────────────────────────
class MyRobotLidarEnv(DirectMARLEnv):
    """
    Gym environment that spins up Isaac Lab with a robot + LIDAR and obstacles.
    """
    def __init__(self, cfg: MyRobotLidarEnvCfg):
        # Force CPU‐only PhysX & enable GUI
        os.environ["OMNI_PHYSX_USE_GPU_NARROWPHASE"] = "0"
        os.environ["OMNI_PHYSX_USE_GPU"]           = "0"
        os.environ["OMNI_RENDER_HEADLESS"]        = "0"
        os.environ["HEADLESS"]                    = "0"
        os.environ["LIVESTREAM"]                  = "0"

        # Launch the Isaac Lab app early so that `isaacsim` and `omni.isaac` load
        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        app_args, _ = parser.parse_known_args([])
        app_launcher = AppLauncher(app_args)
        self.sim_app = app_launcher.app

        # Initialize DirectMARLEnv (sets up gym.Env basics)
        super().__init__(cfg)

        # Customize LIDAR sensor range if supported
        env = self.unwrapped
        if hasattr(env, 'gym_env') and hasattr(env.gym_env, 'configure_sensor'):
            env.gym_env.configure_sensor('lidar', range=getattr(cfg, 'lidar_range', None))

        # (Optional) programmatic obstacle addition
        # for i in range(cfg.obstacle_count):
        #     self.add_obstacle(name=f"obstacle_{i}")
