#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import gym
from gym import spaces
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.scene import InteractiveScene
from scripts.spawn_robot_with_lidar2 import RaycasterSensorSceneCfg, compute_wheel_targets

# Force headless mode and CPU-only PhysX
os.environ["OMNI_PHYSX_USE_GPU_NARROWPHASE"] = "0"
os.environ["OMNI_PHYSX_USE_GPU"] = "0"
os.environ["OMNI_RENDER_HEADLESS"] = "1"
os.environ["HEADLESS"] = "1"
os.environ["LIVESTREAM"] = "0"

class OriginLidarEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'human']}

    def __init__(self, args):
        # 1) Build PhysX & Simulation configs (headless=True disables GUI)
        physx_cfg = PhysxCfg(
            solver_type=0,
            enable_enhanced_determinism=True,
            gpu_collision_stack_size=args.gpu_collision_stack_size,
            gpu_max_rigid_patch_count=args.gpu_max_rigid_patch_count,
            gpu_found_lost_pairs_capacity=args.gpu_found_lost_pairs_capacity,
            min_position_iteration_count=64,
            max_position_iteration_count=64,
            min_velocity_iteration_count=8,
            max_velocity_iteration_count=8,
            enable_ccd=True,
        )
        sim_cfg = SimulationCfg(
            dt=1.0/120.0,
            device=args.device,
            physx=physx_cfg,
            gravity=(0.0, 0.0, -9.81),
            headless=True,    # do not open GUI
            use_gpu=False,    # CPU-only
        )
        # Create simulation context without launching the Kit App
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

        # 2) Build the scene with LiDAR sensor
        scene_cfg = RaycasterSensorSceneCfg(num_envs=args.num_envs, env_spacing=2.0)
        mesh_paths = ["/World/Ground"] + [
            f"/World/envs/env_{eid}/Obstacle{j+1}"
            for eid in range(args.num_envs) for j in range(3)
        ]
        scene_cfg.ray_caster.mesh_prim_paths = mesh_paths
        self.scene = InteractiveScene(scene_cfg)

        # 3) Define action & observation spaces
        self.v_max = args.wheel_speed
        self.w_max = args.yaw_rate
        self.action_space = spaces.Box(
            low=np.array([-self.v_max, -self.w_max], dtype=np.float32),
            high=np.array([ self.v_max,  self.w_max], dtype=np.float32),
            dtype=np.float32
        )
        # LiDAR dimension: channels * horizontal samples
        channels = scene_cfg.ray_caster.pattern_cfg.channels
        samples = int((scene_cfg.ray_caster.pattern_cfg.horizontal_fov_range[1]
                       - scene_cfg.ray_caster.pattern_cfg.horizontal_fov_range[0])
                      / scene_cfg.ray_caster.pattern_cfg.horizontal_res)
        lidar_dim = channels * samples
        self.observation_space = spaces.Box(
            low=0.0, high=100.0, shape=(lidar_dim,), dtype=np.float32
        )

        # 4) Goal & state variables
        self.goal = np.array([5.0, 5.0])
        self.goal_threshold = 0.5
        self.alpha = 0.1
        self.prev_vel = None

    def reset(self):
        # Reset simulation and scene
        self.sim.reset()
        try:
            self.scene.reset()
        except AttributeError:
            pass

        # Warm-up steps
        dt = self.sim.get_physics_dt()
        for _ in range(50):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt)

        # Initialize velocity target
        art = self.scene.articulations['robot']
        self.prev_vel = torch.zeros_like(art.data.joint_vel)
        return self._get_obs()

    def step(self, action):
        v_cmd, w_cmd = action
        art = self.scene.articulations['robot']
        joint_names = art.data.joint_names
        device = art.data.joint_vel.device

        # Compute wheel velocity targets
        raw = compute_wheel_targets(
            joint_names,
            torch.tensor([v_cmd], device=device),
            torch.tensor([w_cmd], device=device),
            device
        )
        vel_target = self.alpha * raw + (1 - self.alpha) * self.prev_vel
        self.prev_vel = vel_target.clone()
        art.set_joint_velocity_target(vel_target)

        # Step simulation
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim.get_physics_dt())

        obs = self._get_obs()
        reward, done = self._compute_reward_done()
        return obs, reward, done, {}

    def _get_obs(self):
        data = self.scene.sensors['ray_caster'].data
        hits = data.ray_hits_w.cpu().numpy()[0]
        finite = hits[np.isfinite(hits).all(axis=1)]
        vec = finite.flatten()
        N = self.observation_space.shape[0]
        if vec.size < N:
            vec = np.pad(vec, (0, N - vec.size), constant_values=100.0)
        else:
            vec = vec[:N]
        return vec.astype(np.float32)

    def _compute_reward_done(self):
        pos = self.scene.articulations['robot'].data.root_pos.cpu().numpy()[0][:2]
        dist = np.linalg.norm(pos - self.goal)
        reward = -dist
        min_dist = np.min(self._get_obs())
        done = bool(dist < self.goal_threshold or min_dist < 0.05)
        if done and min_dist < 0.05:
            reward -= 10.0
        return reward, done

    def render(self, mode='rgb_array'):
        # Not supported in headless mode
        return None

    def close(self):
        # Clean up
        pass

if __name__ == '__main__':
    args = argparse.Namespace(
        num_envs=1,
        gpu_collision_stack_size=130000000,
        gpu_max_rigid_patch_count=2000000,
        gpu_found_lost_pairs_capacity=2000000,
        wheel_speed=2.0,
        yaw_rate=4.0,
        device='cpu',
    )
    env = OriginLidarEnv(args)
    obs = env.reset()
    print('obs shape:', obs.shape)
    action = env.action_space.sample()
    obs2, r, d, _ = env.step(action)
    print(f'reward={r:.3f}, done={d}')
    env.close()
