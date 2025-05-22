#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
import torch
import gym
from gym import spaces

# ─── FORCE CPU-ONLY PhysX & HEADLESS/GUI ───────────────────────────────────────────
os.environ["OMNI_PHYSX_USE_GPU_NARROWPHASE"] = "0"
os.environ["OMNI_PHYSX_USE_GPU"] = "0"
os.environ["OMNI_RENDER_HEADLESS"] = "1"
os.environ["HEADLESS"] = "1"
os.environ["LIVESTREAM"] = "0"

from isaaclab.app import AppLauncher

# ─── PARSE & LAUNCH KIT ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Gym RL env: go A→B, avoid obstacles.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--gpu_collision_stack_size", type=int, default=130000000)
parser.add_argument("--gpu_max_rigid_patch_count", type=int, default=2000000)
parser.add_argument("--gpu_found_lost_pairs_capacity", type=int, default=2000000)
parser.add_argument("--wheel_speed", type=float, default=8.0)
parser.add_argument("--yaw_rate", type=float, default=16.0)
parser.add_argument("--project", type=str, default="isaac-sim-demo")
parser.add_argument("--entity", type=str, default=None)
parser.add_argument("--run_name", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
# DO NOT CALL simulation_app.close() globally — it gets killed prematurely
GLOBAL_SIM_APP = app_launcher.app  # Keep reference alive


# ─── HEAVY IMPORTS AFTER LAUNCH ──────────────────────────────────────────────────
import wandb
import isaaclab.sim as sim_utils
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils import configclass

wandb_key = os.getenv("4593d25c7165eb7adb5091abca9228fe0bd2182d")
if wandb_key:
    wandb.login(key=wandb_key)
else:
    wandb.login()
wandb.init(project=args.project, entity=args.entity, name=args.run_name, config=vars(args))

TRACK_WIDTH = 0.6
WHEEL_RADIUS = 0.1175
GROUND_THICKNESS = 0.10
MARGIN = 0.005
init_z = WHEEL_RADIUS + GROUND_THICKNESS / 2 + MARGIN

_ZERO_INIT_STATES = ArticulationCfg.InitialStateCfg(
    pos=[0.0, 0.0, init_z],
    rot=[2, 0, 0, 0],
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)

alpha = 0.1

def compute_wheel_targets(joint_names, v: torch.Tensor, w: torch.Tensor, device):
    right_idxs = [i for i, n in enumerate(joint_names) if "wheel" in n.lower() and "right" in n.lower()]
    left_idxs = [i for i, n in enumerate(joint_names) if "wheel" in n.lower() and i not in right_idxs]
    base_omega = v / WHEEL_RADIUS
    delta_omega = (w * TRACK_WIDTH * 0.5) / WHEEL_RADIUS
    omega_left = base_omega - delta_omega
    omega_right = base_omega + delta_omega
    num_envs = v.shape[0]
    num_joints = len(joint_names)
    vel_target = torch.zeros((num_envs, num_joints), device=device)
    vel_target[:, left_idxs] = omega_left.unsqueeze(-1)
    vel_target[:, right_idxs] = omega_right.unsqueeze(-1)
    return vel_target

@configclass
class RaycasterSensorSceneCfg(InteractiveSceneCfg):
    env_spacing: float = 2.0

    ground = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Ground",
        spawn=sim_utils.MeshCuboidCfg(
            size=(50.0, 50.0, GROUND_THICKNESS),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.01),
            physics_material=RigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=2.0, restitution=0.0),
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/avular_origin_v18",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "source", "Avular_assets", "origin_v18", "origin_v18.urdf")
            ),
            joint_drive=None,
            fix_base=False,
            merge_fixed_joints=True,
            convert_mimic_joints_to_normal_joints=True,
            root_link_name="main_body",
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
        ),
        init_state=_ZERO_INIT_STATES,
        actuators={
            "wheel_act": IdealPDActuatorCfg(
                joint_names_expr=[".*wheel.*"],
                stiffness=3.0,
                damping=60.0,
                effort_limit=20.0,
                velocity_limit=args.wheel_speed / WHEEL_RADIUS,
            )
        },
    )
    obstacle1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[5.0, 0.0, 1.5], rot=[1, 0, 0, 0]),
        spawn=sim_utils.MeshCuboidCfg(size=(2.5, 2.5, 3.0)),
    )
    obstacle2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-5.0, -3.0, 1.0], rot=[1, 0, 0, 0]),
        spawn=sim_utils.MeshCuboidCfg(size=(1.5, 4.0, 2.0)),
    )
    obstacle3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 5.0, 2.0], rot=[1, 0, 0, 0]),
        spawn=sim_utils.MeshCuboidCfg(size=(3.0, 1.0, 4.0)),
    )
    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/avular_origin_v18/main_body",
        update_period=1,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        attach_yaw_only=False,
        mesh_prim_paths=[
            "/World/envs/env_0/Ground",
            "/World/envs/env_0/Obstacle1",
            "/World/envs/env_0/Obstacle2",
            "/World/envs/env_0/Obstacle3"
        ],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0, 0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0
        ),
        debug_vis=False,
    )


class OriginLidarEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, args):
        self.simulation_app = GLOBAL_SIM_APP

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
            dt=1.0 / 120.0,
            device=args.device,
            physx=physx_cfg,
            gravity=(0.0, 0.0, -9.81),
        )

        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
        self.scene = InteractiveScene(RaycasterSensorSceneCfg(num_envs=args.num_envs))
        self.action_space = spaces.Box(
            low=np.array([-args.wheel_speed, -2 * args.yaw_rate], dtype=np.float32),
            high=np.array([args.wheel_speed, 2 * args.yaw_rate], dtype=np.float32),
            dtype=np.float32
        )


        lidar_dim = int((180 - (-180)) / 1.0) * 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=100.0,
            shape=(lidar_dim,),
            dtype=np.float32
        )
        self.goal = np.array([5.0, 0.0])
        self.goal_threshold = 0.5
        self.prev_vel = None
        self.step_count = 0
        

    def reset(self):
        self.sim.reset()
        try: self.scene.reset()
        except: pass
        for _ in range(50):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim.get_physics_dt())
        art = self.scene.articulations["robot"]
        self.prev_vel = torch.zeros_like(art.data.joint_vel)

        return self._get_obs()

    def step(self, action):
        v_cmd, w_cmd = action
        art = self.scene.articulations["robot"]
        raw = compute_wheel_targets(
            art.data.joint_names,
            torch.tensor([v_cmd], device=art.data.joint_vel.device),
            torch.tensor([w_cmd], device=art.data.joint_vel.device),
            art.data.joint_vel.device
        )
        vel_target = alpha * raw + (1 - alpha) * self.prev_vel
        self.prev_vel = vel_target.clone()
        art.set_joint_velocity_target(vel_target)

        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim.get_physics_dt())

        obs = self._get_obs()
        # compute extras
        reward, done, dist, min_dist = self._compute_reward_done()
        self.step_count += 1
        wandb.log({
            "reward": reward,
            "distance_to_goal": dist,
            "min_lidar": min_dist,
            "v_cmd": float(v_cmd),
            "w_cmd": float(w_cmd)
        }, step=self.step_count)
        print(f"[DEBUG] step reward: {reward:.3f}, done: {done}, min_lidar: {np.min(obs):.2f}")

        return obs, reward, done, {}
    
    def _get_obs(self):
        # Get valid ray hit points: shape (num_rays, 3)
        hits = self.scene.sensors["ray_caster"].data.ray_hits_w.cpu().numpy()[0]
        fin = hits[np.isfinite(hits).all(axis=1)]

        if fin.shape[0] == 0:
            # No valid hits, return padded observation
            return np.full(self.observation_space.shape[0], 100.0, dtype=np.float32)

        # Extract x, y, z
        x, y, z = fin[:, 0], fin[:, 1], fin[:, 2]

        # Compute distance and azimuth (theta)
        distances = np.sqrt(x**2 + y**2 + z**2)
        azimuths = np.arctan2(y, x)  # θ in radians

        # Stack into shape (num_hits, 2) and flatten
        vec = np.stack([distances, azimuths], axis=1).flatten()

        # Pad to match expected observation vector size
        N = self.observation_space.shape[0]
        if vec.size < N:
            vec = np.pad(vec, (0, N - vec.size), constant_values=100.0)

        # Optional debug print
        #print(f"[DEBUG] ray hits (distance, theta): {vec[:N]}")

        return vec[:N].astype(np.float32)

    def _compute_reward_done(self):
        sensor_pos = self.scene.sensors["ray_caster"].data.pos_w.cpu().numpy()[0][:2]
        dist = np.linalg.norm(sensor_pos - self.goal)
        delta_dist = getattr(self, "last_dist", dist) - dist
        self.last_dist = dist

        reward = delta_dist * 10.0  # reward for moving toward goal
        min_dist = -np.min(self._get_obs())

        done = False
        if min_dist < 0.05:
            reward -= 10.0
            done = True
            print(f"[DEBUG] Collision detected: min_lidar_dist={-min_dist:.3f} < 0.05")

        if dist < self.goal_threshold:
            reward += 20.0
            done = True
            print(f"[DEBUG] Goal reached: distance={dist:.3f} < {self.goal_threshold}")

        print(f"[DEBUG] Position: {sensor_pos}, Goal: {self.goal}, Distance: {dist:.3f}, Reward: {reward:.3f}, Done: {done}")

        return reward, done, dist, min_dist


    def render(self, mode="human"):
        pass

    def close(self):
        # Properly shut down the app if you're done
        if hasattr(self, "simulation_app") and self.simulation_app is not None:
            try:
                self.simulation_app.close()
            except Exception as e:
                print(f"[WARN] Error closing simulation_app: {e}")


if __name__ == "__main__":
    import sys
    if "--run-standalone" in sys.argv:
        env = OriginLidarEnv(args)
        for episode in range(1000):
            done = False
            obs = env.reset()
            while not done:
                a = env.action_space.sample()
                obs, reward, done, info = env.step(a)
            print(f"Episode {episode} ended. Restarting…")
        env.close()
        print("[DEBUG] simulation_app.close() was called!")
        GLOBAL_SIM_APP.close()


