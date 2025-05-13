#!/usr/bin/env python3
import os
import sys
# ─────────────────────────────────────────────────────────────────────────────
# FORCE CPU-ONLY PHYSX & ENABLE RENDERING
os.environ["OMNI_PHYSX_USE_GPU_NARROWPHASE"] = "0"
os.environ["OMNI_PHYSX_USE_GPU"]           = "0"
os.environ["OMNI_RENDER_HEADLESS"]        = "0"
os.environ["HEADLESS"]                     = "0"
os.environ["LIVESTREAM"]                   = "0"
# ─────────────────────────────────────────────────────────────────────────────

import argparse
from isaaclab.app import AppLauncher

descr = (
    "Spawn robot + LiDAR, drive wheels with commanded velocities, accumulate LiDAR scans, log to W&B."
)
parser = argparse.ArgumentParser(description=descr)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--gpu_collision_stack_size", type=int, default=130000000)
parser.add_argument("--gpu_max_rigid_patch_count", type=int, default=2000000)
parser.add_argument("--gpu_found_lost_pairs_capacity", type=int, default=2000000)
parser.add_argument("--project", type=str, default="isaac-sim-demo")
parser.add_argument("--entity", type=str, default=None)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument(
    "--wheel_speed", type=float, default=2.0,
    help="Nominal wheel linear velocity target (m/s)"
)
parser.add_argument(
    "--yaw_rate", type=float, default=4.0,
    help="Nominal yaw rate target (rad/s) when pressing A/D"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Simulation App immediately
app_launcher   = AppLauncher(args)
simulation_app = app_launcher.app

# Ensure IsaacSim / Omniverse Python paths are available
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    import carb
except ModuleNotFoundError:
    print("[ERROR] Missing 'carb' module: ensure you have sourced the IsaacSim/Omniverse Python environment.")

# Heavy imports after app startup
import time
import torch
import numpy as np
import wandb
import isaaclab.sim as sim_utils
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg

import sys, select, termios, tty

# WANDB LOGIN & INIT
wandb_key = os.getenv("WANDB_API_KEY")
if wandb_key:
    wandb.login(key=wandb_key)
else:
    wandb.login()
wandb.init(
    project=args.project,
    entity=args.entity,
    name=args.run_name,
    config={
        "num_envs": args.num_envs,
        "gpu_collision_stack_size": args.gpu_collision_stack_size,
        "gpu_max_rigid_patch_count": args.gpu_max_rigid_patch_count,
        "gpu_found_lost_pairs_capacity": args.gpu_found_lost_pairs_capacity,
        "wheel_speed": args.wheel_speed,
        "yaw_rate": args.yaw_rate,
    }
)

# Constants for initial placement
TRACK_WIDTH      = 0.6
WHEEL_RADIUS     = 0.1175
GROUND_THICKNESS = 0.10
MARGIN           = 0.005

# Compute Z so wheels just touch ground
init_z = WHEEL_RADIUS + GROUND_THICKNESS/2 + MARGIN

# Zero initial states for robot
_ZERO_INIT_STATES = ArticulationCfg.InitialStateCfg(
    pos=[0.0, 0.0, init_z],
    rot=[1, 0, 0, 0],
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)

# Filter parameter
alpha = 0.1

def compute_wheel_targets(joint_names, v: torch.Tensor, w: torch.Tensor, device):
    right_idxs = [i for i,n in enumerate(joint_names)
                  if "wheel" in n.lower() and "right" in n.lower()]
    left_idxs  = [i for i,n in enumerate(joint_names)
                  if "wheel" in n.lower() and i not in right_idxs]

    num_envs    = v.shape[0]
    num_joints  = len(joint_names)
    vel_target  = torch.zeros((num_envs, num_joints), device=device)

    base_omega    = v / WHEEL_RADIUS
    delta_omega   = (w * TRACK_WIDTH * 0.5) / WHEEL_RADIUS
    omega_left    = base_omega - delta_omega
    omega_right   = base_omega + delta_omega

    vel_target[:, left_idxs]  = omega_left.unsqueeze(-1)
    vel_target[:, right_idxs] = omega_right.unsqueeze(-1)
    return vel_target

@configclass
class RaycasterSensorSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.MeshCuboidCfg(
            size=(50.0, 50.0, GROUND_THICKNESS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02, rest_offset=0.01
            ),
            physics_material=RigidBodyMaterialCfg(
                static_friction=2.0, dynamic_friction=2.0, restitution=0.0
            ),
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75,0.75,0.75)),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/avular_origin_v18",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "source",
                    "Avular_assets", "origin_v18", "origin_v18.urdf"
                )
            ),
            joint_drive=None,
            fix_base=False,
            merge_fixed_joints=True,
            convert_mimic_joints_to_normal_joints=True,
            root_link_name="main_body",
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.01, rest_offset=0.0
            ),
        ),
        init_state=_ZERO_INIT_STATES,
        actuators={"wheel_act": IdealPDActuatorCfg(
            joint_names_expr=[".*wheel.*"],
            stiffness=3.0,    # softer spring
            damping=60.0,     # stronger brake
            effort_limit=20.0,
            velocity_limit=args.wheel_speed / WHEEL_RADIUS,
        )},
    )
    obstacle1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[5.0,0.0,1.5], rot=[1,0,0,0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(2.5,2.5,3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            physics_material=RigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=2.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8,0.2,0.2)),
        ),
    )
    obstacle2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-5.0,-3.0,1.0], rot=[1,0,0,0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.5,4.0,2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            physics_material=RigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=2.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2,0.8,0.2)),
        ),
    )
    obstacle3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0,5.0,2.0], rot=[1,0,0,0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(3.0,1.0,4.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            physics_material=RigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=2.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2,0.2,0.8)),
        ),
    )
    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/avular_origin_v18/main_body",
        update_period=1,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0,0.0,0.5)),
        attach_yaw_only=False,
        mesh_prim_paths=[],  # set below
        pattern_cfg=patterns.LidarPatternCfg(
            channels=32, vertical_fov_range=(-22.5,22.5), horizontal_fov_range=(-180.0,180.0), horizontal_res=1.0
        ),
        debug_vis=False,
    )


def run_simulator(sim: sim_utils.SimulationContext,
                  scene: InteractiveScene,
                  sim_app):
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    try:
        sim_dt = sim.get_physics_dt()
        sim.reset()
        for _ in range(50):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

        step_count = 0
        print("[INFO]: Running simulation (W/S/A/D to steer, Q to quit)…")

        robot_art   = scene.articulations["robot"]
        joint_names = robot_art.data.joint_names
        device      = robot_art.data.joint_vel.device

        prev_vel_target = torch.zeros_like(robot_art.data.joint_vel)

        while sim_app.is_running():
            sim_app.update()

            # read keyboard
            v_cmd, w_cmd = 0.0, 0.0
            if select.select([sys.stdin], [], [], 0)[0]:
                c = sys.stdin.read(1).lower()
                if   c == 'w': v_cmd =  args.wheel_speed
                elif c == 's': v_cmd = -args.wheel_speed
                elif c == 'a': w_cmd =  args.yaw_rate
                elif c == 'd': w_cmd = -args.yaw_rate
                elif c == 'q':
                    print("[INFO]: Quit command received.")
                    break

            # dead-zone
            if abs(v_cmd) < 1e-3 and abs(w_cmd) < 1e-3:
                v_cmd, w_cmd = 0.0, 0.0

            print(f"[CMD] v = {v_cmd:.2f} m/s, w = {w_cmd:.2f} rad/s")

            # compute & filter
            v_tensor = torch.full((args.num_envs,), v_cmd, device=device)
            w_tensor = torch.full((args.num_envs,), w_cmd, device=device)
            raw_vel  = compute_wheel_targets(joint_names, v_tensor, w_tensor, device)
            vel_target = alpha * raw_vel + (1 - alpha) * prev_vel_target
            prev_vel_target = vel_target.clone()

            # always velocity control
            robot_art.set_joint_velocity_target(vel_target)

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            actual = robot_art.data.joint_vel.cpu().numpy()[0]
            print(f"[DEBUG] actual joint_vel (rad/s) at wheels =", actual)

            pos_w = scene.sensors["ray_caster"].data.pos_w.cpu().numpy()[0]
            wandb.log({"base_height": pos_w[2]}, step=step_count)
            step_count += 1

        print("[INFO]: Simulation stopped.")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    physx_cfg = PhysxCfg(
        solver_type=0,  # use PGS solver
        enable_enhanced_determinism=True,
        gpu_collision_stack_size=args.gpu_collision_stack_size,
        gpu_max_rigid_patch_count=args.gpu_max_rigid_patch_count,
        gpu_found_lost_pairs_capacity=args.gpu_found_lost_pairs_capacity,
        min_position_iteration_count=64,
        max_position_iteration_count=64,
        min_velocity_iteration_count=8,
        max_velocity_iteration_count=8,
        enable_ccd=True
    )
    sim_cfg = SimulationCfg(
        dt=1.0/120.0,
        device=args.device,
        physx=physx_cfg,
        gravity=(0.0,0.0,-9.81),
        physics_material=RigidBodyMaterialCfg(
            static_friction=2.0, dynamic_friction=2.0, restitution=0.0
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.5,3.5,3.5], target=[0.0,0.0,0.0])

    scene_cfg = RaycasterSensorSceneCfg(num_envs=args.num_envs, env_spacing=2.0)
    mesh_paths = ["/World/Ground"] + [
        f"/World/envs/env_{eid}/Obstacle{j+1}"
        for eid in range(args.num_envs) for j in range(3)
    ]
    scene_cfg.ray_caster.mesh_prim_paths = mesh_paths
    scene = InteractiveScene(scene_cfg)

    run_simulator(sim, scene, simulation_app)
