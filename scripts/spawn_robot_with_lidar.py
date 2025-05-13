#!/usr/bin/env python3
import os
# ─────────────────────────────────────────────────────────────────────────────
# FORCE CPU-ONLY PHYSX & HEADLESS RENDERING
os.environ["OMNI_PHYSX_USE_GPU_NARROWPHASE"] = "0"
os.environ["OMNI_PHYSX_USE_GPU"]           = "0"
os.environ["OMNI_RENDER_HEADLESS"]       = "1"
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing & immediate AppLauncher startup (before heavy imports)
import argparse
from isaaclab.app import AppLauncher

descr = (
    "Spawn robot + LiDAR, drive wheels straight, accumulate LiDAR scans, log to W&B."
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
    "--wheel_speed", type=float, default=1.0,
    help="Wheel linear velocity target (m/s)"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Instantly launch the Simulation App before other imports
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
# ─────────────────────────────────────────────────────────────────────────────

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import wandb
import isaaclab.sim as sim_utils
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.sim.schemas import RigidBodyPropertiesCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.utils import configclass

# ─────────────────────────────────────────────────────────────────────────────
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
    }
)
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class RaycasterSensorSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.MeshCuboidCfg(
            size=(50.0, 50.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.05, rest_offset=0.0),
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75,0.75,0.75)),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/avular_origin_v10",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "source", "Avular_assets", "origin_v10.usd")
            ),
            scale=(1,1,1),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=2,
                linear_damping=0.5,
                angular_damping=0.5,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
                max_linear_velocity=100.0,
                max_angular_velocity=360.0,
                sleep_threshold=0.001,
                stabilization_threshold=0.0001,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=2,
                sleep_threshold=0.001,
                stabilization_threshold=0.0001,
                enabled_self_collisions=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.0,0.0,2.0], rot=[1,0,0,0], joint_pos={".*": 0.0}
        ),
        actuators={
            "wheel_act": IdealPDActuatorCfg(
                joint_names_expr=[".*wheel.*"],
                stiffness=200.0,
                damping=20.0,
                effort_limit=100.0,
                velocity_limit=args.wheel_speed,
            )
        },
    )
    obstacle1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[5.0,0.0,1.5], rot=[1,0,0,0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(2.5,2.5,3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8,0.2,0.2)),
        ),
    )
    obstacle2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-5.0,-3.0,1.0], rot=[1,0,0,0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.5,4.0,2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2,0.8,0.2)),
        ),
    )
    obstacle3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0,5.0,2.0], rot=[1,0,0,0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(3.0,1.0,4.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2,0.2,0.8)),
        ),
    )
    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/avular_origin_v10/main_body",
        update_period=1,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0,0.0,0.5)),
        attach_yaw_only=False,
        mesh_prim_paths=[],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=32,
            vertical_fov_range=(-22.5, 22.5),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
        ),
        debug_vis=False,
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, sim_app):
    sim_dt = sim.get_physics_dt()
    sim.reset()
    print("[INFO]: Running simulation...")

    robot_art = scene.articulations["robot"]
    joint_names = robot_art.data.joint_names
    n_envs, n_joints = robot_art.data.joint_vel.shape
    vel_target = torch.zeros((n_envs, n_joints), device=robot_art.data.joint_vel.device)

    scan_traj, scan_points = [], []
    frames = []  # for video
    sim_time, next_log = 0.0, 1.0

    # Loop until the app is closed, no max time enforced
    while sim_app.is_running():
        sim_app.update()

        # ==== skid‑steer velocity command ====
        # tune these to your robot
        linear_speed = args.wheel_speed      # forward speed [m/s]
        yaw_rate     = 0.5                   # turn rate [rad/s]
        track_width  = 0.6                   # wheelbase width [m]
        wheel_radius = 0.1175                # wheel radius [m]

        # base wheel speed is already given as an angular velocity [rad/s]
        base_omega = args.wheel_speed  

        # if you ever want to turn, set yaw_rate != 0; for straight, leave at 0
        yaw_rate = 0.0  
        track_width = 0.6    # [m], distance between left/right wheels
        wheel_radius = 0.1175 # [m]

        # the angular offset needed on each side to achieve yaw_rate:
        delta_omega = (yaw_rate * track_width * 0.5) / wheel_radius

        omega_left  = base_omega - delta_omega
        omega_right = base_omega + delta_omega

        # split indices by side
        left_idxs  = [i for i,n in enumerate(joint_names)
                      if "wheel_fl" in n.lower() or "wheel_rl" in n.lower()]
        right_idxs = [i for i,n in enumerate(joint_names)
                      if "wheel_fr" in n.lower() or "wheel_rr" in n.lower()]

        vel_target[:, left_idxs]  = omega_left
        vel_target[:, right_idxs] = omega_right


        robot_art.set_joint_velocity_target(vel_target)
        scene.write_data_to_sim()

        sim.step()
        scene.update(sim_dt)

        data = scene.sensors["ray_caster"].data
        pos_w = data.pos_w.cpu().numpy()[0]
        hits  = data.ray_hits_w.cpu().numpy()[0]
        finite = hits[np.isfinite(hits).all(axis=1)]
        scan_traj.append((pos_w[0], pos_w[1]))
        scan_points.extend(finite[finite[:,2] > 0.0].tolist())

        sim_time += sim_dt
        if sim_time >= next_log - 1e-6:
            traj = np.array(scan_traj)
            cloud = np.array(scan_points)
            fig, ax = plt.subplots(figsize=(6,6))
            if cloud.size > 0:
                # subsample 40%
                num_pts = cloud.shape[0]
                k = max(1, int(0.4 * num_pts))
                idxs = np.random.choice(num_pts, k, replace=False)
                cloud_vis = cloud[idxs]
                sc = ax.scatter(cloud_vis[:,0], cloud_vis[:,1], c=cloud_vis[:,2], s=2, alpha=0.6, cmap='viridis')
                fig.colorbar(sc, ax=ax, label="height (m)")
            else:
                cloud_vis = cloud
            ax.plot(traj[:,0], traj[:,1], '-m', lw=1.5, label="trajectory")
            ax.scatter(traj[0,0], traj[0,1], c='g', s=50, marker='o', label="start")
            ax.scatter(traj[-1,0], traj[-1,1], c='r', s=50, marker='X', label="end")
            ax.set_aspect('equal','box')
            ax.set(xlabel="X (m)", ylabel="Y (m)", title="LiDAR Scan")
            ax.legend(loc="upper right")
            plt.tight_layout()

            # capture frame
            canvas = FigureCanvas(fig)
            canvas.draw()
            buf, (w, h) = canvas.print_to_buffer()
            img = np.frombuffer(buf, dtype='uint8').reshape(h, w, 4)[:,:,:3]
            frames.append(img)

            # log subsampled point cloud
            colors = np.tile([0,255,0], (cloud_vis.shape[0],1))
            pc3d = np.hstack((cloud_vis, colors))
            wandb.log({
                "lidar_scan_point_cloud": wandb.Object3D(pc3d)
            }, step=int(next_log))
            plt.close(fig)

            scan_traj.clear()
            scan_points.clear()
            next_log += 1.0

    print("[INFO]: Simulation stopped.")

    # log video at end
    if frames:
        vid = np.stack(frames)
        wandb.log({"lidar_scan_video": wandb.Video(vid, fps=1, format="mp4")})

if __name__ == "__main__":
    from isaaclab.sim.schemas import RigidBodyPropertiesCfg
    # Physics and sim setup
    physx_cfg = PhysxCfg(
        gpu_collision_stack_size=args.gpu_collision_stack_size,
        gpu_max_rigid_patch_count=args.gpu_max_rigid_patch_count,
        gpu_found_lost_pairs_capacity=args.gpu_found_lost_pairs_capacity,
        enable_ccd=True
    )
    sim_cfg = SimulationCfg(
        dt=1.0/60.0,
        device=args.device,
        physx=physx_cfg,
        gravity=(0.0, 0.0, -9.81),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.5,3.5,3.5], target=[0.0,0.0,0.0])

    # Scene setup
    scene_cfg = RaycasterSensorSceneCfg(num_envs=args.num_envs, env_spacing=2.0)
    mesh_paths = ["/World/Ground"] + [
        f"/World/envs/env_{eid}/Obstacle{j+1}"
        for eid in range(args.num_envs)
        for j in range(3)
    ]
    scene_cfg.ray_caster.mesh_prim_paths = mesh_paths
    scene = InteractiveScene(scene_cfg)

    # Run simulation loop
    run_simulator(sim, scene, simulation_app)
