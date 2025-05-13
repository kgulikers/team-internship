import sys
import time
import termios
import tty
import select

import torch
import numpy as np
import wandb

from avular.utils.control import compute_wheel_targets
from avular.constants.robot import TRACK_WIDTH, WHEEL_RADIUS

def run_simulator(sim, scene, sim_app, args, alpha=0.1):
    """
    Interactive loop: read keyboard, drive the robot, log to W&B.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    LOG_INTERVAL = 5.0
    last_log_time = time.time()

    try:
        sim_dt = sim.get_physics_dt()
        sim.reset()
        # Warm-up
        for _ in range(50):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

        step_count = 0
        print("[INFO]: Running simulation (W/S/A/D to steer, Q to quit)…")

        robot_art = scene.articulations["robot"]
        joint_names = robot_art.data.joint_names
        device = robot_art.data.joint_vel.device
        prev_vel_target = torch.zeros_like(robot_art.data.joint_vel)

        while sim_app.is_running():
            sim_app.update()

            # Keyboard input
            v_cmd, w_cmd = 0.0, 0.0
            if select.select([sys.stdin], [], [], 0)[0]:
                c = sys.stdin.read(1).lower()
                if c == 'w':
                    v_cmd = args.wheel_speed
                elif c == 's':
                    v_cmd = -args.wheel_speed
                elif c == 'a':
                    w_cmd = args.yaw_rate
                elif c == 'd':
                    w_cmd = -args.yaw_rate
                elif c == 'q':
                    print("[INFO]: Quit command received.")
                    break

            # Deadzone
            if abs(v_cmd) < 1e-3 and abs(w_cmd) < 1e-3:
                v_cmd, w_cmd = 0.0, 0.0

            # Compute & filter wheel targets
            v_tensor = torch.full((args.num_envs,), v_cmd, device=device)
            w_tensor = torch.full((args.num_envs,), w_cmd, device=device)
            raw_vel = compute_wheel_targets(
                joint_names, v_tensor, w_tensor, TRACK_WIDTH, WHEEL_RADIUS
            )
            vel_target = alpha * raw_vel + (1 - alpha) * prev_vel_target
            prev_vel_target = vel_target.clone()
            robot_art.set_joint_velocity_target(vel_target)

            # Step physics
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            # Log base height
            pos_w = scene.sensors["ray_caster"].data.pos_w.cpu().numpy()[0]
            wandb.log({"base_height": pos_w[2]}, step=step_count)

            # Upload full LiDAR cloud periodically
            hits = scene.sensors["ray_caster"].data.ray_hits_w.cpu().numpy()[0]
            finite = hits[np.isfinite(hits).all(axis=1)]
            now = time.time()
            if finite.size > 0 and (now - last_log_time) >= LOG_INTERVAL:
                colors = np.tile([0, 255, 0], (finite.shape[0], 1))
                pc3d_full = np.hstack((finite, colors))
                wandb.log(
                    {"lidar_point_cloud_full": wandb.Object3D(pc3d_full)},
                    step=step_count
                )
                last_log_time = now

            step_count += 1

        print("[INFO]: Simulation stopped.")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
