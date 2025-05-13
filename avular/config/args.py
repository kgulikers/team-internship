import argparse
from isaaclab.app import AppLauncher

def get_args():
    parser = argparse.ArgumentParser(
        description="Spawn robot + LiDAR, drive wheels with keyboard, log scans to W&B."
    )
    # simulation parallelism & physics tuning
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of parallel environments")
    parser.add_argument("--gpu_collision_stack_size", type=int, default=130_000_000)
    parser.add_argument("--gpu_max_rigid_patch_count", type=int, default=2_000_000)
    parser.add_argument("--gpu_found_lost_pairs_capacity", type=int, default=2_000_000)

    # project naming
    parser.add_argument("--project", type=str, default="isaac-sim-demo",
                        help="W&B project name")
    parser.add_argument("--entity", type=str, default=None,
                        help="W&B user or team entity")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name (defaults to timestamp)")

    # control parameters
    parser.add_argument("--wheel_speed", type=float, default=2.0,
                        help="Nominal wheel linear velocity (m/s)")
    parser.add_argument("--yaw_rate",   type=float, default=4.0,
                        help="Nominal yaw rate (rad/s)")

    # Isaac Sim launcher args (e.g. --headless, --device, etc.)
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    # Create the AppLauncher early so that --headless / --renderer get set before imports
    app_launcher = AppLauncher(args)
    return args, app_launcher
