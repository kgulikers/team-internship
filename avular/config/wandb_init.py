import os
import wandb

def init_wandb(args):
    """
    Initialize Weights & Biases run.
    Reads WANDB_API_KEY from env or prompts login.
    """
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        # will open browser or prompt for login
        wandb.login()

    # start the run
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
            # you can add more config entries here if desired
        }
    )
