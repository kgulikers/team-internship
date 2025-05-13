#!/usr/bin/env python3
import os
import sys

# ENV flags for CPU-only PhysX & headful rendering
os.environ["OMNI_PHYSX_USE_GPU_NARROWPHASE"] = "0"
os.environ["OMNI_PHYSX_USE_GPU"]           = "0"
os.environ["OMNI_RENDER_HEADLESS"]        = "0"
os.environ["HEADLESS"]                    = "0"
os.environ["LIVESTREAM"]                  = "0"

def main():
    # minimal import to get Omni going
    from avular.config.args import get_args
    args, app_launcher = get_args()

    # now that Omni’s up, pull in everything else
    from avular.config.wandb_init      import init_wandb
    from avular.scene.raycaster_scene import RaycasterSensorSceneCfg
    from avular.simulation.launcher   import launch_simulation
    from avular.simulation.runner     import run_simulator
    from isaaclab.scene                import InteractiveScene

    # init W&B
    init_wandb(args)

    # launch sim + get sim_app
    sim, sim_app = launch_simulation(app_launcher, args)

    # configure scene
    scene_cfg = RaycasterSensorSceneCfg(num_envs=args.num_envs, env_spacing=2.0)
    mesh_paths = ["/World/Ground"] + [
        f"/World/envs/env_{i}/Obstacle{j+1}"
        for i in range(args.num_envs)
        for j in range(3)
    ]
    scene_cfg.ray_caster.mesh_prim_paths = mesh_paths

    # **instantiate** the scene
    scene = InteractiveScene(scene_cfg)

    # run loop
    run_simulator(sim, scene, sim_app, args)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR]: {e}", file=sys.stderr)
        sys.exit(1)
