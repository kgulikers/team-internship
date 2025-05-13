import isaaclab.sim as sim_utils
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg

def launch_simulation(app_launcher, args):
    """
    Build and return:
      - sim: the SimulationContext
      - sim_app: the running Isaac Sim application
    """
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
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    sim_app = app_launcher.app
    return sim, sim_app
