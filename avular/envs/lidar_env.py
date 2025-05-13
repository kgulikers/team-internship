# avular/avular/envs/lidar_env.py

import gym
import numpy as np
from gym import spaces

class LidarDriveEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, num_envs=1):
        super().__init__()

        # === Lazy imports: only now we pull in IsaacLab/Omniverse ===
        from avular.config.args import get_args
        from avular.simulation.launcher import launch_simulation
        from avular.scene.raycaster_scene import RaycasterSensorSceneCfg
        from isaaclab.scene import InteractiveScene
        from avular.utils.control import compute_wheel_targets
        from avular.constants.robot import TRACK_WIDTH, WHEEL_RADIUS

        # Launch the sim — this starts the OmniverseKit process and makes
        # 'carb' available on PYTHONPATH for all subsequent imports.
        args, app_launcher = get_args()
        self.sim, self.sim_app = launch_simulation(app_launcher, args)

        # Build the scene
        scene_cfg = RaycasterSensorSceneCfg(num_envs=num_envs, env_spacing=2.0)
        mesh = ["/World/Ground"] + [
            f"/World/envs/env_{i}/Obstacle{j+1}"
            for i in range(num_envs) for j in range(3)
        ]
        scene_cfg.ray_caster.mesh_prim_paths = mesh
        self.scene = InteractiveScene(scene_cfg)

        # Observation & action spaces
        self.observation_space = spaces.Box(0.0, 100.0, (32, 360), np.float32)
        self.action_space = spaces.Box(
            low=np.array([-args.wheel_speed, -args.yaw_rate]),
            high=np.array([ args.wheel_speed,  args.yaw_rate]),
            dtype=np.float32
        )

        # Save for later use
        self._compute_wheel_targets = compute_wheel_targets
        self._track_width = TRACK_WIDTH
        self._wheel_radius = WHEEL_RADIUS
        self._num_envs = num_envs

        # warm up
        self._reset_sim()

    def _reset_sim(self):
        dt = self.sim.get_physics_dt()
        self.sim.reset()
        for _ in range(50):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt)

    def reset(self):
        self._reset_sim()
        data = self.scene.sensors["ray_caster"].data
        return data.ranges.cpu().numpy()[0]

    def step(self, action):
        # again, lazily get everything we need
        import torch

        v, w = action
        art = self.scene.articulations["robot"]
        joint_names = art.data.joint_names
        device = art.data.joint_vel.device

        v_t = torch.full((self._num_envs,), v, device=device)
        w_t = torch.full((self._num_envs,), w, device=device)
        vel = self._compute_wheel_targets(
            joint_names, v_t, w_t, self._track_width, self._wheel_radius
        )
        art.set_joint_velocity_target(vel)

        # step physics
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim.get_physics_dt())

        ranges = self.scene.sensors["ray_caster"].data.ranges.cpu().numpy()[0]
        pos = art.data.root_pos.cpu().numpy()[0]
        reward = pos[0]  # simple forward-distance reward
        done = False
        return ranges, reward, done, {}

    def render(self, mode="human"):
        pass

    def close(self):
        self.sim_app.close()
