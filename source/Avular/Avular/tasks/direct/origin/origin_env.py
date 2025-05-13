import torch, math
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import spawn_ground_plane, GroundPlaneCfg
from isaaclab.assets import Articulation
from avular.assets.origin_robot_cfg import OriginRobotCfg
from .origin_env_cfg import OriginDriveEnvCfg

class OriginDriveEnv(DirectRLEnv):
    cfg: OriginDriveEnvCfg

    def _setup_scene(self):
        # spawn ground and environments
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        # load the Origin robot
        self.robot = Articulation(OriginRobotCfg)
        self.scene.articulations["robot"] = self.robot

    def _get_observations(self):
        # LiDAR hits: shape (num_envs, rays, 3) → flatten
        lidar = self.scene.sensors["ray_caster"].data.ray_hits_w
        obs = lidar.reshape(self.num_envs, -1)
        return {"lidar": obs}

    def _get_rewards(self):
        # forward velocity reward minus lateral drift penalty
        linvel = self.robot.data.root_linvel  # (num_envs,3)
        pos    = self.robot.data.root_pose[:, :3]
        forward = linvel[:, 0]
        lateral = pos[:, 1].abs()
        return forward - 5.0 * lateral

    def _get_dones(self):
        pos = self.robot.data.root_pose[:, :3]
        out_of_bounds = pos[:, 1].abs() > 0.5     # >0.5 m sideways  
        time_out = self.episode_length_buf >= self.cfg.max_episode_steps - 1
        return out_of_bounds, time_out
