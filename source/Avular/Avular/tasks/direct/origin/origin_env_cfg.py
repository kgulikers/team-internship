from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.utils import configclass

@configclass
class OriginDriveEnvCfg(DirectRLEnvCfg):
    # Inherit defaults (dt, decimation, etc.)
    sim = DirectRLEnvCfg.sim

    # Scene: ground + LiDAR sensor
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=2.0,
        ground=GroundPlaneCfg(),
        ray_caster=RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/main_body",
            update_period=1,
            offset=RayCasterCfg.OffsetCfg(pos=(0,0,0.5)),
            pattern_cfg=patterns.LidarPatternCfg(
                channels=32,
                vertical_fov_range=(-22.5,22.5),
                horizontal_fov_range=(-10.0,10.0),
                horizontal_res=1.0,
            ),
            debug_vis=False,
        ),
    )
