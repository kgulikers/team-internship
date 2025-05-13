import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from avular.constants.robot import (
    TRACK_WIDTH, WHEEL_RADIUS,
    GROUND_THICKNESS, MARGIN, ZERO_INIT_Z
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

@configclass
class RaycasterSensorSceneCfg(InteractiveSceneCfg):
    # ─── Ground Plane ────────────────────────────────────────────────────────
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

    # ─── Dome Light ──────────────────────────────────────────────────────────
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    # ─── Robot ────────────────────────────────────────────────────────────────
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/avular_origin_v18",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(
                _PROJECT_ROOT
                / "source"
                / "Avular_assets"
                / "origin_v18"
                / "origin_v18.urdf"
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
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.0, 0.0, ZERO_INIT_Z],
            rot=[1, 0, 0, 0],
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "wheel_act": IdealPDActuatorCfg(
                joint_names_expr=[".*wheel.*"],
                stiffness=3.0,
                damping=60.0,
                effort_limit=20.0,
                velocity_limit=None,  # will be set at runtime from args.wheel_speed
            )
        },
    )

    # ─── Obstacles ───────────────────────────────────────────────────────────
    obstacle1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[5.0, 0.0, 1.5], rot=[1, 0, 0, 0]
        ),
        spawn=sim_utils.MeshCuboidCfg(
            size=(2.5, 2.5, 3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.01, rest_offset=0.0
            ),
            physics_material=RigidBodyMaterialCfg(
                static_friction=2.0, dynamic_friction=2.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2)
            ),
        ),
    )
    obstacle2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle2",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-5.0, -3.0, 1.0], rot=[1, 0, 0, 0]
        ),
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.5, 4.0, 2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.01, rest_offset=0.0
            ),
            physics_material=RigidBodyMaterialCfg(
                static_friction=2.0, dynamic_friction=2.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.8, 0.2)
            ),
        ),
    )
    obstacle3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle3",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 5.0, 2.0], rot=[1, 0, 0, 0]
        ),
        spawn=sim_utils.MeshCuboidCfg(
            size=(3.0, 1.0, 4.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.01, rest_offset=0.0
            ),
            physics_material=RigidBodyMaterialCfg(
                static_friction=2.0, dynamic_friction=2.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.2, 0.8)
            ),
        ),
    )

    # ─── LiDAR Sensor ────────────────────────────────────────────────────────
    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/avular_origin_v18/main_body",
        update_period=1,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        attach_yaw_only=False,
        mesh_prim_paths=[],  # fill in main.py before build()
        pattern_cfg=patterns.LidarPatternCfg(
            channels=32,
            vertical_fov_range=(-22.5, 22.5),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
        ),
        debug_vis=False,
    )
