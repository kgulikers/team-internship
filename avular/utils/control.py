import torch
from typing import List

def compute_wheel_targets(
    joint_names: List[str],
    v: torch.Tensor,
    w: torch.Tensor,
    track_width: float,
    wheel_radius: float
) -> torch.Tensor:
    """
    Compute per-wheel angular velocity targets for a differential-drive robot.

    Args:
        joint_names: list of joint name strings (used to select left/right wheels)
        v: forward linear velocity tensor, shape (num_envs,)
        w: yaw rate tensor, shape (num_envs,)
        track_width: distance between left & right wheels (m)
        wheel_radius: wheel radius (m)

    Returns:
        vel_target: tensor of shape (num_envs, num_joints), where each column
                    corresponding to a wheel joint gets its angular velocity target.
    """
    # identify right-wheel indices
    right_idxs = [
        i for i, name in enumerate(joint_names)
        if "wheel" in name.lower() and "right" in name.lower()
    ]
    # all other wheel joints are left
    left_idxs = [i for i in range(len(joint_names)) if i not in right_idxs]

    base_omega  = v / wheel_radius
    delta_omega = (w * track_width * 0.5) / wheel_radius
    omega_left  = base_omega - delta_omega
    omega_right = base_omega + delta_omega

    num_envs   = v.shape[0]
    num_joints = len(joint_names)
    vel_target = torch.zeros((num_envs, num_joints), device=v.device)

    vel_target[:, left_idxs]  = omega_left.unsqueeze(-1)
    vel_target[:, right_idxs] = omega_right.unsqueeze(-1)
    return vel_target
