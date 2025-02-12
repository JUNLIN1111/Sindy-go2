import numpy as np
from dm_control.utils.rewards import tolerance
import torch
from scipy.spatial.transform import Rotation

def go2_straight_walk_reward(obs, action):
    """
    奖励函数设计目标：
    1. 鼓励机器人向前行走（最大化基座前进速度）。
    2. 保持机器人姿态稳定（最小化基座俯仰角、侧倾角）。
    3. 减少能量消耗（最小化关节扭矩）。
    4. 避免不稳定的运动（惩罚关节速度过大）。
    5. 保持足端接触稳定（惩罚滑移或悬空）。

    参数：
    - obs: 62 维观察向量，由 _get_obs 返回。
    - action: 当前动作向量，形状为 (12,)，表示关节力矩。

    返回：
    - reward: 综合奖励值
    """
    # 分解观察值（假设 obs 结构如下）
    base_linear_velocity = obs[31:34]  # 基座线速度 (x, y, z)
    base_orientation = obs[3:7]        # 基座姿态（四元数）
    q_joints = obs[7:19]               # 关节角度 (12,)
    dq_joints = obs[19:31]             # 关节速度 (12,)
    foot_contact = obs[37:41]          # 足端接触信息 (4,)
    joint_torques = obs[44:62]         # 关节力矩 (18,)

    # 1. 前进速度奖励（基座 x 方向线速度）
    forward_vel = base_linear_velocity[0]  # 假设 x 方向为前进方向
    vel_reward = 10.0 * forward_vel

    # 2. 姿态稳定惩罚（从四元数计算俯仰角、侧倾角）
    # 四元数转换为欧拉角（假设使用 ZYX 顺序）
    
    
    #if np.linalg.norm(base_orientation) == 0:
     #   base_orientation = np.array([1, 0, 0, 0])  # 设置为单位四元数
    #rot = Rotation.from_quat(base_orientation)
    #pitch, roll, _ = rot.as_euler('zyx', degrees=False)  # 俯仰角、侧倾角
    #posture_penalty = 2.0 * (pitch**2 + roll**2)

    # 3. 能量消耗惩罚（关节力矩的平方和）
    energy_cost = 0.01 * np.sum(np.square(joint_torques))

    # 4. 关节速度惩罚（避免不稳定的运动）
    joint_speed_penalty = 0.005 * np.sum(np.square(dq_joints))

    # 5. 足端接触稳定性惩罚（鼓励足端稳定接触地面）
    # 若足端接触标志为 0（悬空），则惩罚
    foot_air_penalty = 0.1 * np.sum(1.0 - foot_contact)

    # 综合奖励
    reward = (
        vel_reward 
        - energy_cost 
        #- posture_penalty
        - joint_speed_penalty 
        - foot_air_penalty
    )
    print(f"reward:",reward)
    return reward
#robot = Go1EnvWithBounds()

'''
def _reward_lin_vel_z(robot):
    # Penalize z axis base linear velocity
    #return torch.square(self.base_lin_vel[:, 2])
    return torch.square(robot.velocity[2])

def _reward_ang_vel_xy(robot):
    # Penalize xy axes base angular velocity
    #return torch.sum(torch.square(robot.base_ang_vel[:, :2]), dim=1)
    return torch.sum(torch.square(robot.velocity[:2]), dim=1)

def _reward_orientation(robot):
    # Penalize non flat base orientation
    return torch.sum(torch.square(robot.projected_gravity[:, :2]), dim=1)

def _reward_base_height(robot):
    # Penalize base height away from target
    base_height = torch.mean(robot.root_states[:, 2].unsqueeze(1) - robot.measured_heights, dim=1)
    return torch.square(base_height - robot.cfg.rewards.base_height_target)

def _reward_torques(robot):
    # Penalize torques
    return torch.sum(torch.square(robot.torques), dim=1)

def _reward_dof_vel(robot):
    # Penalize dof velocities
    return torch.sum(torch.square(robot.dof_vel), dim=1)

def _reward_dof_acc(robot):
    # Penalize dof accelerations
    return torch.sum(torch.square((robot.last_dof_vel - robot.dof_vel) / robot.dt), dim=1)

def _reward_action_rate(robot):
    # Penalize changes in actions
    return torch.sum(torch.square(robot.last_actions - robot.actions), dim=1)

def _reward_collision(robot):
    # Penalize collisions on selected bodies
    return torch.sum(1.*(torch.norm(robot.contact_forces[:, robot.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

def _reward_termination(robot):
    # Terminal reward / penalty
    return robot.reset_buf * ~robot.time_out_buf

def _reward_dof_pos_limits(robot):
    # Penalize dof positions too close to the limit
    out_of_limits = -(robot.dof_pos - robot.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
    out_of_limits += (robot.dof_pos - robot.dof_pos_limits[:, 1]).clip(min=0.)
    return torch.sum(out_of_limits, dim=1)

def _reward_dof_vel_limits(robot):
    # Penalize dof velocities too close to the limit
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    return torch.sum((torch.abs(robot.dof_vel) - robot.dof_vel_limits*robot.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

def _reward_torque_limits(robot):
    # penalize torques too close to the limit
    return torch.sum((torch.abs(robot.torques) - robot.torque_limits*robot.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

def _reward_tracking_lin_vel(robot):
    # Tracking of linear velocity commands (xy axes)
    lin_vel_error = torch.sum(torch.square(robot.commands[:, :2] - robot.base_lin_vel[:, :2]), dim=1)
    return torch.exp(-lin_vel_error/robot.cfg.rewards.tracking_sigma)

def _reward_tracking_ang_vel(robot):
    # Tracking of angular velocity commands (yaw) 
    ang_vel_error = torch.square(robot.commands[:, 2] - robot.base_ang_vel[:, 2])
    return torch.exp(-ang_vel_error/robot.cfg.rewards.tracking_sigma)

def _reward_feet_air_time(robot):
    # Reward long steps
    # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    contact = robot.contact_forces[:, robot.feet_indices, 2] > 1.
    contact_filt = torch.logical_or(contact, robot.last_contacts) 
    robot.last_contacts = contact
    first_contact = (robot.feet_air_time > 0.) * contact_filt
    robot.feet_air_time += robot.dt
    rew_airTime = torch.sum((robot.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
    rew_airTime *= torch.norm(robot.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    robot.feet_air_time *= ~contact_filt
    return rew_airTime

def _reward_stumble(robot):
    # Penalize feet hitting vertical surfaces
    return torch.any(torch.norm(robot.contact_forces[:, robot.feet_indices, :2], dim=2) >\
        5 *torch.abs(robot.contact_forces[:, robot.feet_indices, 2]), dim=1)
    
def _reward_stand_still(robot):
    # Penalize motion at zero commands
    return torch.sum(torch.abs(robot.dof_pos - robot.default_dof_pos), dim=1) * (torch.norm(robot.commands[:, :2], dim=1) < 0.1)

def _reward_feet_contact_forces(robot):
    # penalize high contact forces
    return torch.sum((torch.norm(robot.contact_forces[:, robot.feet_indices, :], dim=-1) -  robot.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

def go_reward():
    return _reward_lin_vel_z()+_reward_ang_vel_xy()'''