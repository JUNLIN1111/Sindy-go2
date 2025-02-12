# --------------------------------------------------------
# Intended to be used during dynamics models predict(x, u)
# --------------------------------------------------------

import numpy as np

def project_cartpole(z):
    '''Projecting the dm_control swingup task back onto the circle'''
    cart_pos, cos_th, sin_th, dx, dth = z
    
    u = np.array([cos_th, sin_th])
    new_cos, new_sin = u/np.linalg.norm(u)
    return np.array([cart_pos, new_cos, new_sin, dx, dth])


def project_go2(z):
    """
    将 Go2 机器人的状态投影到合理的范围内。
    
    参数：
    - z: Go2 的状态向量，形状由 _get_obs 方法决定。
    
    返回：
    - projected_z: 投影后的状态向量，形状与输入一致。
    """
    # 假设 z 的结构与 _get_obs 方法一致
    # 分解状态
    base_position = z[:3]  # 基座位置 (x, y, z)
    base_orientation = z[3:7]  # 基座姿态（四元数）
    q_joints = z[7:19]  # 关节角度 (12,)
    dq_joints = z[19:31]  # 关节速度 (12,)
    base_linear_velocity = z[31:34]  # 基座线速度 (3,)
    base_angular_velocity = z[34:37]  # 基座角速度 (3,)
    foot_contact = z[37:41]  # 足端接触信息 (4,)
    external_forces = z[41:44]  # 外部力信息 (3,)
    joint_torques = z[44:62]  # 关节力矩 (12,)
    goal_info = z[62:]  # 目标信息（如果有）

    # 1. 关节位置投影：将角度限制在 [-π, π] 范围内
    q_joints = np.arctan2(np.sin(q_joints), np.cos(q_joints))

    # 2. 关节速度投影：限制在合理范围内（假设最大速度为 10 rad/s）
    dq_joints = np.clip(dq_joints, -10, 10)

    # 3. 基座姿态投影：归一化四元数
    base_orientation = base_orientation 
    
    # 4. 基座角速度投影：限制在合理范围内（假设最大角速度为 5 rad/s）
    base_angular_velocity = np.clip(base_angular_velocity, -5, 5)

    # 5. 基座线速度投影：限制在合理范围内（假设最大线速度为 2 m/s）
    base_linear_velocity = np.clip(base_linear_velocity, -2, 2)

    # 6. 足端接触信息投影：限制在 [0, 1] 范围内（假设已经是二进制值）
    foot_contact = np.clip(foot_contact, 0, 1)

    # 7. 外部力投影：限制在合理范围内（假设最大力为 100 N）
    external_forces = np.clip(external_forces, -100, 100)

    # 8. 关节力矩投影：限制在合理范围内（假设最大力矩为 50 Nm）
    joint_torques = np.clip(joint_torques, -50, 50)

    # 9. 目标信息投影：如果有目标信息，限制在合理范围内
    if len(goal_info) > 0:
        goal_info = np.clip(goal_info, -10, 10)  # 假设目标信息在 [-10, 10] 范围内

    # 返回投影后的状态
    projected_z = np.concatenate([
        base_position, base_orientation,  # 机器人基座信息
        q_joints, dq_joints,  # 机器人关节状态
        base_linear_velocity, base_angular_velocity,  # 机器人运动状态
        foot_contact, external_forces, joint_torques,  # 触地信息 & 外部力 & 关节力矩
        goal_info  # 目标信息
    ])
    return projected_z
