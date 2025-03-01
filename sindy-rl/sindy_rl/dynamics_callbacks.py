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
'''
def project_latent_space(z):
    """
    将潜在空间向量投影到合理范围（适用于24维VAE编码输出）
    
    参数：
    - z: 24维潜在空间向量
    
    返回：
    - projected_z: 投影后的24维向量
    """
    # 输入验证
    assert len(z) == 24, f"潜在空间维度应为24，当前输入维度为{len(z)}"
    
    # 创建副本避免修改原始数据
    projected_z = np.copy(z)
    
    # 1. 核心维度约束（基于典型VAE潜在空间统计特性）
    # 前3维（通常编码运动方向）限制在[-3, 3]范围
    projected_z[:3] = np.clip(z[:3], -3.0, 3.0)
    
    # 2. 中间维度动态缩放（4-18维）
    # 应用双曲正切函数约束在(-1,1)范围
    projected_z[4:18] = np.tanh(z[4:18]) * 1.5  # 保留15%余量
    
    # 3. 高频噪声抑制（最后6维）
    # 添加低通滤波：当前值保留90%，前值10%（需维护状态）
    if not hasattr(project_latent_space, "prev_z"):
        project_latent_space.prev_z = np.zeros(6)
    projected_z[-6:] = 0.9 * z[-6:] + 0.1 * project_latent_space.prev_z
    project_latent_space.prev_z = projected_z[-6:]
    
    # 4. 整体幅值约束
    norm = np.linalg.norm(projected_z)
    if norm > 5.0:  # 经验阈值
        projected_z = projected_z * (5.0 / norm)
        
    return projected_z

'''
def project_latent_space(z):
    """
    将潜在空间向量投影到合理范围（适用于 18 维 VAE 编码输出）

    参数：
    - z: 18 维潜在空间向量

    返回：
    - projected_z: 投影后的 18 维向量
    """
    # 输入验证
    assert len(z) == 18, f"潜在空间维度应为 18，当前输入维度为 {len(z)}"
    
    # 创建副本避免修改原始数据
    projected_z = np.copy(z)
    
    # 1. **基座核心信息（前 4 维）**
    projected_z[0] = np.clip(z[0], 0.1, 0.6)   # 基座高度
    projected_z[1] = np.clip(z[1], -1.5, 1.5)  # 俯仰角
    projected_z[2] = np.clip(z[2], 0.0, 3.0)   # 水平速度
    projected_z[3] = np.clip(z[3], -2.0, 2.0)  # 偏航角速度

    # 2. **关节状态（中间 8 维）**
    projected_z[4:6] = np.clip(z[4:6], -1.5, 1.5)   # 前腿关节角度
    projected_z[6:8] = np.clip(z[6:8], -2.5, 2.5)   # 前腿关节速度
    projected_z[8:10] = np.clip(z[8:10], -1.5, 1.5) # 后腿关节角度
    projected_z[10:12] = np.clip(z[10:12], -2.5, 2.5) # 后腿关节速度

    # 3. **步态相位（2 维）**
    projected_z[12:14] = np.clip(z[12:14], -1.0, 1.0)  # 步态相位（正弦 & 余弦）

    # 4. **平衡关键指标（4 维）**
    projected_z[14:16] = np.clip(z[14:16], -1.0, 1.0)  # 质心水平投影 X & Y
    projected_z[16] = np.clip(z[16], 0.0, 2.0)         # 动能/势能比
    projected_z[17] = np.clip(z[17], 0.0, 0.5)         # 支撑多边形面积

    # 5. **整体幅值约束**
    norm = np.linalg.norm(projected_z)
    if norm > 5.0:
        projected_z = projected_z * (5.0 / norm)
        
    return projected_z

import numpy as np

import numpy as np

def project_low_space(obs):
    # 分解观测值（确保obs长度为13）
    (
        base_height,                  # [1]
        base_orientation,             # [1] 假设是标量（如角度）
        base_lin_velocity_x,          # [2]
        base_lin_velocity_y,
        base_ang_velocity,            # [1]
        front_angle_diff_1,           # [2]
        front_angle_diff_2,
        front_vel_mean_1,             # [2]
        front_vel_mean_2,
        rear_angle_diff_1,            # [2]
        rear_angle_diff_2,
        rear_vel_mean_1,              # [2]
        rear_vel_mean_2
    ) = obs

    # 将标量角度转换为二维方向向量并归一化
    theta = base_orientation  # 假设是方向角
    orientation_vec = np.array([np.cos(theta), np.sin(theta)])
    norm = np.linalg.norm(orientation_vec)
    if norm < 1e-8:
        orientation_normalized = np.array([1.0, 0.0])  # 默认方向
    else:
        orientation_normalized = orientation_vec / norm

    # 将归一化后的方向向量重新映射回标量（使用反正切）
    new_theta = np.arctan2(orientation_normalized[1], orientation_normalized[0])

    # 重构观测值（保持13个元素）
    projected_obs = [
        base_height,
        new_theta,  # 替换原base_orientation为归一化后的角度
        base_lin_velocity_x,
        base_lin_velocity_y,
        base_ang_velocity,
        front_angle_diff_1,
        front_angle_diff_2,
        front_vel_mean_1,
        front_vel_mean_2,
        rear_angle_diff_1,
        rear_angle_diff_2,
        rear_vel_mean_1,
        rear_vel_mean_2
    ]

    return np.array(projected_obs)