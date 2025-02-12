import numpy as np
import mujoco
import mujoco.viewer
from gymnasium import Env, spaces
from typing import Optional
from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv

class DMCEnvWrapper(DMCEnv):
    '''
    A wrapper for all dm-control environments using RLlib's 
    DMCEnv wrapper. 
    '''
    # need to wrap with config dict instead of just passing kwargs
    def __init__(self, config=None):
        env_config = config or {}
        super().__init__(**env_config)

class Go2Sim(Env):
    def __init__(
        self, render_mode: Optional[str] = None, record_path=None, useFixedBase=False
    ):  
        # 加载MuJoCo模型
        model_path = "D:\\EPS2_project\\move\\go2\\scene.xml"
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model from {model_path}: {e}")
        
        self.data = mujoco.MjData(self.model)
        '''
        # 初始化MuJoCo的渲染器
        self.viewer = None
        if render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        '''
        # 设置仿真参数
        self.model.opt.timestep = 1 / 240  # 与PyBullet时间步一致
        self.target = None #target 先初始化为0
        
        # 获取关节和执行器信息
        self.n_j = self.model.nu  # 执行器数量
        self.n_q = self.model.nq  # 广义坐标数量
        self.n_v = self.model.nv  # 广义速度数量
        
        # 打印自由度信息
        print(f"n_q: {self.n_q}, n_v: {self.n_v}")
        
        # 设置观察空间和动作空间
        # 观察空间：关节位置和速度
        obs_low = -np.inf * np.ones(62)  # 维度改为 62
        obs_high = np.inf * np.ones(62)  # 维度改为 
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        
        # 动作空间：执行器的控制范围
        if self.model.actuator_ctrlrange is not None:
            act_low = self.model.actuator_ctrlrange[:, 0]
            act_high = self.model.actuator_ctrlrange[:, 1]
        else:
            # 如果没有定义控制范围，假设为[-1, 1]
            act_low = -np.ones(self.n_j)
            act_high = np.ones(self.n_j)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # 重置仿真数据
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置初始关节位置和速度
        q_init = np.zeros(self.n_q)
        dq_init = np.zeros(self.n_v)
        self.data.qpos[:] = q_init
        self.data.qvel[:] = dq_init
        mujoco.mj_forward(self.model, self.data)
        
        # 获取初始观察值
        obs = self._get_obs()
        print("Initial observation:", obs)
        print("Observation space low:", self.observation_space.low)
        print("Observation space high:", self.observation_space.high)
        return obs, {}

    def step(self, action):
        # 应用控制指令
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        '''
        # 更新渲染（如果需要可视化）
        if self.viewer is not None:
            self.viewer.sync()
        '''
        # 返回观察值、奖励、终止标志、截断标志和信息
        obs = self._get_obs()
        print("Step observation:", obs)
        reward = self._get_reward()
        terminated = self._get_done()  # 任务是否因失败而终止
        truncated = False  # 任务是否因时间限制而终止
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # 读取 MuJoCo 的状态
        q = self.data.qpos.copy().astype(np.float32)  # 关节角度 + 可能的基座状态
        dq = self.data.qvel.copy().astype(np.float32)  # 关节速度 + 可能的基座速度
        
        # 处理基座状态（如果基座可以移动）
        base_position = q[:3]  # 基座位置 (x, y, z)
        base_orientation = q[3:7]  # 基座姿态（四元数）

        q_joints = q[7:]  # 关节角度

        base_linear_velocity = dq[:3]  # 基座线速度
        base_angular_velocity = dq[3:6]  # 基座角速度
        dq_joints = dq[6:]  # 关节角速度

        # 额外环境信息
        foot_contact = self.get_foot_contacts()  # 获取足端接触信息，返回形状 (4,)
        external_forces = self.get_external_forces()  # 获取外部力信息，形状 (3,)
        joint_torques = self.data.qfrc_actuator.copy().astype(np.float32)  # 关节执行器力矩
        
        # 目标信息（如果有目标）
        if self.target is not None:
            goal_info = self.get_goal_info()  # 目标位置或目标状态
        else:
            goal_info = np.array([])  # 没有目标时为空
        
        # 组合所有状态
        obs = np.concatenate([
            base_position, base_orientation,  # 机器人基座信息
            q_joints, dq_joints,  # 机器人关节状态
            base_linear_velocity, base_angular_velocity,  # 机器人运动状态
            foot_contact, external_forces, joint_torques,  # 触地信息 & 外部力 & 关节力矩
            goal_info  # 目标信息
        ])
        print("base_orientation:",base_orientation)

        return obs



    def get_foot_contacts(self):
        """获取足端接触信息，返回形状 (4,)，表示四个足端是否接触地面"""
        contact_flags = np.zeros(4, dtype=np.float32)  # 4 个足端

        # 获取几何体名称列表
        geom_names = [self.model.geom(i).name for i in range(self.model.ngeom)]

        # 使用 self.data.ncon 获取当前时间步的实际接触点数
        for contact in self.data.contact[: self.data.ncon]:
            # 获取几何体 ID 对应的名称
            geom1_name = geom_names[contact.geom1]
            geom2_name = geom_names[contact.geom2]

            # 检查是否是足端接触地面（假设足端几何体命名为 foot_0, foot_1, foot_2, foot_3）
            for i in range(4):
                if f"foot_{i}" in (geom1_name, geom2_name):  
                    contact_flags[i] = 1.0  # 设为 1，表示接触地面

        return contact_flags  # 返回四个足端的接触状态（0 或 1）

    def get_external_forces(self):
        """获取基座受到的外部力，返回形状 (3,)，表示 x, y, z 方向的外力"""
        base_force = self.data.cfrc_ext[0, :3]  # 机器人基座的外力（前三个分量）
        return base_force.astype(np.float32)

    def _get_reward(self):
        # 定义奖励函数（根据任务需求）
        return 0.0

    def _get_done(self):
        # 定义终止条件（根据任务需求）
        return False

    def _get_info(self):
        # 返回额外的信息（可选）
        return {}
    '''
    def close(self):
        # 关闭渲染器
        if self.viewer is not None:
            self.viewer.close()
    '''


class Go2wrapper(Go2Sim):
    def __init__(self, config=None):
        super().__init__(**(config or {}))  # 如果 config=None，则用空字典 {}
