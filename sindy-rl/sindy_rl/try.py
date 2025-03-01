import numpy as np
from sindy_rl.registry import Go2Sim  # 根据实际路径调整

# 初始化环境（带渲染）
env = Go2Sim(render_mode="human")

# 获取动作空间参数
action_dim = env.action_space.shape[0]  # 通常为12（每条腿3个关节）
action_low = env.action_space.low[0]    # 假设动作空间对称
action_high = env.action_space.high[0]

# 重置环境
obs = env.reset()
done = False

# 安全运行参数配置
MAX_STEPS = 100000        # 最大运行步数
TILT_THRESHOLD = 0.8    # 基座倾斜角安全阈值（根据实际单位调整）
Z_HEIGHT_MIN = 0.2      # 最低允许高度（米）

try:
    step = 0
    while step < MAX_STEPS:
        # 生成随机动作（均匀分布）
        action = np.random.uniform(low=action_low, 
                                  high=action_high, 
                                  size=(action_dim,))
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 渲染更新（根据具体实现可能需要）
        if hasattr(env, 'render'):
            env.render()
        
        # 安全监测：基座高度和倾斜角
        base_height = obs[2]                  # 假设z轴位置在索引2
        base_orientation = obs[3:6]           # 假设欧拉角在3-5索引
        
        # 计算俯仰/横滚角幅度（示例方法）
        tilt_magnitude = np.linalg.norm(base_orientation[:2]) 
        
        # 检测危险状态
        if base_height < Z_HEIGHT_MIN or tilt_magnitude > TILT_THRESHOLD:
            print(f"安全终止！高度：{base_height:.2f}m，倾斜：{tilt_magnitude:.2f}rad")
            done = True
        
        # 打印状态信息
        print(f"Step {step}: Reward={reward:.2f} | "
              f"Height={base_height:.2f}m | "
              f"Tilt={tilt_magnitude:.2f}rad")
        
        step += 1

except KeyboardInterrupt:
    print("\n用户手动终止操作！")
finally:
    env.close()
    print("环境已安全关闭")

# 运行结果报告
if done:
    print(f"因安全限制终止于第{step}步")
else:
    print(f"正常完成{MAX_STEPS}步测试")