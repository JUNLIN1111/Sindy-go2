import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from sindy_rl.policy import RandomPolicy
from sindy_rl.registry import Go2Sim  # 确保 Go2Sim 可以正确导入

import numpy as np
import pickle
import mujoco
import mujoco.viewer

import numpy as np
import pickle
import mujoco
from gym import spaces
from typing import Optional
from sindy_rl.registry import Go2Sim  # 确保这个路径是正确的


# 初始化环境
env = Go2Sim(render_mode="human")
env.reset()

# 加载训练好的策略
params_path = "D:\\EPS2_project\\go\\go2_sparse_policy.pkl"
try:
    with open(params_path, "rb") as f:
        sparse_policy = pickle.load(f)
except FileNotFoundError:
    print(f"Error: File {params_path} not found.")
    exit()
except EOFError:
    print("Error: params.pkl file is corrupted.")
    exit()

# 选择策略

SparseEnsemblePolicy = sparse_policy

# 创建 MuJoCo Viewer
done = False
obs = env.reset()  # 这是一个字典
obs = obs[0]
while not done:
    # 从策略中获取动作
    action = SparseEnsemblePolicy.compute_action(obs)
    
    # 执行动作并获取新的状态和奖励
    next_state, reward, done, truncated, info= env.step(action)
    
    # 渲染环境
    env.render(mode="human")  # 渲染到 MuJoCo viewer
    
    # 更新环境状态
    
    obs = next_state
    print("reward:",reward)

# 关闭 Viewer
env.viewer.close()


        