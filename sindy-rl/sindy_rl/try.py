

import mujoco
import mujoco.viewer
import numpy as np

# 加载模型
XML_PATH = "D:\\EPS2_project\\go\\sindy-rl\\unitree_robots\\go2\\scene.xml"
mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
mj_data = mujoco.MjData(mj_model)

# 时间变量
t = 0
dt = 0.01  # 仿真步长

# 启动可视化
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        # 让机器人关节按照正弦波运动
        mj_data.ctrl[:] = np.sin(t)

        # 进行仿真步进
        mujoco.mj_step(mj_model, mj_data)

        # 更新时间
        t += dt

        # 同步渲染
        viewer.sync()
