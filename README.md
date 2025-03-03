# Sindy-go2 - Reinforcement Learning Framework

[![Python 3.10](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conda](https://img.shields.io/conda/v/conda-forge/conda)](https://docs.conda.io/en/latest/)

A PyTorch-based reinforcement learning framework for robotic control.

## 🚀 Installation

### Prerequisites
- Python 3.10.x
- Conda (Miniconda/Anaconda)
- Git

### Step-by-Step Setup

1. **Create Conda Environment**
```bash
conda create -n sindy_go2 python=3.10.16
conda activate sindy_go2
```

2. **Clone Repository**
```bash
git clone https://github.com/JUNLIN1111/Sindy-go2.git
cd Sindy-go2/sindy-rl
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
pip install -e .  # Install in editable mode
```

### Verify Installation
```bash
python -c "import sindy_rl; print(sindy_rl.__version__)"
```

## 💻 Basic Usage
```python
from sindy_rl import make_env

env = make_env('HalfCheetah-v4')
obs = env.reset()
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
```

## 📚 Documentation
See [docs/](docs/) directory for API reference and tutorials.

---

# 中文版

## 🚀 安装指南

### 环境要求
- Python 3.10.x
- Conda (Miniconda/Anaconda)
- Git

### 逐步安装

1. **创建Conda环境**
```bash
conda create -n sindy_go2 python=3.10.16
conda activate sindy_go2
```

2. **克隆代码仓库**
```bash
git clone https://github.com/JUNLIN1111/Sindy-go2.git
cd Sindy-go2/sindy-rl
```

3. **安装依赖项**
```bash
pip install -r requirements.txt
pip install -e .  # 可编辑模式安装
```

### 验证安装
```bash
python -c "import sindy_rl; print(sindy_rl.__version__)"
```

## 💻 基础使用
```python
from sindy_rl import make_env

env = make_env('HalfCheetah-v4')
obs = env.reset()
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
```

## 📚 文档
API参考和教程请见 [docs/](docs/) 目录
