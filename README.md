# Sindy-go2 - Use Sindy-rl in go2 Robots

[![Python 3.10](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)


A PyTorch-based reinforcement learning framework for robotic control.

## 🚀 Installation

### Prerequisites
- Python 3.10
- Conda 
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

