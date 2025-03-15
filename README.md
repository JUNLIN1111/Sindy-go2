# 🦿 SINDy-Go2: Sparse Dynamics-Based Control and Imitation Learning for Quadruped Robots

## 📝 Project Overview
This project leverages **SINDy (Sparse Identification of Nonlinear Dynamics)** combined with **Imitation Learning** to achieve efficient walking and jumping control for quadruped robots (e.g., Unitree Go2 and Boston Dynamics Spot). By extracting sparse dynamical models from expert demonstrations, we reduce the computational complexity of policy learning while maintaining model interpretability, accelerating deployment from simulation (MuJoCo) to real-world hardware.

---

## 🚀 Installation Guide

### Requirements
- **Python**: 3.10.x
- **Package Manager**: Conda (Miniconda/Anaconda)
- **Version Control**: Git

### Step-by-Step Installation
```bash
# 1. Create and activate Conda environment
conda create -n sindy_go2 python=3.10.16
conda activate sindy_go2

# 2. Clone repository
git clone https://github.com/JUNLIN1111/Sindy-go2.git
cd Sindy-go2/sindy-rl

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .  # Install in editable mode

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


