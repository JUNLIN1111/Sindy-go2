import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# 1. 定义 VAE 的编码器（只用编码器）
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        
        # 编码器
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 均值
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 方差
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)  # 返回均值和方差
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z  # 只返回潜在空间表示


# 2. 创建 VAE 编码器
encoder = Encoder(input_dim=62, hidden_dim=256, latent_dim=16)

# 3. 使用编码器进行降维
def get_latent_state(obs):
    # 假设 obs 是62维的 numpy 数组
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    latent_state = encoder(obs_tensor)  # 获得低维潜在表示
    return latent_state.detach().numpy()





