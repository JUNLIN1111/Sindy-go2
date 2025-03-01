import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveEncoder(nn.Module):
    def __init__(self, input_dim=62, latent_dim=24):  # 默认维度调整为24


        super().__init__()
        self.latent_dim = latent_dim
        
        # 动态计算各传感器分组维度
        dynamics_dim = 3 + 4 + 12 + 12 + 3 + 3 + 4  # 41维
        secondary_dim = 3 + 12 + 3                  # 18维
        
        # 自适应编码结构
        self.dynamics_encoder = nn.Sequential(
            nn.Linear(dynamics_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        
        self.secondary_encoder = nn.Sequential(
            nn.Linear(secondary_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64)
        )

        # 融合层动态调整
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, latent_dim * 2)  # 输出mu和logvar
        )

    def encode(self, x):
        dynamics_part = x[:, :41]
        secondary_part = x[:, 41:]
        
        h1 = self.dynamics_encoder(dynamics_part)
        h2 = self.secondary_encoder(secondary_part)
        fused = torch.cat([h1, h2], dim=1)
        return self.fusion(fused).chunk(2, dim=-1)  # 分割为mu和logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

def create_dim_reducer(input_dim=62, latent_dim=24, device='cpu'):#我电脑没有英伟达GPU
    """创建并初始化降维器"""
    model = AdaptiveEncoder(input_dim, latent_dim).to(device)
    return model

def train_dim_reducer(model, train_data, val_data=None, 
                     epochs=100, lr=1e-4, weight_decay=1e-5):
    """训练降维模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 自动生成权重矩阵（示例比例）
    weights = torch.cat([
        torch.ones(41)*1.0,   # 动力学参数高权重
        torch.ones(21)*0.3    # 其他参数低权重
    ]).to(train_data.device)
    
    loss_fn = nn.ModuleDict({
        'recon': nn.MSELoss(reduction='none'),
        'kl': lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    })
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_data:
            mu, logvar = model.encode(batch)
            z = model.reparameterize(mu, logvar)
            
            # 简化解码过程（实际需替换为真实解码器）
            recon = z @ torch.randn(model.latent_dim, 62, device=z.device)  
            
            recon_loss = (loss_fn['recon'](recon, batch) * weights).mean()
            kl_loss = loss_fn['kl'](mu, logvar).mean()
            loss = recon_loss + kl_loss * 0.1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")
        
        # 验证集监控
        if val_data:
            val_loss = evaluate(model, val_data, loss_fn, weights)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f"best_dim_{model.latent_dim}d.pth")
    
    return model

def evaluate(model, data, loss_fn, weights):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data:
            mu, logvar = model.encode(batch)
            z = model.reparameterize(mu, logvar)
            recon = z @ torch.randn(model.latent_dim, 62, device=z.device)
            loss = (loss_fn['recon'](recon, batch) * weights).mean()
            total_loss += loss.item()
    return total_loss / len(data)

def reduce_dimensionality(model, obs, normalize=True):
    """执行降维操作"""
    if normalize:
        obs = (obs - obs.mean(axis=0)) / (obs.std(axis=0) + 1e-8)
        
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs).to(next(model.parameters()).device)
        return model(obs_tensor).cpu().numpy()


# 使用示例
if __name__ == "__main__":
    # 初始化降维器（可选维度：12/16/24/32）
    reducer = create_dim_reducer(latent_dim=24)
    
    # 假设已有训练数据（形状：[N, 62]）
    train_loader = torch.utils.data.DataLoader(torch.randn(1000, 62), batch_size=32)
    
    # 训练降维模型
    trained_model = train_dim_reducer(reducer, train_loader, epochs=50)
    
    # 使用降维
    sample_obs = torch.randn(62)
    latent = reduce_dimensionality(trained_model, sample_obs)
    print(f"降维结果：{latent.shape}")  # 输出：(24,)