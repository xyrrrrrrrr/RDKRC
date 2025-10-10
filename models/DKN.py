import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Tuple


class KoopmanOperator(nn.Module):
    # 参数化 Koopman 线性算子, 对嵌合状态进行训练
    def __init__(self, cat_dim: int, control_embed_dim: int, device: torch.device):
        super().__init__()
        # A: 状态转移矩阵（z_dim → z_dim），对应文档IV.B节“线性状态转移”
        self.A = nn.Linear(cat_dim, cat_dim, device=device, bias=False)
        # B: 控制增益矩阵（控制嵌入维度 → z_dim），对应文档IV.A节“控制嵌入到z空间”
        self.B = nn.Linear(control_embed_dim, cat_dim, device=device, bias=False)
        # 初始化：A用单位矩阵（初始近似恒等转移），B用小权重（初始控制影响弱）
        nn.init.eye_(self.A.weight)
        nn.init.normal_(self.B.weight, mean=0.0, std=0.01)

    def forward(self, z_prev: torch.Tensor, g_u: torch.Tensor) -> torch.Tensor:
        """
        嵌入空间线性动力学计算：z_next = A z_prev + B g_u
        input: z_prev (B, z_dim), g_u (B, control_embed_dim)
        output: z_next (B, z_dim)
        """
        return self.A(z_prev) + self.B(g_u)
    

class StateEmbedding(nn.Module):
    # 实现论文中的状态编码
    def __init__(self, x_dim: int, z_dim: int, hidden_dim:int, device: torch.device):
        super().__init__()
        self.Linear1 = nn.Linear(x_dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, z_dim)
        self.activation = nn.Tanh()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.cat_dim = x_dim + z_dim

    def forward(self, x):
        # result = torch.cat([x, u], dim=1)
        result = self.Linear1(x)
        result = self.activation(result)
        result = self.Linear2(result)
        if len(x.shape) > 3:
            result = torch.cat([x, result], dim=3)
        elif len(x.shape) > 2:
            result = torch.cat([x, result], dim=2)
        elif len(x.shape) > 1:
            result = torch.cat([x, result], dim=1)
        else:
            result = torch.cat([x, result], dim=0)
        return result


import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlEmbeddingVAE(nn.Module):
    """
    基于VAE（变分自编码器）结构实现论文中的控制编码功能
    核心改进：引入变分推断，编码器输出 latent 分布的均值和对数方差，通过重参数化采样获取 latent 变量，解码器重建输入的状态-控制对
    """
    def __init__(self, x_dim: int, control_dim: int, hidden_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        
        # 1. 确定VAE输入维度（是否拼接状态x和控制u）
        self.cat = x_dim != 0  # 当x_dim=0时，仅用控制u作为输入；否则拼接x和u
        self.input_dim = control_dim if not self.cat else (x_dim + control_dim)
        
        # 2. VAE编码器：输入（x+u 或 u）→ 输出 latent 分布的均值(mu)和对数方差(log_var)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),  # 共享特征提取层
            nn.Tanh(),
        )
        self.fc_mu = nn.Linear(hidden_dim, control_dim)  # 输出latent均值
        self.fc_log_var = nn.Linear(hidden_dim, control_dim)  # 输出latent对数方差（避免方差为负）
        
        # 3. VAE解码器：输入 latent 变量 → 重建原始输入（x+u 或 u）
        self.decoder = nn.Sequential(
            nn.Linear(control_dim, hidden_dim),  # latent特征映射层
            nn.Tanh(),
            nn.Linear(hidden_dim, self.input_dim)  # 输出维度=输入维度，用于重建
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        VAE核心重参数化技巧：从N(mu, std²)中采样latent变量，确保梯度可反向传播
        Args:
            mu: latent分布的均值，shape=(batch_size, latent_dim)
            log_var: latent分布的对数方差，shape=(batch_size, latent_dim)
        Returns:
            latent: 采样后的latent变量，shape=(batch_size, latent_dim)
        """
        std = torch.exp(0.5 * log_var)  # 标准差 = 对数方差的指数开平方
        eps = torch.randn_like(std, device=self.device)  # 从标准正态分布N(0,1)采样eps
        return mu + eps * std  # 重参数化：mu + eps*std ~ N(mu, std²)

    def encode(self, x: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        编码器前向传播：处理输入（x+u 或 u）→ 输出latent分布的mu和log_var
        Args:
            x: 系统状态，shape=(batch_size, x_dim)；若x_dim=0，可传入None
            u: 控制输入，shape=(batch_size, control_dim)
        Returns:
            mu: latent均值，shape=(batch_size, latent_dim)
            log_var: latent对数方差，shape=(batch_size, latent_dim)
        """
        # 拼接输入（若需要）
        if self.cat:
            input_data = torch.cat([x, u], dim=1)  # shape=(batch_size, x_dim+control_dim)
        else:
            input_data = u  # shape=(batch_size, control_dim)
        
        # 提取特征并输出分布参数
        feat = self.encoder(input_data)
        mu = self.fc_mu(feat)
        log_var = self.fc_log_var(feat)
        return mu, log_var

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        解码器前向传播：从latent变量重建原始输入（x+u 或 u）
        Args:
            latent: 采样后的latent变量，shape=(batch_size, latent_dim)
        Returns:
            recon_data: 重建的输入数据，shape=(batch_size, input_dim)
        """
        recon_data = self.decoder(latent)
        return recon_data

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAE完整前向传播：编码→重参数化→解码
        Args:
            x: 系统状态，shape=(batch_size, x_dim)；若x_dim=0，可传入None
            u: 控制输入，shape=(batch_size, control_dim)
        Returns:
            latent: 采样后的latent变量（控制编码结果），shape=(batch_size, latent_dim)
            recon_data: 重建的输入数据，shape=(batch_size, input_dim)
            mu: latent分布均值，shape=(batch_size, latent_dim)
            log_var: latent分布对数方差，shape=(batch_size, latent_dim)
        """
        # 1. 编码：获取latent分布参数
        mu, log_var = self.encode(x, u)
        # 2. 重参数化：采样latent变量
        latent = self.reparameterize(mu, log_var)
        # 3. 解码：重建原始输入
        recon_data = self.decode(latent)
        if self.cat:
            input_data = torch.cat([x, u], dim=1)  # shape=(batch_size, x_dim+control_dim)
        else:
            input_data = u  # shape=(batch_size, control_dim)
        loss = self.vae_loss(recon_data, input_data, mu, log_var )
        
        return latent, recon_data, mu, log_var, loss


    # 示例：VAE训练时的损失计算（需结合重建损失和KL散度）
    def vae_loss(self, recon_data: torch.Tensor, input_data: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        VAE损失函数：重建损失（MSE） + KL散度（正则化latent分布接近标准正态）
        Args:
            recon_data: 解码器输出的重建数据，shape=(batch_size, input_dim)
            input_data: 原始输入数据（x+u 或 u），shape=(batch_size, input_dim)
            mu: latent分布均值，shape=(batch_size, latent_dim)
            log_var: latent分布对数方差，shape=(batch_size, latent_dim)
        Returns:
            total_loss: VAE总损失（重建损失 + KL散度）
        """
        # 1. 重建损失：MSE（匹配原始输入和重建结果）
        recon_loss = F.mse_loss(recon_data, input_data, reduction="mean")
        # 2. KL散度：正则化latent分布接近N(0,1)（公式推导自变分推断）
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        # 3. 总损失（可通过超参数平衡两者权重，此处默认1:1）
        total_loss = recon_loss + kl_loss
        return total_loss



class ControlEmbedding(nn.Module):
    # 实现论文中的控制编码
    def __init__(self, x_dim: int, control_dim: int, hidden_dim:int, device: torch.device):
        super().__init__()
        if x_dim == 0:
            self.cat = False
            embed_dim = control_dim
        else:
            self.cat = True
            embed_dim = x_dim + control_dim
        self.Linear1 = nn.Linear(embed_dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, control_dim)
        # self.Linear3 = nn.Linear(embed_dim, hidden_dim)
        self.Linear3 = nn.Linear(control_dim, hidden_dim)
        self.Linear4 = nn.Linear(hidden_dim, control_dim)
        self.activation = nn.Tanh()

    def encoder(self, x, u):
        result = torch.cat([x, u], dim=1)
        result = self.Linear1(result)
        result = self.activation(result)
        return self.Linear2(result)
    
    def decoder(self, latent):
        result = self.Linear3(latent)
        result = self.activation(result)
        return self.Linear4(result)
    
    def forward(self, x, u):
        control_embed = self.encoder(x, u)
        control_decode = self.decoder(control_embed)

        return control_embed, control_decode, u  


class KStepsPredictor(nn.Module):
    def __init__(self, x_dim: int, control_dim: int, z_dim: int, hidden_dim: int, low: Union[List[float], np.ndarray], high: Union[List[float], np.ndarray], K_steps:int, device: torch.device):
        super().__init__()
        self.StateEmbedding = StateEmbedding(x_dim, z_dim, hidden_dim, device)
        self.ControlEmbedding = ControlEmbedding(x_dim, control_dim, hidden_dim, device)
        # self.ControlEmbedding = ControlEmbeddingVAE(x_dim, control_dim, hidden_dim, device)
        self.cat_dim = x_dim + z_dim
        self.KoopmanOperator = KoopmanOperator(self.cat_dim, control_dim, device)
        self.K_steps = K_steps
        self.device = device
        self.low = torch.tensor(low, device=device)
        self.high = torch.tensor(high, device=device)

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """状态归一化（保持原逻辑）"""
        x_clamped = torch.clamp(x, self.low, self.high)
        x_norm = (x_clamped - self.low) / (self.high - self.low + 1e-8)
        return x_norm

    def forward(self, x_init, u_series):
        """
        K步预测前向传播（🔶1-54至🔶1-61节逻辑）
        Args:
            x_init: 初始状态（形状：B × x_dim，B为批量大小）
            u_series: K步控制输入序列（形状：B × K_steps × control_dim）
        Returns:
            x_pred_series: 预测K步状态序列（形状：B × K_steps × x_dim）
        """
        z_pred_series = []  # 存储每步预测的原始状态，最终输出
        u_decode_series = []
        # 1. 初始状态嵌入：按🔶1-47节Equation 9，z = [原始状态x; 网络编码特征]
        z_prev = self.StateEmbedding(x_init)[:, 0, :]
        # 2. 递推执行K步预测（遵循🔶1-55节Equation 13的线性动力学）
        u_series = u_series.permute(1, 0, 2) 
        for step in range(self.K_steps):
            # 2.1 从当前嵌入向量z_prev提取原始状态x_prev（🔶1-48节Equation 10：x = C·z，C=[I_n, 0]）
            # 注：StateEmbedding输出的z前x_dim维为原始状态，需依赖其x_dim属性记录原始状态维度
            x_prev = z_prev[:, :self.StateEmbedding.x_dim]
            # 2.2 获取当前步的控制输入u_step（从K步控制序列中截取对应时间步）
            u_step = u_series[step, :, :]  # 维度：B × control_dim
            # 2.3 计算控制嵌入g_u(x_prev, u_step)（🔶1-51节DKAC逻辑，建模状态依赖的非线性控制项）
            g_u_step, u_step_decode, _ = self.ControlEmbedding(self.normalize_x(x_prev), u_step)  # 维度：B × control_dim
            # 2.4 用Koopman算子预测下一步嵌入向量z_next（🔶1-55节线性动力学：z_{t+1}=A·z_t + B·g_u）
            z_next = self.KoopmanOperator(z_prev, g_u_step)  # 维度：B × (x_dim+z_dim)（cat_dim）
            # 2.6 存储预测结果，并更新z_prev为下一步的输入
            z_pred_series.append(z_next)
            # u_decode_series.append(u_step_decode[:, self.StateEmbedding.x_dim:])
            u_decode_series.append(u_step_decode)
            z_prev = z_next  # 滚动更新：当前z_next作为下一步的z_prev
        # 3. 整理预测结果维度：从列表（K_steps × B × x_dim）转为张量（B × K_steps × x_dim）
        z_pred_series = torch.stack(z_pred_series, dim=1)
        u_decode_series = torch.stack(u_decode_series, dim=1)

        return z_pred_series, u_decode_series
    
    def decode_control(self, control_embed):
        return self.ControlEmbedding.decoder(control_embed)

