import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Optional
import numpy as np
from sklearn.manifold import Isomap  # 用于近似原空间测地线距离（🔶2-77节）

# 1. 保留原FlowVectorField，但修正输入维度对齐论文微分同胚定义（🔶2-49节）
class FlowVectorField(nn.Module):
    """PFM核心组件：Neural ODE向量场（适配论文M=M_d'×R^(d-d')结构，🔶2-49、🔶2-35节）"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1时间嵌入（论文默认操作，🔶2-50节）
            nn.SiLU(),  # 论文用Swish/SiLU激活，🔶2-50节
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_expanded = t.expand(x.shape[0], 1)
        tx = torch.cat([t_expanded, x], dim=1)
        return self.net(tx)

# 2. 保留ode_solve，确保RK4求解器对齐论文（🔶2-52节用Runge-Kutta）
def ode_solve(func: nn.Module, x0: torch.Tensor, t_span: torch.Tensor, 
              method: str = "rk4", step_size: float = 0.01) -> torch.Tensor:
    device = x0.device
    solutions = [x0]
    x = x0.clone()
    for i in range(len(t_span) - 1):
        t0, t1 = t_span[i], t_span[i + 1]
        dt = t1 - t0
        if method == "rk4":
            k1 = func(t0, x)
            k2 = func(t0 + dt/2, x + dt/2 * k1)
            k3 = func(t0 + dt/2, x + dt/2 * k2)
            k4 = func(t1, x + dt * k3)
            x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        else:
            x = x + dt * func(t0, x)
    return torch.stack(solutions, dim=0)

class PFM_DKN(nn.Module):
    """融合PFM思想的改进DKN网络（全量对齐论文，🔶2-22、🔶2-38、🔶2-45节）"""
    def __init__(self, x_dim: int, u_dim: int, hidden_dim: int, manifold_dim: int, 
                 latent_dim: int,  # 潜流形维度=M_d'维度+冗余维度（🔶2-35节M=M_d'×R^(d-d')）
                 state_low: Union[List[float], np.ndarray], 
                 state_high: Union[List[float], np.ndarray], 
                 action_low: Union[List[float]], 
                 action_high: Union[List[float]], 
                 dij,
                 device: torch.device):
        super().__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.manifold_dim = manifold_dim  # M_d'维度（论文默认d'=1，🔶2-74节）
        self.latent_dim = latent_dim      # 潜流形总维度（=manifold_dim + 冗余维度）
        self.hidden_dim = hidden_dim
        self.device = device
        
        # 状态/动作范围（对齐论文物理约束，🔶2-82节）
        self.state_low = torch.tensor(state_low, dtype=torch.float32, device=device)
        self.state_high = torch.tensor(state_high, dtype=torch.float32, device=device)
        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)
        
        # 3. 修正PFM流形映射：对齐论文微分同胚结构φ=[ψ⁻¹,I]∘φ∘T_μ（🔶2-48节）
        # 3.1 状态→低维流形M_d'（ψ⁻¹对应部分）
        self.state_to_manifold = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim)  # 输出M_d'维度
        )
        # 3.2 流形→潜流形M（M_d'×R^(d-d')，I对应冗余维度恒等映射）
        self.manifold_to_latent = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)  # 潜流形总维度（含冗余）
        )
        # 3.3 潜流形→流形（φ逆映射，🔶2-48节）
        self.latent_to_manifold = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim)
        )
        # 3.4 PFM核心：Neural ODE参数化微分同胚φ（🔶2-49节）
        self.flow_encoder = FlowVectorField(latent_dim, hidden_dim, latent_dim)  # M_d'→M
        self.flow_decoder = FlowVectorField(latent_dim, hidden_dim, latent_dim)  # M→M_d'
        
        # 4. 高维Koopman算子（对齐论文🔶2-40节，A/B作用于潜流形）
        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(u_dim, latent_dim, bias=False)
        
        # 5. 控制网络（保留原结构，对齐论文🔶2-21节控制嵌入）
        self.control_net = nn.Sequential(
            nn.Linear(x_dim + u_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, u_dim)
        )
        self.inv_control_net = nn.Sequential(
            nn.Linear(x_dim + u_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, u_dim)
        )
        
        # 6. 状态恢复（对齐论文🔶2-35节，从M_d'→原状态）
        self.manifold_to_state = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, x_dim)
        )

        self.dij = dij
        self.dij_max = dij.max()
        self.dij_diff_max =  (dij - dij.T).abs().max()

        # 7. 拉回度量参数（对齐论文🔶2-33节，基于潜流形度量）
        self.metric_scale = nn.Parameter(torch.ones(1, device=device))
        self.t_span = torch.linspace(0, 1, 5, device=device)  # 论文默认5个时间步，🔶2-52节
        
        # 8. 论文实验超参（🔶2-246节Table 5）
        self.alpha1 = 1.0    # 全局等距损失权重
        self.alpha3 = 1.0    # 子流形损失权重
        self.alpha4 = 0.001  # 稳定性正则化权重
        self.n_neighbors = 10 # Isomap近邻数（🔶2-74节）

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """状态归一化（保留原逻辑，对齐论文数据预处理，🔶2-62节）"""
        x_clamped = torch.clamp(x, self.state_low, self.state_high)
        return (x_clamped - self.state_low) / (self.state_high - self.state_low + 1e-8)
    
    def normalize_u(self, u: torch.Tensor) -> torch.Tensor:
        """控制归一化（同上）"""
        u_clamped = torch.clamp(u, self.action_low, self.action_high)
        return (u_clamped - self.action_low) / (self.action_high - self.action_low + 1e-8)
    
    # 9. 修正embed_to_latent：对齐论文微分同胚流程（🔶2-48、🔶2-49节）
    def embed_to_latent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """M_d'→M：低维流形→潜流形（含Neural ODE流）"""
        z_manifold = self.state_to_manifold(self.normalize_x(x))  # [B, d']
        redundant_zero = torch.zeros(
        size=(z_manifold.shape[0], self.latent_dim - self.manifold_dim),  # [B, d-d']
        dtype=torch.float32,
        device=self.device
        )
        # 拼接低维子流形与冗余维度，得到高维潜流形初始状态（[B, d]）
        z_manifold = torch.cat([z_manifold, redundant_zero], dim=1)  # [B, d' + (d-d')] = [B, d]
        # Neural ODE生成微分同胚（🔶2-49节：φ通过Neural ODE参数化）
        z_latent = ode_solve(
            self.flow_encoder, 
            z_manifold, 
            self.t_span
        )[-1]  # [B, latent_dim]（M_d'×R^(d-d')）
        return z_latent, z_manifold
    
    # 10. 修正recover_from_latent：对齐论文逆映射（🔶2-48节φ⁻¹）
    def recover_from_latent(self, z_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """M→M_d'→原状态：潜流形→低维流形→状态"""
        # 反向Neural ODE（🔶2-51节：逆映射需反向时间积分）
        z_manifold = ode_solve(
            self.flow_decoder, 
            z_latent, 
            self.t_span.flip(0)
        )[-1]  # [B, d]
        z_manifold = z_manifold[:, :self.manifold_dim]
        z_manifold_ = z_manifold[:, self.manifold_dim:]  # 提取M_d'部分
        x_recon = self.manifold_to_state(z_manifold)  # [B, x_dim]
        return x_recon, z_manifold_
    
    # 11. 修正pullback_metric：对齐论文拉回度量定义（🔶2-33节公式(21)）
    def pullback_metric(self, z_manifold: torch.Tensor) -> torch.Tensor:
        """基于潜流形M的拉回度量：(Ξ,Φ)^φ = (φ_*[Ξ],φ_*[Φ])^M"""
        B = z_manifold.shape[0]
        # 潜流形M的度量（论文默认M_d'为欧氏，冗余维度为欧氏，🔶2-219节）
        metric_M = torch.eye(self.latent_dim, device=self.device).unsqueeze(0).repeat(B, 1, 1)  # [B, L, L]
        # 计算pushforward φ_*（简化：用Neural ODE雅可比近似，🔶2-215节）
        J = torch.autograd.functional.jacobian(
            lambda x: self.flow_encoder(t=torch.tensor(0.5, device=self.device), x=x),
            z_manifold
        ).squeeze(1)  # [B, L, d']
        # 拉回度量：J^T · metric_M · J（🔶2-33节公式）
        pullback_metric = torch.matmul(torch.matmul(J.transpose(1,2), metric_M), J)  # [B, d', d']
        return pullback_metric * self.metric_scale
    
    # 12. 重写isometry_loss：融合论文3大等距约束（🔶2-59节公式）
    def isometry_loss(self, x: torch.Tensor, z_latent: torch.Tensor) -> torch.Tensor:
        """
        总等距损失 = 全局等距损失 + 子流形损失 + 稳定性正则化
        文档依据：🔶2-59（损失公式）、🔶2-68（ε_iso/ε_ld指标）、🔶2-77（Isomap）
        """
        B = x.shape[0]
        z_manifold1 = self.latent_to_manifold(z_latent[:, :self.latent_dim])  # [B, d']
        z_manifold2 = self.latent_to_manifold(z_latent[:, self.latent_dim:])  # [B, d']
        # 12.1 全局等距损失（🔶2-59节第一项）：d_D(xi,xj) ≈ d_M(φ(xi),φ(xj))
        # 原空间真实距离d_ij：Isomap近似（🔶2-77节标准做法）
        x_norm1 = self.normalize_x(x[:, :self.x_dim])
        x_norm2 = self.normalize_x(x[:, self.x_dim:])
        dij
        
        # 潜空间距离d_M：对齐论文公式(2)（🔶2-33节）
        d_M = torch.cdist(z_latent, z_latent)  # [B, B]
        global_loss = self.alpha1 * torch.mean(torch.square(d_ij - d_M))  # 平方损失（论文定义）
        
        # 12.2 子流形损失（🔶2-59节第三项）：强制潜流形映射到M_d'（冗余维度→0）
        mask = torch.zeros_like(z_latent, device=self.device)
        mask[:, :self.manifold_dim] = 1.0  # 保留M_d'维度，掩码冗余维度
        redundant_dim = z_latent * (1 - mask)  # [B, L-d']
        submanifold_loss = self.alpha3 * torch.mean(torch.norm(redundant_dim, p=1, dim=1))
        
        # 12.3 稳定性正则化（🔶2-59节第四项）：局部等距约束
        stability_loss = torch.tensor(0.0, device=self.device)
        if B > 1:
            # 局部近邻距离一致性（简化雅可比正则化，🔶2-60节思想）
            z_dist = torch.cdist(z_manifold1, z_manifold2)
            # z_dist = z_dist.fill_diagonal_(float('inf'))
            _, nn_idx = torch.min(z_dist, dim=1)  # 每个样本的1近邻
            x_nn_dist = d_ij[torch.arange(B), nn_idx]  # 原空间近邻距离
            z_nn_dist = z_dist[torch.arange(B), nn_idx]  # 潜空间近邻距离
            stability_loss = self.alpha4 * torch.mean(torch.abs(x_nn_dist - z_nn_dist))
        
        # 总等距损失
        total_iso_loss = global_loss + submanifold_loss + stability_loss
        # 输出论文指标（🔶2-68节）
        with torch.no_grad():
            eps_iso = torch.mean(torch.abs(d_ij - d_M)).item()
            eps_ld = torch.mean(torch.norm(redundant_dim, p=1, dim=1)).item()
            print(f"ε_iso: {eps_iso:.6f} | ε_ld: {eps_ld:.6f}")
        return total_iso_loss
    
    # 13. 修正flow_matching_loss：对齐论文PFM目标函数（🔶2-40节公式(6)）
    def flow_matching_loss(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """PFM流匹配损失：匹配潜流形测地线的时间导数"""
        z0, _ = self.embed_to_latent(x0)  # [B, L]
        z1, _ = self.embed_to_latent(x1)  # [B, L]
        
        # 论文🔶2-40节：时间调度器κ(t)（单调递减，κ(0)=1, κ(1)=0）
        t = torch.rand(z0.shape[0], 1, device=self.device)  # [B, 1]
        kappa_t = 1 - t  # 简化调度器（论文常用形式）
        
        # 潜流形测地线（论文公式(3)：γ^φ = φ⁻¹(γ^M)，此处M为欧氏，测地线为线性插值）
        z_t = torch.lerp(z0, z1, kappa_t)  # [B, L]
        
        # 目标向量场：测地线时间导数（🔶2-40节：ẋ_t = κ’(t)·log_x1(x0)，简化为z1-z0）
        kappa_prime_t = -1.0  # κ(t)=1-t的导数
        u_t = kappa_prime_t * (z0 - z1)  # [B, L]
        
        # 模型预测向量场（论文用神经网络参数化v_t，非直接用A(z_t)）
        # 修正：新增向量场网络（原模型用A(z_t)错误，🔶2-40节要求独立v_t）
        if not hasattr(self, 'flow_vector_net'):
            self.flow_vector_net = nn.Sequential(
                nn.Linear(self.latent_dim + 1, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.latent_dim)
            ).to(self.device)
        # 时间嵌入（🔶2-50节：向量场需输入t）
        t_expanded = t.expand_as(z_t[:, :1])
        vt_input = torch.cat([z_t, t_expanded], dim=1)  # [B, L+1]
        v_t = self.flow_vector_net(vt_input)  # [B, L]
        
        # 流匹配损失（论文公式(6)：L2损失）
        return torch.mean(torch.norm(v_t - u_t, dim=1))
    
    # 14. 保留控制网络（无修改，对齐论文控制嵌入逻辑🔶2-21节）
    def forward_control(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(u.shape) < 2:
            u = u.unsqueeze(0)
        x_u = torch.cat([self.normalize_x(x), self.normalize_u(u)], dim=1)
        return self.control_net(x_u)
    
    def forward_inv_control(self, x: torch.Tensor, hat_u: torch.Tensor) -> torch.Tensor:
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(hat_u.shape) < 2:
            hat_u = hat_u.unsqueeze(0)
        x_hat_u = torch.cat([self.normalize_x(x), self.normalize_u(hat_u)], dim=1)
        return self.inv_control_net(x_hat_u)
    
    # 15. 保留Koopman演化（无修改，对齐论文🔶2-40节）
    def forward_koopman(self, z_latent: torch.Tensor, g_phi: torch.Tensor) -> torch.Tensor:
        return self.A(z_latent) + self.B(g_phi)
    
    # 16. 保留多步预测（无修改，对齐论文🔶2-85节生成逻辑）
    def predict_k_steps(self, x0: torch.Tensor, u_seq: torch.Tensor, k: int) -> torch.Tensor:
        batch_size = x0.shape[0]
        x_seq = [x0]
        z_latent_prev, _ = self.embed_to_latent(x0)
        z_manifold_sum = 0.0  # 用于监控冗余维度
        for t in range(k):
            u_t = u_seq[t].view(batch_size, self.u_dim)
            g_phi_t = self.forward_control(x_seq[-1], u_t)
            z_latent_next = self.forward_koopman(z_latent_prev, g_phi_t)
            x_next, tool = self.recover_from_latent(z_latent_next)
            x_seq.append(x_next)
            z_manifold_sum += torch.mean(torch.abs(tool)).item()
            z_latent_prev = z_latent_next
        return torch.stack(x_seq, dim=0), z_manifold_sum

# 17. 修正compute_pfm_total_loss：对齐论文损失权重（🔶2-246节）
def compute_pfm_total_loss(
    model: PFM_DKN,
    x_prev: torch.Tensor,
    x_next: torch.Tensor,
    u_prev: torch.Tensor,
    u0: torch.Tensor,
    lambda_L1: float = 1.0,    # 论文默认1.0
    lambda_L2: float = 1.0,    # 论文默认1.0
    lambda_isometry: float = 1.0,  # 对齐α1=1.0
    lambda_flow: float = 0.5,  # 流匹配损失权重
    **kwargs
) -> Tuple[torch.Tensor, dict]:
    """融合论文PFM总损失（🔶2-59、🔶2-40节）"""
    # 潜空间表示（修正原模型xz_prev/xz_next错误，应为z_prev/z_next）
    z_prev, _ = model.embed_to_latent(x_prev)
    z_next, _ = model.embed_to_latent(x_next)
    
    # 17.1 L1损失（高维Koopman预测误差，保留原逻辑）
    v_prev = u_prev - u0
    z_next_pred = torch.matmul(z_prev, model.A.weight.T) + torch.matmul(v_prev, model.B.weight.T)
    L1 = torch.mean(torch.norm(z_next - z_next_pred, p='fro', dim=1))
    
    # 17.2 L2损失（能控性+正则化，保留原逻辑，🔶2-59节第二项）
    from rdkrc.utils.matrix_utils import compute_controllability_matrix
    controllability_mat = compute_controllability_matrix(model.A.weight, model.B.weight)
    _, singular_vals, _ = torch.svd(controllability_mat)
    rank = (singular_vals > 1e-5).sum().item()
    N = model.A.weight.shape[0]
    rank_penalty = (N - rank) / N
    A_l1 = torch.norm(model.A.weight, p=1)
    B_l1 = torch.norm(model.B.weight, p=1)
    L2 = rank_penalty + A_l1 + B_l1
    
    # 17.3 等距损失（调用修正后的函数，融合3大约束）
    isometry_loss = model.isometry_loss(torch.cat([x_prev, x_next], dim=1), 
                                       torch.cat([z_prev, z_next], dim=1))
    
    # 17.4 流匹配损失（调用修正后的函数，对齐论文公式）
    flow_loss = model.flow_matching_loss(x_prev, x_next)
    
    # 17.5 重构损失（保留原逻辑，对齐论文🔶2-82节状态恢复）
    x_prev_recon, _ = model.recover_from_latent(z_prev)
    x_next_recon, _ = model.recover_from_latent(z_next)
    recon_loss = 0.5 * (torch.mean(torch.norm(x_prev - x_prev_recon, dim=1)) +
                       torch.mean(torch.norm(x_next - x_next_recon, dim=1)))
    
    # 17.6 总损失（权重对齐论文实验配置）
    total_loss = (lambda_L1 * L1 +
                 lambda_L2 * L2 +
                 lambda_isometry * isometry_loss +
                 lambda_flow * flow_loss +
                 0.5 * recon_loss)
    
    # 损失监控（保留原逻辑）
    loss_components = {
        'total': total_loss.item(),
        'L1': L1.item(),
        'L2': L2.item(),
        'isometry': isometry_loss.item(),
        'flow': flow_loss.item(),
        'recon': recon_loss.item()
    }
    return total_loss, loss_components