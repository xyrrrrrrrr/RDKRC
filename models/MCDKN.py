import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Tuple

class DKN_MC(nn.Module):
    """带流形约束和逆映射的DKN网络"""
    def __init__(self, x_dim: int, u_dim: int, hidden_dim: int, manifold_dim: int, state_low:Union[List[float], np.ndarray], state_high:Union[List[float], np.ndarray], 
                 action_low:Union[List[float]], action_high:Union[List[float]], device: torch.device):
        super().__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.manifold_dim = manifold_dim  # 流形维度
        
        # 1. 嵌入网络（原DKN + 流形投影层）
        self.embed_theta = nn.Sequential(  # g_theta(x)
            nn.Linear(x_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.embed_dim = x_dim + hidden_dim
        # 流形投影层：z -> z·P（P: [embed_dim, manifold_dim]）
        self.manifold_proj = nn.Linear(self.embed_dim, manifold_dim, bias=False)
        # 恢复投影（用于后续LQR控制，确保原状态可恢复）
        self.manifold_recov = nn.Linear(manifold_dim, self.embed_dim, bias=False)
        self.state_low = torch.tensor(state_low, dtype=torch.float32, device=device)
        self.state_high = torch.tensor(state_high, dtype=torch.float32, device=device)
        self.action_low =  torch.tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)
        # 2. Koopman算子（A: 流形内线性演化，B: 控制嵌入到流形）
        self.A = nn.Linear(manifold_dim, manifold_dim, bias=False)  # A: 流形维度→流形维度
        self.B = nn.Linear(u_dim, manifold_dim, bias=False)        # B: 控制维度→流形维度
        
        # 3. 控制网络（原DKN + 逆映射分支）
        self.control_net = nn.Sequential(  # g_phi(x,u)
            nn.Linear(x_dim + u_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, u_dim)
        )
        self.inv_control_net = nn.Sequential(  # g_phi^{-1}(x, hat_u)
            nn.Linear(x_dim + u_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, u_dim)
        )
        
        # 4. 恢复矩阵C（文档Eq.11）
        self.C = torch.eye(x_dim, self.embed_dim, device=device)
    
    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """状态归一化（保持原逻辑）"""
        x_clamped = torch.clamp(x, self.state_low, self.state_high)
        x_norm = (x_clamped - self.state_low) / (self.state_high - self.state_low + 1e-8)
        return x_norm
    
    def normalize_u(self, u: torch.Tensor) -> torch.Tensor:
        """状态归一化（保持原逻辑）"""
        u_clamped = torch.clamp(u, self.action_low, self.action_high)
        u_norm = (u_clamped - self.action_low) / (self.action_high - self.action_low + 1e-8)
        return u_norm

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """改进的嵌入函数：z -> 流形投影z_M"""
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        g_theta = self.embed_theta(self.normalize_x(x))  # [batch, hidden_dim]
        z = torch.cat([x, g_theta], dim=1)  # [batch, embed_dim]（原DKN嵌入）
        z_M = self.manifold_proj(z)  # [batch, manifold_dim]（流形投影）
        return z_M

    def recover_z(self, z_M: torch.Tensor) -> torch.Tensor:
        """从流形投影恢复原嵌入z"""
        return 

    def recover_x(self, z: torch.Tensor) -> torch.Tensor:
        """文档Eq.10：从原嵌入z恢复状态x"""
        return z @ self.C.T  # [batch, x_dim]

    def forward_control(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """控制网络前向：g_phi(x,u)"""
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(u.shape) < 2:
            u = u.unsqueeze(0)
        x_u = torch.cat([self.normalize_x(x), self.normalize_u(u)], dim=1)  # [batch, x_dim+u_dim]
        return self.control_net(x_u)  # [batch, u_dim]

    def forward_inv_control(self, x: torch.Tensor, hat_u: torch.Tensor) -> torch.Tensor:
        """逆控制网络前向：g_phi^{-1}(x, hat_u)"""
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(hat_u.shape) < 2:
            hat_u = hat_u.unsqueeze(0)
        x_hat_u = torch.cat([self.normalize_x(x), self.normalize_u(hat_u)], dim=1)  # [batch, x_dim+u_dim]
        return self.inv_control_net(x_hat_u)  # [batch, u_dim]

    def forward_koopman(self, z_M_t: torch.Tensor, g_phi_t: torch.Tensor) -> torch.Tensor:
        """改进的Koopman线性演化（流形内）：z_M_{t+1} = A z_M_t + B g_phi_t"""
        return self.A(z_M_t) + self.B(g_phi_t)  # [batch, manifold_dim]

    def predict_k_steps(self, x0: torch.Tensor, u_seq: torch.Tensor, k: int) -> torch.Tensor:
        """改进的K步预测（基于流形嵌入）"""
        batch_size = x0.shape[0]
        x_seq = [x0]
        z_M_prev = self.embed(x0)  # 初始流形嵌入 [batch, manifold_dim]
        
        for t in range(k):
            u_t = u_seq[t].view(batch_size, self.u_dim)  # [batch, u_dim]
            # 1. 控制嵌入
            g_phi_t = self.forward_control(x_seq[-1], u_t)  # [batch, u_dim]
            # 2. 流形内线性演化
            z_M_next = self.forward_koopman(z_M_prev, g_phi_t)  # [batch, manifold_dim]
            # 3. 恢复原嵌入和状态
            z_next = self.recover_z(z_M_next)  # [batch, embed_dim]
            x_next = self.recover_x(z_next)  # [batch, x_dim]
            # 4. 迭代
            x_seq.append(x_next)
            z_M_prev = z_M_next
        
        return torch.stack(x_seq, dim=0)  # [k+1, batch, x_dim]
    
class DKN_MC2(nn.Module):
    """带流形约束和逆映射的DKN网络"""
    def __init__(self, x_dim: int, u_dim: int, hidden_dim: int, manifold_dim: int, control_manifold_dim:int, state_low:Union[List[float], np.ndarray], state_high:Union[List[float], np.ndarray], 
                 action_low:Union[List[float]], action_high:Union[List[float]], device: torch.device):
        super().__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.manifold_dim = manifold_dim  # 流形维度
        self.control_manifold_dim = control_manifold_dim  # 控制流形维度
        
        # 1. 嵌入网络（原DKN + 流形投影层）
        self.embed_theta = nn.Sequential(  # g_theta(x)
            nn.Linear(manifold_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim),
            nn.Tanh()
        )
        # self.embed_dim = x_dim + hidden_dim
        # 流形投影层：z -> z·P（P: [embed_dim, manifold_dim]）
        # self.manifold_proj = nn.Linear(self.embed_dim, manifold_dim, bias=False)
        # 恢复投影（用于后续LQR控制，确保原状态可恢复）
        self.manifold_recov = nn.Linear(manifold_dim, manifold_dim, bias=False)
        self.state_low = torch.tensor(state_low, dtype=torch.float32, device=device)
        self.state_high = torch.tensor(state_high, dtype=torch.float32, device=device)
        self.action_low =  torch.tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)
        # 2. Koopman算子（A: 流形内线性演化，B: 控制嵌入到流形）
        self.A = nn.Linear(manifold_dim, manifold_dim, bias=False)  # A: 流形维度→流形维度
        self.B = nn.Linear(control_manifold_dim, manifold_dim, bias=False)        # B: 控制维度→流形维度
        # 初始化
        nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.B.weight, a=np.sqrt(5))
        # 3. 控制网络（原DKN + 逆映射分支）
        self.control_net = nn.Sequential(  # g_phi(x,u)
            nn.Linear(x_dim + u_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, control_manifold_dim)
        )
        self.inv_control_net = nn.Sequential(  # g_phi^{-1}(x, hat_u)
            nn.Linear(control_manifold_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, x_dim + u_dim)
        )
        
        # 4. 恢复矩阵C（文档Eq.11）
        self.C = torch.eye(x_dim, hidden_dim, device=device)
    
    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """状态归一化（保持原逻辑）"""
        x_clamped = torch.clamp(x, self.state_low, self.state_high)
        x_norm = (x_clamped - self.state_low) / (self.state_high - self.state_low + 1e-8)
        return x_norm
    
    def normalize_u(self, u: torch.Tensor) -> torch.Tensor:
        """状态归一化（保持原逻辑）"""
        u_clamped = torch.clamp(u, self.action_low, self.action_high)
        u_norm = (u_clamped - self.action_low) / (self.action_high - self.action_low + 1e-8)
        return u_norm

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """改进的嵌入函数：z -> 流形投影z_M"""
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        # 补零到manifold_dim维度
        x = torch.cat([self.normalize_x(x), torch.zeros((x.shape[0], self.manifold_dim - self.x_dim), device=x.device)], dim=1)
        g_theta = self.embed_theta(x)  # [batch, hidden_dim]
        # z = torch.cat([x, g_theta], dim=1)  # [batch, embed_dim]（原DKN嵌入）
        # z_M = self.manifold_proj(z)  # [batch, manifold_dim]（流形投影）
        # return z_M
        return g_theta

    def recover_x(self, z: torch.Tensor) -> torch.Tensor:
        """文档Eq.10：从原嵌入z恢复状态x"""
        return self.manifold_recov(z)[:, :self.x_dim]  # [batch, x_dim]
    
    def forward_control(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """控制网络前向：g_phi(x,u)"""
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(u.shape) < 2:
            u = u.unsqueeze(0)
        x_u = torch.cat([self.normalize_x(x), self.normalize_u(u)], dim=1)  # [batch, x_dim+u_dim]
        return self.control_net(x_u)  # [batch, u_dim]

    def forward_koopman(self, z_M_t: torch.Tensor, g_phi_t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """改进的Koopman线性演化（流形内）：z_M_{t+1} = A z_M_t + B g_phi_t"""
        return self.A(z_M_t) + self.B(g_phi_t * u)  # [batch, manifold_dim]

    def predict_k_steps(self, x0: torch.Tensor, u_seq: torch.Tensor, k: int) -> torch.Tensor:
        """改进的K步预测（基于流形嵌入）"""
        batch_size = x0.shape[0]
        x_seq = [x0]
        z_prev = self.embed(x0)  # 初始流形嵌入 [batch, manifold_dim]
        
        for t in range(k):
            u_t = u_seq[t].view(batch_size, self.u_dim)  # [batch, u_dim]
            # 1. 控制嵌入
            g_phi_t = self.forward_control(x_seq[-1], u_t)  # [batch, u_dim]
            # 2. 流形内线性演化
            z_next = self.forward_koopman(z_prev, g_phi_t, u_t)  # [batch, manifold_dim]
            # 3. 恢复原嵌入和状态
            x_next = self.recover_x(z_next)  # [batch, x_dim]
            # 4. 迭代
            x_seq.append(x_next)
            z_prev = z_next
        
        return torch.stack(x_seq, dim=0)  # [k+1, batch, x_dim]