import torch
import torch.nn.functional as F
from typing import Tuple
from dkrc.utils.matrix_utils import compute_controllability_matrix


def compute_L1_loss(
    z_prev: torch.Tensor,
    z_next: torch.Tensor
) -> torch.Tensor:
    """
    计算L1损失:保证Koopman线性化精度(原文Equation 7中L1部分)
    L1 = (1/(L-1)) * Σ||z(x_{t+1}) - K·z(x_t)||,其中K = z(x_{t+1})·z(x_t)^†(伪逆)
    
    Args:
        z_prev (torch.Tensor): t时刻线性化状态z(x_t),形状[batch_size, N]
        z_next (torch.Tensor): t+1时刻线性化状态z(x_{t+1}),形状[batch_size, N]
    
    Returns:
        torch.Tensor: L1损失值(标量)
    """
    # 计算K = z_next @ z_prev.T @ (z_prev @ z_prev.T)^†(Koopman算子近似)
    z_prev_T = z_prev.T  # [N, batch_size]
    gram_matrix = z_prev @ z_prev_T  # [N, N]
    gram_matrix_pinv = torch.pinverse(gram_matrix)  # [N, N],伪逆
    K = z_next @ z_prev_T @ gram_matrix_pinv  # [batch_size, N] → 实际应为[N, N],需调整维度
    
    # 修正K的维度:批量平均后得到[N, N]的K(原文中K是全局算子,非批量依赖)
    K = K.mean(dim=0).reshape(z_prev.shape[1], z_prev.shape[1])  # [N, N]
    
    # 计算线性化误差:z_next - K·z_prev
    linear_error = z_next - z_prev @ K.T  # [batch_size, N]
    # 计算F范数(批量内平均)
    L1 = torch.norm(linear_error, p='fro', dim=1).mean()
    
    return L1


def compute_L2_loss(
    A: torch.Tensor,
    B: torch.Tensor,
    lambda_rank: float = 1.0,
    lambda_A: float = 0.1,
    lambda_B: float = 0.1
) -> torch.Tensor:
    """
    计算L2损失:保证系统能控性与矩阵稀疏性(原文Algorithm 1中L2定义)
    L2 = λ_rank·(N - rank(Cont(A,B))) + λ_A·||A||₁ + λ_B·||B||₁
    其中Cont(A,B)为能控性矩阵,N为基函数维度
    
    Args:
        A (torch.Tensor): Koopman线性化矩阵,形状[N, N]
        B (torch.Tensor): 控制输入矩阵,形状[N, m](m为控制维度)
        lambda_rank (float): 能控性秩惩罚权重(默认1.0)
        lambda_A (float): A矩阵1范数惩罚权重(默认0.1)
        lambda_B (float): B矩阵1范数惩罚权重(默认0.1)
    
    Returns:
        torch.Tensor: L2损失值(标量)
    """
    N = A.shape[0]
    # 1. 计算能控性矩阵及其秩
    controllability_mat = compute_controllability_matrix(A, B)  # [N, N×m]
    # 用SVD求秩(奇异值大于1e-6的数量)
    _, singular_vals, _ = torch.svd(controllability_mat)
    rank = (singular_vals > 1e-6).sum().item()
    rank_penalty = lambda_rank * (N - rank)
    
    # 2. 计算A、B矩阵的1范数(稀疏性惩罚)
    A_l1 = lambda_A * torch.norm(A, p=1)
    B_l1 = lambda_B * torch.norm(B, p=1)
    
    # 总L2损失
    L2 = rank_penalty + A_l1 + B_l1
    return L2


def compute_total_loss(
    z_prev: torch.Tensor,
    z_next: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    lambda_L1: float = 1.0,
    lambda_L2: float = 1.0,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算DKRC总损失:L = λ_L1·L1 + λ_L2·L2
    
    Args:
        z_prev (torch.Tensor): t时刻z(x_t),[batch_size, N]
        z_next (torch.Tensor): t+1时刻z(x_{t+1}),[batch_size, N]
        A (torch.Tensor): Koopman矩阵A,[N, N]
        B (torch.Tensor): 控制矩阵B,[N, m]
        lambda_L1 (float): L1损失权重(默认1.0)
        lambda_L2 (float): L2损失权重(默认1.0)
        **kwargs: 传递给compute_L2_loss的额外参数(如lambda_rank)
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            total_loss: 总损失,
            L1: L1损失,
            L2: L2损失
    """
    L1 = compute_L1_loss(z_prev, z_next)
    L2 = compute_L2_loss(A, B, **kwargs)
    total_loss = lambda_L1 * L1 + lambda_L2 * L2
    return total_loss, L1, L2