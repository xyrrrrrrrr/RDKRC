import torch
import torch.nn.functional as F
from typing import Tuple
from rdkrc.utils.matrix_utils import compute_controllability_matrix


def compute_L1_loss(
    z_prev: torch.Tensor,
    z_next: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    u_prev: torch.Tensor,
    u0: torch.Tensor
) -> torch.Tensor:
    """
    计算L1损失（文档Algorithm 1步骤2 + Equation 7）：
    确保高维线性系统的预测误差最小化，核心公式为：
    L1(θ) = (1/L) · Σ||z_{t+1} - A z_t - B(u_t - u0)||_F 
    （L为批量大小，对应文档“t=0到L-1求和”，L = 批量样本数）
    
    文档依据：
    - Algorithm 1步骤2：L1(θ) = (1/(L-1))·Σ||z(x_{t+1}) - K·z(x_t)||（K为Koopman算子初步近似）
    - Section II Equation 7：L = Σ||z_{t+1} - A z_t - B(u - u0)||_F（最终线性模型误差）
    此处融合两者：用当前迭代的A/B替代K，加入u0项，确保线性模型精度。

    Args:
        z_prev (torch.Tensor): t时刻高维状态z(x_t)，形状[batch_size, N]（N=基函数维度，如128）
        z_next (torch.Tensor): t+1时刻高维状态z(x_{t+1})，形状[batch_size, N]
        A (torch.Tensor): 当前迭代的Koopman矩阵，形状[N, N]（来自`matrix_utils.update_A_B`）
        B (torch.Tensor): 当前迭代的控制矩阵，形状[N, m]（m=控制维度，如倒立摆m=1）
        u_prev (torch.Tensor): t时刻控制输入，形状[batch_size, m]（与`matrix_utils.update_A_B`输入一致）
        u0 (torch.Tensor): 控制固定点（来自PsiMLP.forward_u0），形状[batch_size, m]

    Returns:
        torch.Tensor: 批量平均后的L1损失（标量）
    """
    # 1. 设备一致性确保（避免CPU/GPU混合计算错误）
    device = z_prev.device
    A, B, u_prev, u0 = A.to(device), B.to(device), u_prev.to(device), u0.to(device)
    
    # 2. 计算变换后控制输入v_t = u_t - u0（文档Equation 4）
    v_prev = u_prev - u0  # 形状[batch_size, m]
    
    # 3. 高维线性模型预测z_next（文档Equation 5：z_{t+1} = A z_t + B v_t）
    # 批量矩阵乘法：z_prev [B,N] × A.T [N,N] → [B,N]；v_prev [B,m] × B.T [m,N] → [B,N]
    z_next_pred = torch.matmul(z_prev, A.T) + torch.matmul(v_prev, B.T)  # 形状[batch_size, N]
    
    # 4. 计算每个样本的F范数误差（文档Equation 7的||·||_F）
    # dim=1：对每个样本的N维特征计算F范数（等价于L2范数）
    sample_errors = torch.norm(z_next - z_next_pred, p='fro', dim=1)  # 形状[batch_size]
    
    # 5. 批量平均（文档Algorithm 1步骤2的1/(L-1)，此处L=batch_size，因批量为t=0到L-1的L个样本）
    total_L1 = sample_errors.mean()  # 等价于sum(sample_errors) / batch_size
    
    return total_L1


def compute_L2_loss(
    A: torch.Tensor,
    B: torch.Tensor,
    lambda_rank: float = 0.2,
    lambda_A: float = 0.4,
    lambda_B: float = 0.4
) -> torch.Tensor:
    """
    计算L2损失（文档Algorithm 1步骤3）：
    确保系统能控性与矩阵参数正则化，核心公式为：
    L2(θ) = (N - rank(Cont(A,B))) + ||A||₁ + ||B||₁
    其中Cont(A,B)为能控性矩阵（由`matrix_utils.compute_controllability_matrix`计算）。

    文档依据：
    - Algorithm 1步骤3：L2(θ) = (N - rank(controllability(A,B))) + ||A||₁ + ||B||₁
    - 注：lambda_rank/A/B为超参数，文档未指定权重，默认设为1.0以对齐原文结构，可按需调整。

    Args:
        A (torch.Tensor): Koopman矩阵，形状[N, N]
        B (torch.Tensor): 控制矩阵，形状[N, m]
        lambda_rank (float): 能控性秩惩罚权重（默认1.0，对齐原文）
        lambda_A (float): A矩阵1范数惩罚权重（默认1.0，对齐原文）
        lambda_B (float): B矩阵1范数惩罚权重（默认1.0，对齐原文）

    Returns:
        torch.Tensor: L2损失（标量）
    """
    # 1. 设备一致性确保
    device = A.device
    B = B.to(device)
    
    # 2. 计算能控性矩阵（调用`matrix_utils`的辅助函数，严格对齐文档定义）
    controllability_mat = compute_controllability_matrix(A, B)  # 形状[N, N×m]
    
    # 3. 计算能控性矩阵的秩（奇异值>1e-5视为有效秩，避免数值误差）
    _, singular_vals, _ = torch.svd(controllability_mat)
    rank = (singular_vals > 1e-5).sum().item()  # 有效秩
    N = A.shape[0]
    rank_penalty = lambda_rank * (N - rank)  # 能控性惩罚（秩不足则惩罚增大）
    
    # 4. 计算A/B的1范数（文档Algorithm 1步骤3的正则化项）
    A_l1 = lambda_A * torch.norm(A, p=1)  # A矩阵1范数
    B_l1 = lambda_B * torch.norm(B, p=1)  # B矩阵1范数
    
    # 5. 总L2损失（文档定义的三项之和）
    total_L2 = rank_penalty + A_l1 + B_l1
    
    return total_L2


def compute_total_loss(
    z_prev: torch.Tensor,
    z_next: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    u_prev: torch.Tensor,
    u0: torch.Tensor,
    lambda_L1: float = 1.0,
    lambda_L2: float = 1.0,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算DKRC总损失（文档Algorithm 1步骤4）：
    总损失 = λ_L1·L1 + λ_L2·L2（λ_L1/λ_L2为损失平衡超参数，默认1.0）

    文档依据：
    - Algorithm 1步骤4：L(θ) = L1(θ) + L2(θ)（此处保留λ_L1/λ_L2以支持灵活调参，默认对齐原文）

    Args:
        z_prev (torch.Tensor): t时刻高维状态，形状[batch_size, N]
        z_next (torch.Tensor): t+1时刻高维状态，形状[batch_size, N]
        A (torch.Tensor): Koopman矩阵，形状[N, N]
        B (torch.Tensor): 控制矩阵，形状[N, m]
        u_prev (torch.Tensor): t时刻控制输入，形状[batch_size, m]
        u0 (torch.Tensor): 控制固定点，形状[batch_size, m]
        lambda_L1 (float): L1损失权重（默认1.0，对齐原文）
        lambda_L2 (float): L2损失权重（默认1.0，对齐原文）
        **kwargs: 传递给`compute_L2_loss`的额外参数（如lambda_rank）

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            total_loss: 总损失（标量）
            L1: L1损失（标量）
            L2: L2损失（标量）
    """
    # 1. 计算L1损失（传入A/B/u_prev/u0，对齐线性模型误差）
    L1 = compute_L1_loss(z_prev, z_next, A, B, u_prev, u0)
    
    # 2. 计算L2损失（调用`compute_L2_loss`，支持额外超参数）
    L2 = compute_L2_loss(A, B, **kwargs)
    
    # 3. 计算总损失（文档Algorithm 1步骤4）
    total_loss = lambda_L1 * L1 + lambda_L2 * L2
    
    return total_loss, L1, L2