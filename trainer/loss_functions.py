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
    lambda_rank: float = 0.8,
    lambda_A: float = 0.1,
    lambda_B: float = 0.1
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
    rank_penalty = lambda_rank * (N - rank) / N  # 归一化秩惩罚，避免随N变化过大
    
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
        version (str): 损失版本（"v1"或"v2"）
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

def compute_L_track(z_fused: torch.Tensor, z_ref: torch.Tensor) -> torch.Tensor:
    """
    z_fused: 模型输出的时序特征 [batch, T, 256]
    z_ref: 参考轨迹的时序特征（由参考状态x_ref通过PsiMLP生成） [batch, T, 256]
    返回：时序MSE损失（含帧间平滑项）
    """
    # 1. 帧内跟踪误差（每帧z与参考z的MSE）
    frame_loss = torch.norm(z_fused - z_ref, p=2, dim=2).mean()  # [batch, T] → 标量
    
    # 2. 帧间平滑误差（避免相邻帧跳变，现实硬件需平滑控制）
    smooth_loss = torch.norm(z_fused[:, 1:, :] - z_fused[:, :-1, :], p=2, dim=2).mean()
    
    # 总跟踪损失（平滑项权重0.5，平衡精度与平滑）
    return frame_loss + 0.5 * smooth_loss

def compute_L_obstacle(x_seq: torch.Tensor, obs: torch.Tensor, safe_dist=0.5) -> torch.Tensor:
    """
    x_seq: 时序状态 [batch, T, input_dim]（input_dim含x/y坐标，如2D场景取前2维）
    obs: 障碍物信息 [batch, 4]（x_min,y_min,x_max,y_max）
    safe_dist: 安全距离（如0.5m，根据现实硬件尺寸设定）
    返回：障碍规避损失（仅当距离<安全阈值时惩罚）
    """
    batch_size, T, _ = x_seq.shape
    # 提取状态的x/y坐标（假设前2维为位置）
    x_pos = x_seq[..., 0].unsqueeze(2)  # [batch, T, 1]
    y_pos = x_seq[..., 1].unsqueeze(2)  # [batch, T, 1]
    
    # 计算到障碍物的最小距离（2D轴对齐障碍框）
    # 障碍左边界距离：x_pos - obs[...,0]（x_pos > obs左边界时为正）
    dist_left = x_pos - obs[:, 0].unsqueeze(1).unsqueeze(2).expand(-1, T, -1)  # [batch, T, 1]
    dist_right = obs[:, 2].unsqueeze(1).unsqueeze(2).expand(-1, T, -1) - x_pos  # [batch, T, 1]
    dist_bottom = y_pos - obs[:, 1].unsqueeze(1).unsqueeze(2).expand(-1, T, -1)  # [batch, T, 1]
    dist_top = obs[:, 3].unsqueeze(1).unsqueeze(2).expand(-1, T, -1) - y_pos     # [batch, T, 1]
    
    # 最小距离（仅取正距离，即状态在障碍外的距离）
    min_dist = torch.min(torch.cat([dist_left, dist_right, dist_bottom, dist_top], dim=2), dim=2)[0]  # [batch, T]
    
    # 势场损失：距离越近，惩罚越大（Hinge损失变体）
    obstacle_loss = torch.max(torch.tensor(0.0, device=x_seq.device), safe_dist - min_dist).mean()
    return obstacle_loss

def compute_L_control(u: torch.Tensor, u_min: torch.Tensor, u_max: torch.Tensor) -> torch.Tensor:
    """
    u: 模型输出的控制输入 [batch, T, m]
    u_min: 控制输入下限（如电机最小扭矩） [m]
    u_max: 控制输入上限（如电机最大扭矩） [m]
    返回：控制约束损失
    """
    # 扩展u_min/u_max到批量时序维度
    u_min_expand = u_min.unsqueeze(0).unsqueeze(0).expand(u.shape[0], u.shape[1], -1)  # [batch, T, m]
    u_max_expand = u_max.unsqueeze(0).unsqueeze(0).expand(u.shape[0], u.shape[1], -1)  # [batch, T, m]
    
    # 惩罚超出下限的部分：max(0, u_min - u)
    loss_min = torch.max(torch.tensor(0.0, device=u.device), u_min_expand - u).mean()
    # 惩罚超出上限的部分：max(0, u - u_max)
    loss_max = torch.max(torch.tensor(0.0, device=u.device), u - u_max_expand).mean()
    
    return loss_min + loss_max
