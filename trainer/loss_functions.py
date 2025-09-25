import torch
import torch.nn.functional as F
from typing import Tuple, List
from rdkrc.utils.matrix_utils import compute_controllability_matrix


def compute_L1_loss(
    z_prev_batch: torch.Tensor,  # 原文Batch版z(x_t)：[N, BatchSize]，每列=单个z_prev_i（[N,1]）
    z_next_batch: torch.Tensor   # 原文Batch版z(x_{t+1})：[N, BatchSize]，每列=单个z_next_i（[N,1]）
) -> torch.Tensor:
    """
    分离Batch中每个z的L1损失计算（完全基于原文）
    逻辑：对Batch中每个样本的z_prev_i、z_next_i单独计算，再汇总损失
    对应原文Algorithm 1步骤2：L1(θ) = (1/(L-1))·Σ||z(x_{t+1}) - K·z(x_t)||，K=z(x_{t+1})·z(x_t)^†
    
    Args:
        z_prev_batch: 原文Batch线性化状态z(x_t)，维度[N, BatchSize]（N=基函数维度，如128）
        z_next_batch: 原文Batch线性化状态z(x_{t+1})，维度[N, BatchSize]
    
    Returns:
        total_L1: Batch总L1损失（标量，对应原文求和后平均）
        single_K_list: 每个样本的Koopman算子K_i列表（每个元素[N,N]，对应原文单样本K）
        single_error_list: 每个样本的线性误差列表（每个元素[N,1]，对应原文单样本误差）
    """
    BatchSize = z_prev_batch.shape[1]  # Batch样本数（如64）
    L1_sum = torch.tensor(0.0)  # 初始化总L1损失

    # -------------------------- 步骤1：分离Batch，逐个处理每个样本的z --------------------------
    for i in range(BatchSize):
        # 1.1 提取单个样本的z（严格匹配原文单样本z维度[N,1]）
        z_prev_i = z_prev_batch[:, i].unsqueeze(1)  # 第i个样本的z(x_t)：[N,1]
        z_next_i = z_next_batch[:, i].unsqueeze(1)  # 第i个样本的z(x_{t+1})：[N,1]
        # 1.2 单样本计算Koopman算子K_i（原文K=z(x_{t+1})·z(x_t)^†）
        # 原文单样本Gram矩阵：z_prev_i · z_prev_i^T → [N,1] @ [1,N] = [N,N]（符合原文线性算子近似需求）
        z_prev_i_T = z_prev_i.T  # [1,N]
        gram_matrix_i = z_prev_i @ z_prev_i_T  # [N,N]
        # 单样本伪逆（原文公式8中A/B更新用伪逆，此处完全复用逻辑）
        gram_matrix_pinv_i = torch.pinverse(gram_matrix_i)  # [N,N]
        # 单样本K_i：K_i = z_next_i · z_prev_i^T · gram_matrix_pinv_i → 维度链[N,1]@[1,N]@[N,N] = [N,N]
        K_i = (z_next_i @ z_prev_i_T) @ gram_matrix_pinv_i  # 严格匹配原文K定义

        # 1.3 单样本计算线性误差（原文误差项||z(x_{t+1}) - K·z(x_t)||）
        # 单样本线性预测：K_i · z_prev_i → [N,N] @ [N,1] = [N,1]（匹配z_next_i维度）
        linear_pred_i = K_i @ z_prev_i
        # 单样本误差：z_next_i - linear_pred_i → [N,1]
        error_i = z_next_i - linear_pred_i

        # 1.4 单样本计算L1误差（原文||·||：F范数）
        l1_i = torch.norm(error_i, p='fro')  # 标量，单样本误差范数
        L1_sum += l1_i  # 累加到总L1损失

    # -------------------------- 步骤2：汇总Batch损失（符合原文求和平均逻辑） --------------------------
    # 原文L1(θ)是对时间序列L个样本求和后除以L-1, L= BatchSize
    total_L1 = L1_sum / (BatchSize - 1)  # 标量，Batch总L1损失

    return total_L1


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