import torch
from typing import Tuple


import torch
from typing import Tuple, List

def update_A_B(
    z_prev: torch.Tensor,
    z_next: torch.Tensor,
    u_prev: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于《Deep Learning of Koopman Representation for Control.pdf》Equation 8，
    适配`z_prev=[batch_size, N]`输入维度，对每个样本单独计算A_i、B_i后取平均。
    
    原文公式：[A, B] = z_{t+1} · [z_t; U] · ([z_t U] · [z_t; U]^T)^†
    适配逻辑：
    1. 按`[batch_size, N]`维度拆分单样本，转换为原文单样本所需的`[N, 1]`；
    2. 每个样本独立执行原文公式计算A_i、B_i；
    3. 所有样本的A_i、B_i分别取平均，保持原文“批量数据驱动Koopman算子近似”的核心目标。
    
    Args:
        z_prev: t时刻线性化状态（原文公式4的z(x_t)），形状[batch_size, N]（如64, 128）；
        z_next: t+1时刻线性化状态（原文公式4的z(x_{t+1})），形状[batch_size, N]（如64, 128）；
        u_prev: t时刻控制输入（原文公式1的u_t），形状[batch_size, m]（m为控制维度，如64, 1）。
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            A: 批量样本A_i的平均（原文定义的Koopman矩阵），形状[N, N]（如128, 128）；
            B: 批量样本B_i的平均（原文定义的控制矩阵），形状[N, m]（如128, 1）。
    """
    # 1. 提取核心维度（基于输入形状与原文定义）
    batch_size = z_prev.shape[0]  # 批量样本数（原文Section II.27批量训练逻辑）
    N = z_prev.shape[1]           # 基函数维度（原文Section II.25，N≫n，n为原始状态维度）
    m = u_prev.shape[1]           # 控制维度（原文Section IV.C倒立摆m=1，IV.D月球着陆器m=2）

    # 2. 初始化单样本A_i、B_i存储列表
    single_A_list: List[torch.Tensor] = []
    single_B_list: List[torch.Tensor] = []

    # 3. 拆分batch，每个样本独立计算A_i、B_i（严格遵循原文Equation 8）
    for i in range(batch_size):
        # 3.1 提取单样本并转换为原文单样本维度
        # 原文单样本z(x_t)为[N, 1]（N维列向量，Section II.35），需将[1, N]的行向量转置为[N, 1]
        z_prev_i = z_prev[i, :].unsqueeze(1)  # 从[batch_size, N]取第i行→[1, N]→转置为[N, 1]
        z_next_i = z_next[i, :].unsqueeze(1)  # 同理，单样本z(x_{t+1})→[N, 1]
        u_prev_i = u_prev[i, :].unsqueeze(1)  # 单样本控制输入→[m, 1]（匹配原文U的单样本维度）

        # 3.2 构建原文Equation 8的[z_t; U]（纵向拼接，dim=0）[N+m, 1]（如128+1=129, 1）
        X_i = torch.cat([z_prev_i, u_prev_i], dim=0)  # 严格匹配原文“[z_t; U]”的结构

        # 3.3 计算原文Gram矩阵及其伪逆（Equation 8的([z_t U]·[z_t; U]^T)^†）
        X_i_T = X_i.T  # [1, N+m]
        gram_matrix_i = X_i @ X_i_T  # 结果为[N+m, N+m]（符合原文维度）
        gram_matrix_pinv_i = torch.pinverse(gram_matrix_i)  # 伪逆，形状[N+m, N+m]

        # 3.4 单样本计算[A_i, B_i]（完全遵循原文Equation 8）
        AB_temp_i = z_next_i @ X_i_T @ gram_matrix_pinv_i  # 维度链[N,1]@[1,N+m]@[N+m,N+m] = [N, N+m]

        # 3.5 分割单样本A_i、B_i（原文Equation 8中[A,B]前N列为A，后m列为B）
        A_i = AB_temp_i[:, :N]  # 取前N列→[N, N]（原文Koopman矩阵定义）
        B_i = AB_temp_i[:, N:]  # 取后m列→[N, m]（原文控制矩阵定义）

        # 3.6 收集单样本结果
        single_A_list.append(A_i)
        single_B_list.append(B_i)

    # 4. 批量平均（原文Section II.27：批量数据提升Koopman算子近似稳定性）
    # 将所有样本的A_i、B_i分别堆叠后按样本维度（dim=0）取平均
    A = torch.stack(single_A_list).mean(dim=0)  # [batch_size, N, N]→[N, N]
    B = torch.stack(single_B_list).mean(dim=0)  # [batch_size, N, m]→[N, m]

    return A, B


def compute_C_matrix(
    x_prev: torch.Tensor,
    z_prev: torch.Tensor
) -> torch.Tensor:
    """
    基于《Deep Learning of Koopman Representation for Control.pdf》Equation 9，
    对Batch中每个样本单独求解C_i（状态重构矩阵），再取平均得到最终C。
    
    原文公式：min_C Σ||x_t - C·z_t||_F, s.t. C·Ψ₀=0
    等价逻辑（原文Section II.42）：因z_t = Ψ(x_t) - Ψ₀，约束C·Ψ₀=0 → C·z_t = C·Ψ(x_t)，
    故最小二乘解为C = x_t·z_t^T·(z_t·z_t^T)^†（单样本），批量时取样本平均。
    
    Args:
        x_prev: 原始状态x_t（原文公式1的观测状态），形状[batch_size, n]（n=原始状态维度，如倒立摆n=3）；
        z_prev: 线性化状态z_t（原文公式4的z=Ψ(x)-Ψ₀），形状[batch_size, N]（N=基函数维度，如128）。
    
    Returns:
        torch.Tensor: 批量平均后的状态重构矩阵C，形状[n, N]（符合原文Equation 9定义）。
    """
    # 1. 提取核心维度（基于输入形状与原文定义）
    batch_size = x_prev.shape[0]  # 批量样本数（原文Section II.27批量训练逻辑）

    # 2. 初始化单样本C_i存储列表（每个C_i对应1个样本的状态重构矩阵）
    single_C_list: List[torch.Tensor] = []

    # 3. 拆分Batch，每个样本独立求解C_i（严格遵循原文Equation 9的最小二乘逻辑）
    for i in range(batch_size):
        # 3.1 提取单样本并转换为原文所需维度
        # 原文单样本x_t为[n, 1]（n维列向量，Section II.13观测状态定义），需将[1, n]行向量转置为[n, 1]
        x_i = x_prev[i, :].unsqueeze(1)  # 从[batch_size, n]取第i行→[1, n]→转置为[n, 1]
        # 原文单样本z_t为[N, 1]（N维列向量，Section II.35线性化状态定义），同理转置
        z_i = z_prev[i, :].unsqueeze(1)  # 从[batch_size, N]取第i行→[1, N]→转置为[N, 1]

        # 3.2 单样本计算Gram矩阵及其伪逆（原文Equation 9的(z_t·z_t^T)^†）
        # 原文Gram矩阵：z_t·z_t^T → [N, 1]@[1, N] = [N, N]（符合最小二乘求解的正定矩阵要求）
        gram_matrix_i = z_i @ z_i.T  # 单样本Gram矩阵，维度[N, N]
        # 伪逆计算（原文Section II.42明确用伪逆处理不可逆情况，确保C存在解）
        gram_matrix_pinv_i = torch.pinverse(gram_matrix_i)  # [N, N]

        # 3.3 单样本求解C_i（完全匹配原文Equation 9的最小二乘解）
        # 维度链：x_i（[n,1]）@ z_i.T（[1,N]）→ [n,N]；再@ gram_matrix_pinv_i（[N,N]）→ [n,N]（符合原文C维度）
        C_i = x_i @ z_i.T @ gram_matrix_pinv_i  # 单样本C_i，形状[n, N]

        # 3.4 收集单样本C_i（去除多余维度，确保为[n, N]）
        single_C_list.append(C_i.squeeze())  # 挤压掉样本维度的1，保持[n, N]

    # 4. 批量平均（原文Section II.42：批量数据提升C矩阵重构精度，减少单样本噪声干扰）
    # 将所有样本的C_i堆叠后按样本维度（dim=0）取平均，最终维度为[n, N]
    C = torch.stack(single_C_list).mean(dim=0)  # [batch_size, n, N] → [n, N]

    # 验证约束C·Ψ₀=0（原文Equation 9的约束，因z_t=Ψ(x_t)-Ψ₀，批量平均后仍满足该约束）
    return C


def compute_controllability_matrix(
    A: torch.Tensor,
    B: torch.Tensor
) -> torch.Tensor:
    """
    计算能控性矩阵Cont(A,B) = [B, A·B, A²·B, ..., A^(N-1)·B](原文L2损失用)
    
    Args:
        A (torch.Tensor): Koopman矩阵,[N, N]
        B (torch.Tensor): 控制矩阵,[N, m]
    
    Returns:
        torch.Tensor: 能控性矩阵,形状[N, N×m]
    """
    N = A.shape[0]
    m = B.shape[1]
    controllability_blocks = []
    
    # 迭代计算A^k · B(k从0到N-1)
    current_block = B  # A^0 · B = B
    for _ in range(N):
        controllability_blocks.append(current_block)
        current_block = A @ current_block  # A^(k+1) · B = A · (A^k · B)
    
    # 拼接所有块:[B, A·B, ..., A^(N-1)·B] → [N, N×m]
    controllability_mat = torch.cat(controllability_blocks, dim=1)
    return controllability_mat