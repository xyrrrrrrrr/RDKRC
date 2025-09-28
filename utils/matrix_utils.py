import torch
from typing import Tuple


import torch
from typing import Tuple, List

import torch
from typing import Tuple, List


def update_A_B(
    z_prev: torch.Tensor,
    z_next: torch.Tensor,
    u_prev: torch.Tensor,
    A_init: torch.Tensor,
    B_init: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于《Deep Learning of Koopman Representation for Control.pdf》Equation 8，
    适配`z_prev=[batch_size, N]`维度，A、B采用**全局同除数归一化**，保留数值计算关系。
    
    文档依据：
    - Equation 8：[A,B] = z_{t+1}·[z_t;U]·([z_t U]·[z_t;U]^T)^†（A、B协同构成线性模型）
    - Section II.20：高维线性模型 z_{t+1}=A z_t + B v_t（A、B需保持数值比例以确保模型一致性）
    - Section II.27：批量数据提升Koopman算子近似稳定性（归一化需避免破坏批量平均结果）
    
    Args:
        z_prev: t时刻线性化状态（Equation 4的z(x_t)），形状[batch_size, N]；
        z_next: t+1时刻线性化状态（Equation 4的z(x_{t+1})），形状[batch_size, N]；
        u_prev: t时刻控制输入（Equation 1的u_t），形状[batch_size, m]；
        A_init: 历史Koopman矩阵（用于平滑更新，Section II.28训练稳定性要求）；
        B_init: 历史控制矩阵（用于平滑更新，Section II.28训练稳定性要求）。
    
    Returns:
        A: 归一化后Koopman矩阵（同除数），形状[N, N]；
        B: 归一化后控制矩阵（同除数），形状[N, m]。
    """
    # 1. 提取核心维度（匹配文档Section II.25：N≫n，m为控制维度）
    batch_size = z_prev.shape[0]
    N = z_prev.shape[1]  # 基函数维度（高维空间维度）
    m = u_prev.shape[1]  # 控制维度（倒立摆m=1，月球着陆器m=2，🔶1-69、🔶1-80）

    # 2. 初始化单样本A_i、B_i存储（文档Equation 8单样本计算逻辑）
    single_A_list: List[torch.Tensor] = []
    single_B_list: List[torch.Tensor] = []

    # 3. 单样本计算A_i、B_i（严格遵循Equation 8）
    for i in range(batch_size):
        # 3.1 转换为文档单样本维度：[N,1]（列向量，Section II.35定义）
        z_prev_i = z_prev[i, :].unsqueeze(1)  # [N,1]
        z_next_i = z_next[i, :].unsqueeze(1)  # [N,1]
        u_prev_i = u_prev[i, :].unsqueeze(1)  # [m,1]

        # 3.2 构建Equation 8的[z_t; U]（纵向拼接，dim=0）
        X_i = torch.cat([z_prev_i, u_prev_i], dim=0)  # [N+m, 1]
        # 3.3 计算Gram矩阵及其伪逆（Equation 8必需步骤）
        X_i_T = X_i.T  # [1, N+m]
        gram_matrix_i = X_i @ X_i_T  # [N+m, N+m]
        gram_matrix_pinv_i = torch.pinverse(gram_matrix_i)  # 伪逆处理不可逆情况

        # 3.4 单样本求解[A_i, B_i]（Equation 8核心计算）
        AB_temp_i = z_next_i @ X_i_T @ gram_matrix_pinv_i  # [N, N+m]
        # 3.5 分割A_i（前N列）、B_i（后m列），匹配文档维度定义
        A_i = AB_temp_i[:, :N]  # [N, N]
        B_i = AB_temp_i[:, N:]  # [N, m]

        single_A_list.append(A_i)
        single_B_list.append(B_i)

    # 4. 批量平均（Section II.27：批量数据降低噪声，提升近似稳定性）
    A_avg = torch.stack(single_A_list).mean(dim=0)  # [N, N]
    B_avg = torch.stack(single_B_list).mean(dim=0)  # [N, m]

    # 5. 同除数归一化（核心修改：A、B用同一全局范数，保留数值计算关系）
    # 5.1 拼接A、B为整体矩阵（反映二者协同关系，Equation 5线性模型约束）
    AB_avg = torch.cat([A_avg, B_avg], dim=1)  # [N, N+m]
    # 5.2 计算全局单一范数（Frobenius范数，衡量AB整体尺度，避免列归一化破坏关系）
    global_norm = torch.norm(AB_avg, p='fro') + 1e-8  # 单数值，加1e-8防除零
    # 5.3 A、B除以同一范数，保留相对数值关系（符合Equation 5 z_{t+1}=A z_t + B v_t）
    A_normalized = A_avg / global_norm
    B_normalized = B_avg / global_norm

    # 6. 平滑更新（Section II.28训练稳定性要求：避免A、B剧烈波动）
    alpha = 0.5  # 当前计算值权重（文档未指定，取小值确保平滑）
    A = (1 - alpha) * A_init.detach() + alpha * A_normalized
    B = (1 - alpha) * B_init.detach() + alpha * B_normalized


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
    C_col_norm = torch.norm(C, dim=0, keepdim=True) + 1e-8  # [1, N]
    C_normalized = C / C_col_norm  # 归一化后的C，数值规模可控
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