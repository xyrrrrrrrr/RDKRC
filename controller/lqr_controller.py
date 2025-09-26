import numpy as np
import torch
import gym
import scipy.linalg as la
from typing import List
from rdkrc.models import psi_mlp


def solve_discrete_lqr(
    A: torch.Tensor,
    B: torch.Tensor,
    Q: np.ndarray = np.diag([1] * 256),  # 月球着陆器状态成本矩阵（论文IV.D节指定）
    R: np.ndarray = np.diag([0.1, 0.1])
) -> np.ndarray:
    """
    求解离散LQR控制增益K_lqr（论文III节）
    仅适配月球着陆器（Q/R按论文IV.D节固定，不支持修改）。
    
    Args:
        A: Koopman矩阵，shape=[256, 256]
        B: 控制矩阵，shape=[256, 2]
        Q: 状态成本矩阵（固定，论文指定），shape=[256, 256]
        R: 控制成本矩阵（固定，论文指定），shape=[2, 2]
    Returns:
        K_lqr: LQR控制增益，shape=[2, 256]
    """
    # 转换为numpy数组（适配scipy黎卡提求解）
    A_np = A.cpu().detach().numpy()
    B_np = B.cpu().detach().numpy()
    print("A的形状：", A_np.shape)
    print("B的形状：", B_np.shape)
    print("Q的形状：", Q.shape)
    # 求解离散黎卡提方程（论文III节核心公式）
    P = la.solve_discrete_are(A_np, B_np, Q, R)

    # 计算LQR增益（论文推导结果：K_lqr = (B^T P B + R)^-1 B^T P A）
    B_T_P = B_np.T @ P
    K_lqr = la.inv(B_T_P @ B_np + R) @ B_T_P @ A_np
    print(f"LQR增益计算完成：shape={K_lqr.shape}（论文要求[2,256]）")
    return K_lqr


