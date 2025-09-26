import numpy as np
import torch
import scipy.linalg as la


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


def solve_discrete_lqr_v2(
    A: torch.Tensor,
    B: torch.Tensor,
    Q: np.ndarray = None,  # 改为动态生成优化后的Q，保留输入接口
    R: np.ndarray = np.diag([0.1, 0.1])  # R不变（控制成本，论文IV.D节指定）
) -> np.ndarray:
    """
    适配改进PsiMLP：优化Q矩阵权重分配，匹配多尺度基函数的物理意义
    核心变化：Q默认值按基函数分支分配权重，而非全1
    """
    # 1. 转换A/B为numpy，获取Koopman维度N（仍为256，与改进后PsiMLP输出一致）
    A_np = A.cpu().detach().numpy()
    B_np = B.cpu().detach().numpy()
    N = A_np.shape[0]  # N=256，确保Q维度匹配
    
    # 2. 动态生成优化后的Q矩阵（核心改进）
    if Q is None:
        # 基函数分支划分（与改进PsiMLP的_branch_low/_branch_high输出维度对齐）
        low_dim_size = N // 4  # 低维分支维度：256//4=64（慢变关键特征）
        high_dim_size = N - low_dim_size  # 高维分支维度：256-64=192（快变辅助特征）
        
        # 权重分配原则：
        # - 低维分支（关键特征）：权重=10（匹配论文对原始状态y/θ的高惩罚）
        # - 高维分支（辅助特征）：权重=1（抑制震荡，避免过度惩罚）
        Q_diag = np.ones(N, dtype=np.float32)  # 基础权重
        Q_diag[:low_dim_size] = 10.0  # 低维分支（前64维）权重提升到10
        
        Q = np.diag(Q_diag)  # 最终Q形状仍为[256,256]，维度匹配A
    
    # 3. 校验维度（避免意外错误）
    assert A_np.shape == Q.shape, f"A形状{A_np.shape}与Q形状{Q.shape}不匹配，需均为[{N},{N}]"
    
    # 4. 原有LQR求解逻辑不变（确保兼容性）
    P = la.solve_discrete_are(A_np, B_np, Q, R)
    B_T_P = B_np.T @ P
    K_lqr = la.inv(B_T_P @ B_np + R) @ B_T_P @ A_np
    
    print(f"LQR增益计算完成：shape={K_lqr.shape}（应为[2,256]，适配改进PsiMLP）")
    return K_lqr

