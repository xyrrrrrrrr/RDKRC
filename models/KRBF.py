import numpy as np
import math
from typing import Optional, List
from scipy.linalg import pinv, solve_discrete_are


class KRBFModel:
    """
    KRBF模型：基于Koopman算子与RBF的线性预测器（适配月球着陆器）
    对应`KRBF.pdf` 3.2节（EDMD）、4节（数值算法）、5节（MPC）
    """
    def __init__(self, n: int = 6, m: int = 2, N_rbf: int = 100, Np: int = 50, state_low = [-2, -2, -5, -5, -math.pi, -5], state_high = [2, 2, 5, 5, math.pi, 5], action_low=[-1,0], action_high=[1,1]) -> None:
        """
        Args:
            n: 状态维度（月球着陆器6维）
            m: 控制维度（2维：主引擎+侧引擎）
            N_rbf: RBF数量（`KRBF.pdf` 8.1节用100个）
            Np: MPC预测时域（50步=0.5s，符合控制频率）
        """
        self.n = n
        self.m = m
        self.N_rbf = N_rbf
        self.N = n + N_rbf  # 提升空间维度（z ∈ ℝ^N）
        self.Np = Np

        # Koopman算子近似矩阵（`KRBF.pdf` 3.2.1节）
        self.A: Optional[np.ndarray] = None  # [N, N] 状态矩阵
        self.B: Optional[np.ndarray] = None  # [N, m] 控制矩阵
        self.C: Optional[np.ndarray] = None  # [n, N] 状态重构矩阵（`KRBF.pdf` 注1）

        # 数据与RBF参数
        self.X: Optional[np.ndarray] = None  # [n, K] 单步状态数据
        self.U: Optional[np.ndarray] = None  # [m, K] 单步控制数据
        self.Y: Optional[np.ndarray] = None  # [n, K] 单步下一状态数据
        self.state_low = state_low
        self.state_high = state_high
        self.action_low = action_low
        self.action_high = action_high
        self.c_list: List[np.ndarray] = []   # RBF中心列表（[N_rbf]个，每个[n,]）

    def set_data(self, X_single: np.ndarray, U_single: np.ndarray, Y_single: np.ndarray) -> None:
        """设置KRBF训练数据（单步格式，`KRBF.pdf` 4节EDMD输入）"""
        assert X_single.shape[1] == self.n and U_single.shape[1] == self.m, "数据维度不匹配"
        self.X = X_single.T
        self.U = U_single.T
        self.Y = Y_single.T
        # 归一化
        self.X = (self.X - np.array(self.state_low).reshape(-1,1)) / (np.array(self.state_high).reshape(-1,1) - np.array(self.state_low).reshape(-1,1))
        self.Y = (self.Y - np.array(self.state_low).reshape(-1,1)) / (np.array(self.state_high).reshape(-1,1) - np.array(self.state_low).reshape(-1,1))
        self.U = (self.U - np.array(self.action_low).reshape(-1,1)) / (np.array(self.action_high).reshape(-1,1) - np.array(self.action_low).reshape(-1,1))
        # 采样RBF中心（`KRBF.pdf` 8.1节：从训练数据随机选择）
        K = self.X.shape[1]
        center_idx = np.random.choice(K, self.N_rbf, replace=False)
        self.c_list = [self.X[:, idx] for idx in center_idx]
        print(f"[KRBF] 数据加载完成：K={K}个单步样本，RBF中心={self.N_rbf}个")

    def _psi(self, x: np.ndarray) -> np.ndarray:
        """
        提升映射（`KRBF.pdf` 3.2.1节式(12)）：x→z（n维→N维）
        结构：z = [x（原状态）; RBF（薄盘样条）]
        """
        # 1. 原状态分量（`KRBF.pdf` 注1：简化C矩阵求解）
        base = x.copy()
        # 2. RBF分量（`KRBF.pdf` 8.1节：ψ(x) = ||x - c||²·log(||x - c||)）
        rbf_vals = []
        for c in self.c_list:
            dx = x - c
            norm_dx = np.linalg.norm(dx)
            rbf_val = norm_dx ** 2 * np.log(norm_dx) if norm_dx > 1e-8 else 0.0
            rbf_vals.append(rbf_val)
        return np.concatenate([base, np.array(rbf_vals)])  # [N,]



    def solve_koopman(self) -> None:
        """
        求解Koopman算子近似（`KRBF.pdf` 4节EDMD算法）
        核心：最小二乘求解 A、B（正规方程优化大数据效率），C（最优线性投影，遵循式(20)-(22)）
        """
        if self.X is None or self.U is None or self.Y is None:
            raise ValueError("请先调用set_data设置训练数据")
        
        # 维度定义（严格匹配文档：n=原状态维度，N=提升维度，K=样本数）
        n = self.n  # 原状态维度（x ∈ Rⁿ）
        N = self.N  # 提升维度（z ∈ Rᴺ，N = n + N_rbf，N_rbf为RBF数量）
        K = self.X.shape[1]  # 样本数（文档中K为数据量）

        # 1. 计算提升矩阵（`KRBF.pdf` 式(18)-(19)）
        # X_lift: [N, K]，每列对应一个样本的提升向量ψ(x)
        # Y_lift: [N, K]，每列对应一个样本的下一状态提升向量ψ(y)
        print(f"[KRBF] 计算提升矩阵（N={N}，K={K}）...")
        X_lift = np.array([self._psi(self.X[:, idx]) for idx in range(K)]).T  # [N, K]
        Y_lift = np.array([self._psi(self.Y[:, idx]) for idx in range(K)]).T  # [N, K]

        # 2. 求解A、B（`KRBF.pdf` 式(22)正规方程，适配大数据）
        G = np.vstack([X_lift, self.U])  # [N+m, K] 增广矩阵（m=控制维度）
        G_GT = G @ G.T                   # [N+m, N+m]（与K无关，避免大数据下高维计算）
        G_YT = G @ Y_lift.T              # [N+m, N]

        # 摩尔-彭罗斯伪逆求解最优M = [A; B]（式(22)解析解）
        M = pinv(G_GT) @ G_YT
        self.A = M[:N, :].T              # [N, N] 状态矩阵（文档式(2)中A）
        self.B = M[N:, :].T              # [N, m] 控制矩阵（文档式(2)中B）

        # 3. 求解C矩阵（`KRBF.pdf` 式(20)-(22)，核心修复部分）
        # C: [n, N]，最小二乘最优投影：min_C ||X - C·X_lift||_F，解析解C = X·X_lift†
        print(f"[KRBF] 求解C矩阵（原状态维度n={n}，提升维度N={N}）...")
        # X: [n, K]（原状态数据），X_lift†: [K, N]（提升矩阵的伪逆）
        X_lift_pinv = pinv(X_lift)
        self.C = self.X @ X_lift_pinv    # [n, N]（文档式(22)解析解）

        # （可选）验证Remark 1场景：若提升函数前n个为原状态，C应近似[I, 0]，可输出提示
        if hasattr(self, '_psi_includes_original_state') and self._psi_includes_original_state:
            # 检查X_lift前n行是否等于原状态X
            is_psi_include_state = np.allclose(X_lift[:n, :], self.X, atol=1e-6)
            if is_psi_include_state:
                # 构造理论上的[I, 0]矩阵
                C_theory = np.hstack([np.eye(n), np.zeros((n, N - n))])
                # 验证数值解与理论解是否一致
                if np.allclose(self.C, C_theory, atol=1e-6):
                    print(f"[KRBF] 符合Remark 1场景：C ≈ [I_n, 0]（误差<1e-6）")
                else:
                    print(f"[KRBF] 警告：提升函数含原状态，但C与[I_n, 0]偏差较大（建议检查_psi实现）")

        print(f"[KRBF] Koopman矩阵求解完成：A={self.A.shape}, B={self.B.shape}, C={self.C.shape}")

    def predict(self, x0: np.ndarray, U_seq: np.ndarray) -> np.ndarray:
        """
        线性预测（`KRBF.pdf` 2节式(2)）：基于提升空间线性模型预测状态轨迹
        Args:
            x0: 初始状态 [n,]
            U_seq: 控制序列 [m, Np]
        Returns:
            X_pred: 预测轨迹 [n, Np+1]
        """
        if self.A is None or self.B is None or self.C is None:
            raise ValueError("请先调用solve_koopman求解Koopman矩阵")
        
        Np = U_seq.shape[1]
        X_pred = np.zeros((self.n, Np + 1))
        X_pred[:, 0] = x0
        # 归一化初始状态
        x0 = (x0 - np.array(self.state_low)) / (np.array(self.state_high) - np.array(self.state_low))
        # 归一化控制序列
        U_seq = (U_seq - np.array(self.action_low).reshape(-1,1)) / (np.array(self.action_high).reshape(-1,1) - np.array(self.action_low).reshape(-1,1))
        # 初始提升状态（`KRBF.pdf` 式(3)：z0 = ψ(x0)）
        z_curr = self._psi(x0)

        # 逐步预测
        for t in range(Np):
            z_next = self.A @ z_curr + self.B @ U_seq[:, t]
            X_pred[:, t+1] = self.C @ z_next
            z_curr = z_next
        # 反归一化预测状态
        X_pred[:, 1:] = X_pred[:, 1: ] * (np.array(self.state_high).reshape(-1,1) - np.array(self.state_low).reshape(-1,1)) + np.array(self.state_low).reshape(-1,1)

        return X_pred

    def compute_lqr_gain(self, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        求解离散LQR增益（`KRBF.pdf` 5节控制设计）
        Args:
            Q: 状态权重 [n, n]
            R: 控制权重 [m, m]
        Returns:
            K_lqr: LQR增益 [m, N]（适配提升空间z）
        """
        if self.A is None or self.B is None:
            raise ValueError("请先调用solve_koopman求解Koopman矩阵")
        
        # 1. 转换权重到提升空间（`KRBF.pdf` 5.1节：Q' = C^T Q C）
        Q_lift = self.C.T @ Q @ self.C  # [N, N]
        R_lift = R.copy()                # [m, m]

        # 2. 求解Riccati方程（离散LQR）
        P = solve_discrete_are(self.A, self.B, Q_lift, R_lift)
        # 3. 计算LQR增益（K = (B^T P B + R)^-1 B^T P A）
        K_lqr = pinv(self.B.T @ P @ self.B + R_lift) @ self.B.T @ P @ self.A

        print(f"[KRBF] LQR增益计算完成：K_lqr={K_lqr.shape}")
        return K_lqr

    def save_koopman_matrix(self, save_path: str = "./data/krbf_koopman_matrix.npz") -> None:
        """保存Koopman矩阵（A/B/C），避免重复计算"""
        if self.A is None or self.B is None or self.C is None:
            raise ValueError("请先调用solve_koopman求解Koopman矩阵")
        np.savez_compressed(
            save_path,
            A=self.A, B=self.B, C=self.C,
            N_rbf=self.N_rbf, n=self.n, m=self.m
        )
        print(f"[KRBF] Koopman矩阵保存至：{save_path}")

    def load_koopman_matrix(self, load_path: str = "./data/krbf_koopman_matrix.npz") -> None:
        """加载预训练的Koopman矩阵"""
        data = np.load(load_path)
        self.A = data["A"]
        self.B = data["B"]
        self.C = data["C"]
        self.N_rbf = data["N_rbf"]
        self.n = data["n"]
        self.m = data["m"]
        self.N = self.n + self.N_rbf
        print(f"[KRBF] 加载Koopman矩阵：A={self.A.shape}, B={self.B.shape}, C={self.C.shape}")