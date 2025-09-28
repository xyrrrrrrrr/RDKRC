import torch
import numpy as np
import cvxpy as cp
from typing import Tuple, Optional
from rdkrc.models.psi_mlp import PsiMLP


class DKRCMPCController:
    """
    修正 DCP 问题的 DKRC-MPC 控制器类（确保目标函数符合 DCP 规则）
    核心修正：对 QuadForm 矩阵添加正则化，确保半正定；对 C 矩阵归一化，提升数值稳定性。
    """
    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        psi_net: PsiMLP,
        x_star: torch.Tensor,
        u0: np.ndarray,
        pred_horizon: int = 4,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        Qf: Optional[np.ndarray] = None,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
        x_min: Optional[np.ndarray] = None,
        x_max: Optional[np.ndarray] = None,
        eps_reg: float = 1e-6  # 新增：正则化系数，确保矩阵半正定
    ):
        self.device = next(psi_net.parameters()).device
        self.N = A.shape[0]  # 高维空间维度（如256）
        self.n = C.shape[0]  # 原状态维度（6）
        self.m = B.shape[1]  # 控制维度（2）
        self.L = pred_horizon  # 预测时域
        self.eps_reg = eps_reg  # 正则化系数

        # 1. 核心矩阵转换与数值稳定（新增 C 矩阵归一化）
        self.A = A.cpu().detach().numpy()
        self.B = B.cpu().detach().numpy()
        # 新增：C 矩阵列归一化，控制数值范围，避免矩阵乘积过大
        self.C = self._normalize_matrix(C.cpu().detach().numpy(), axis=0)  # 按列归一化
        self.x_star = x_star.cpu().detach().numpy()
        self.u0 = u0
        self.psi_net = psi_net

        # 2. 成本矩阵默认值（保持原文档逻辑）
        if Q is None:
            Q = np.diag([1] * 256)
        if R is None:
            R = np.diag([0.1, 0.1])  # 控制成本
        if Qf is None:
            Qf = 5 * Q  # 终端成本
        self.Q = Q
        self.R = R
        self.Qf = Qf

        # 3. 预计算 DCP 合规的成本矩阵（核心修正：添加正则化确保半正定）
        # 状态跟踪成本矩阵（z 域）：C^T Q C + eps*I（强制半正定）
        self.state_cost_mat = self._ensure_positive_semidefinite(self.C.T @ Q @ self.C)
        # 终端状态成本矩阵：C^T Qf C + eps*I
        self.terminal_cost_mat = self._ensure_positive_semidefinite(self.C.T @ Qf @ self.C)

        # 4. 约束默认值（保持原文档逻辑）
        if u_min is None:
            self.u_min = np.array([0.0, -1.0])
        else:
            self.u_min = u_min
        if u_max is None:
            self.u_max = np.array([1.0, 1.0])
        else:
            self.u_max = u_max
        self.v_min = self.u_min - self.u0
        self.v_max = self.u_max - self.u0

        if x_min is None:
            self.x_min = np.array([-1.5, 0.0, -5.0, -5.0, -np.pi, -8.0])
        else:
            self.x_min = x_min
        if x_max is None:
            self.x_max = np.array([1.5, 1.5, 5.0, 5.0, np.pi, 8.0])
        else:
            self.x_max = x_max

        # 5. 预计算 z_star（始终为0，简化目标函数）
        self.z_star = np.zeros(self.N)

    def _normalize_matrix(self, mat: np.ndarray, axis: int = 0) -> np.ndarray:
        """矩阵归一化（按行/列），减少数值范围过大导致的不稳定"""
        norm = np.linalg.norm(mat, axis=axis, keepdims=True)
        norm[norm < 1e-8] = 1e-8  # 避免除零
        return mat / norm

    def _ensure_positive_semidefinite(self, mat: np.ndarray) -> np.ndarray:
        """确保矩阵半正定（DCP 核心要求）：
        1. 计算矩阵的特征值分解；
        2. 将负特征值置为 eps_reg（正则化）；
        3. 重构矩阵，确保半正定。
        """
        # 特征值分解（对称矩阵）
        eig_vals, eig_vecs = np.linalg.eigh(mat)
        # 修正负特征值（置为 eps_reg，确保半正定）
        eig_vals = np.maximum(eig_vals, self.eps_reg)
        # 重构矩阵
        mat_pos = eig_vecs @ np.diag(eig_vals) @ eig_vecs.T
        # 确保矩阵对称（数值误差修正）
        mat_pos = (mat_pos + mat_pos.T) / 2
        return mat_pos

    def _compute_z(self, x: np.ndarray) -> np.ndarray:
        """计算高维状态 z（保持原逻辑）"""
        self.psi_net.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, device=self.device, dtype=torch.float32).unsqueeze(0)
            x_star_tensor = torch.tensor(self.x_star, device=self.device, dtype=torch.float32).unsqueeze(0)
            z = self.psi_net.compute_z(x_tensor, x_star_tensor).cpu().detach().numpy().squeeze()
        return z

    def _predict_state_sequence(self, z0: np.ndarray, v_seq: cp.Variable) -> Tuple[cp.Variable, list]:
        """预测状态序列（保持原逻辑）"""
        z_seq = cp.Variable((self.L + 1, self.N))
        constraints = [z_seq[0, :] == z0]  # 初始状态约束

        # 线性递推：z_{k+1} = A z_k + B v_k（文档 Equation 5）
        for k in range(self.L):
            constraints.append(z_seq[k+1, :] == z_seq[k, :] @ self.A.T + v_seq[k, :] @ self.B.T)

        return z_seq, constraints

    def _build_optimization_problem(self, z0: np.ndarray) -> Tuple[cp.Problem, cp.Variable, cp.Variable]:
        """构建 DCP 合规的优化问题（核心修正：使用预计算的半正定矩阵）"""
        v_seq = cp.Variable((self.L, self.m))  # 控制输入序列
        z_seq, constraints = self._predict_state_sequence(z0, v_seq)

        # # 1. 控制输入约束（原逻辑）
        # for k in range(self.L):
        #     constraints.append(v_seq[k, :] >= self.v_min)
        #     constraints.append(v_seq[k, :] <= self.v_max)

        # 2. 原状态约束（原逻辑，通过 C 映射）
        with torch.no_grad():
            x_star_tensor = torch.tensor(self.x_star, device=self.device, dtype=torch.float32).unsqueeze(0)
            psi_x_star = self.psi_net(x_star_tensor).cpu().detach().numpy().squeeze()
        C_psi_xstar = self.C @ psi_x_star  # 原状态偏移项

        for k in range(self.L + 1):
            x_k = self.C @ z_seq[k, :] + C_psi_xstar
            # constraints.append(x_k >= self.x_min)
            # constraints.append(x_k <= self.x_max)

        # 3. 目标函数（DCP 合规：使用半正定矩阵的 QuadForm）
        cost = 0.0
        # 阶段成本：∑(z_k^T * state_cost_mat * z_k + v_k^T * R * v_k)
        for k in range(self.L):
            # 状态跟踪成本（z_k 相对于 z_star=0，简化原逻辑）
            state_cost = cp.QuadForm(z_seq[k, :], self.state_cost_mat)
            # 控制成本（R 正定，天然符合 DCP）
            control_cost = cp.QuadForm(v_seq[k, :], self.R)
            cost += state_cost + control_cost

        # 终端成本：z_L^T * terminal_cost_mat * z_L
        terminal_cost = cp.QuadForm(z_seq[self.L, :], self.terminal_cost_mat)
        cost += terminal_cost

        # 4. 定义 DCP 合规的优化问题
        prob = cp.Problem(cp.Minimize(cost), constraints)
        return prob, v_seq, z_seq

    def compute_control(self, x_current: np.ndarray) -> np.ndarray:
        """计算当前步控制输入（保持原接口逻辑）"""
        z0 = self._compute_z(x_current)
        prob, v_seq, _ = self._build_optimization_problem(z0)

        # 求解（使用 ECOS 求解器，适合中小型 QP 问题）
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception as e:
            print(f"MPC 求解异常：{e}，使用默认控制 u0")
            return self.u0

        # 处理求解状态（确保安全）
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"MPC 求解状态：{prob.status}，使用默认控制 u0")
            return self.u0

        # 提取当前步控制输入并转换为原控制空间
        v_current = v_seq.value[0, :]
        u_current = v_current + self.u0
        u_current = np.clip(u_current, self.u_min, self.u_max)  # 双重保险

        return u_current