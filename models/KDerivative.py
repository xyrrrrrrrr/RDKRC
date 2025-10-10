import gym
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from typing import Optional, List, Tuple
from scipy.linalg import pinv, solve_discrete_are
from scipy.signal import savgol_filter


class KDerivativeKoopman:
    """
    KDerivative-Koopman算法类（适配月球着陆器）
    对应KDERIVATE.pdf核心逻辑：导数基函数、数据驱动Koopman、误差边界、LQR控制
    新增：2*K_steps轨迹预测评估（多轮实验、log10误差计算、结果保存）
    """
    def __init__(self, n: int = 6, m: int = 2, n_deriv: int = 2, 
                 dt: float = 0.01, Np: int = 50):
        """初始化参数（保持原有逻辑，新增轨迹预测相关参数占位）"""
        # 核心维度与时间参数
        self.n = n
        self.m = m
        self.n_deriv = n_deriv  # 导数阶数（状态+各阶导数构成基函数）
        self.dt = dt
        self.Np = Np
        
        # Koopman算子相关矩阵（对应KDERIVATE.pdf 2.3节、3.1节）
        self.Kd: Optional[np.ndarray] = None  # 离散Koopman算子 [w×w]
        self.A: Optional[np.ndarray] = None  # 状态演化矩阵 [w_s×w_s]
        self.B: Optional[np.ndarray] = None  # 控制矩阵 [w_s×m]
        self.C: Optional[np.ndarray] = None  # 状态重构矩阵 [n×w]
        
        # 数据存储（含导数信息）
        self.X: Optional[np.ndarray] = None  # 状态序列 [K, n]
        self.U: Optional[np.ndarray] = None  # 控制序列 [K, m]
        self.Y: Optional[np.ndarray] = None  # 下一状态 [K, n]
        self.dX: Optional[List[np.ndarray]] = None  # 各阶导数 [n_deriv, K, n]
        
        # 基函数与误差边界参数
        self.w_s: int = n * (n_deriv + 1)  # 状态相关基函数维度（状态+各阶导数）
        self.w: int = self.w_s + m  # 总基函数维度（状态相关+控制）
        self.f_max_deriv: Optional[float] = None  # (n_deriv+1)阶导数最大值
        self.env: Optional[gym.Env] = None  # Gym环境实例

    def compute_derivatives(self, X_seq: np.ndarray, filter_window: int = 5) -> List[np.ndarray]:
        """
        数值计算状态的高阶导数（对应KDERIVATE.pdf 3.4节）
        Args:
            X_seq: 状态序列 [ep_len, n]
            filter_window: 滤波窗口（奇数）
        Returns:
            dX: 各阶导数 [n_deriv, ep_len, n]
        """
        ep_len, n = X_seq.shape
        dX = []
        current_X = X_seq.copy()
        filter_window = filter_window if filter_window % 2 == 1 else filter_window + 1
        
        for order in range(1, self.n_deriv + 1):
            dx = np.zeros_like(current_X)
            # 中心差分（中间样本）
            for i in range(1, ep_len - 1):
                dx[i] = (current_X[i+1] - current_X[i-1]) / (2 * self.dt)
            # 边界差分（避免索引越界）
            dx[0] = (current_X[1] - current_X[0]) / self.dt
            dx[-1] = (current_X[-1] - current_X[-2]) / self.dt
            # 平滑滤波（降低噪声）
            if filter_window > 1 and ep_len > filter_window:
                dx = savgol_filter(dx, window_length=filter_window, polyorder=2, axis=0)
            
            dX.append(dx)
            current_X = dx
        return dX

    def estimate_f_max_deriv(self) -> None:
        """估计(n_deriv+1)阶导数最大值（对应KDERIVATE.pdf 3.2节式29）"""
        if self.X is None or self.Y is None:
            raise ValueError("请先生成训练数据")
        
        K = self.X.shape[0]
        e1_list = []
        for k in range(K):
            psi_curr = self._psi(self.X[k], self.U[k])[:self.w_s]
            psi_next = self._psi(self.Y[k], np.zeros(self.m))[:self.w_s]
            e1 = np.linalg.norm(psi_next - psi_curr)
            e1_list.append(e1)
        
        e1_max = np.max(e1_list)
        self.f_max_deriv = e1_max * math.factorial(self.n_deriv + 1) / (self.dt ** (self.n_deriv + 1))
        print(f"[KDeriv] 估计(n_deriv+1)阶导数最大值：{self.f_max_deriv:.4f}")

    def _psi(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        构造Koopman基函数（对应KDERIVATE.pdf 3.1节式20、25）
        结构：Ψ = [s; s^(1); s^(2); ... ; s^(n_deriv); u]
        """
        psi_s = [s]
        # 训练时：从预计算的dX中取对应导数（需确保dX与状态样本索引匹配）
        if self.dX is not None and hasattr(self, '_current_sample_idx'):
            k = self._current_sample_idx
            for i in range(self.n_deriv):
                psi_s.append(self.dX[i][k] if len(self.dX[i]) > k else np.zeros_like(s))
        # 预测时：用实时计算的导数（通过hist_s传入）
        else:
            for i in range(self.n_deriv):
                psi_s.append(np.zeros_like(s))
        
        psi_s = np.concatenate(psi_s, axis=0)
        psi = np.concatenate([psi_s, u], axis=0)
        return psi

    def solve_koopman(self) -> None:
        if self.X is None or self.U is None or self.Y is None:
            raise ValueError("请先生成训练数据")
        K = self.X.shape[0]
        w = self.w
        
        Psi_k = np.zeros((K, w))
        Psi_k1 = np.zeros((K, w))
        print(f"[KDeriv] 构造基函数矩阵（K={K}, w={w}）...")
        
        for k in trange(K):
            self._current_sample_idx = k
            Psi_k[k] = self._psi(self.X[k], self.U[k])
            Psi_k1[k] = self._psi(self.Y[k], np.zeros(self.m))
        
        # 方案：添加正则项到G矩阵，避免Koopman算子求解病态（KRBF.pdf 4.1节）
        reg_eps = 1e-5
        A = (Psi_k1.T @ Psi_k) / K
        G = (Psi_k.T @ Psi_k) / K + reg_eps * np.eye(w)  # G矩阵正则化
        self.Kd = A @ pinv(G)
        
        # 后续拆分A、B、C矩阵（保持不变）
        self.A = self.Kd[:self.w_s, :self.w_s]
        self.B = self.Kd[:self.w_s, self.w_s:]
        self.C = np.hstack([np.eye(self.n), np.zeros((self.n, self.w - self.n))])
        
        # 再次验证A稳定性
        spectral_radius_A = np.max(np.abs(np.linalg.eigvals(self.A)))
        print(f"[KDeriv] Koopman求解完成：A谱半径={spectral_radius_A:.4f}")

    def update_koopman_online(self, X_new: np.ndarray, U_new: np.ndarray, Y_new: np.ndarray) -> None:
        """在线增量更新Koopman算子（对应KDERIVATE.pdf 4.2节式39-40）"""
        if self.Kd is None:
            raise ValueError("请先调用solve_koopman求解初始算子")
        K_old = self.X.shape[0]
        K_new = X_new.shape[0]
        K_total = K_old + K_new
        
        # 计算新数据基函数
        Psi_k_new = np.zeros((K_new, self.w))
        Psi_k1_new = np.zeros((K_new, self.w))
        for k in range(K_new):
            self._current_sample_idx = K_old + k
            Psi_k_new[k] = self._psi(X_new[k], U_new[k])
            Psi_k1_new[k] = self._psi(Y_new[k], np.zeros(self.m))
        
        # 增量更新A和G
        A_old = (self.Kd @ pinv((self.Psi_k.T @ self.Psi_k) / K_old)) * K_old
        G_old = (self.Psi_k.T @ self.Psi_k) / K_old * K_old
        
        A_new = (A_old + Psi_k1_new.T @ Psi_k_new) / K_total
        G_new = (G_old + Psi_k_new.T @ Psi_k_new) / K_total
        
        # 更新算子与拆分矩阵
        self.Kd = A_new @ pinv(G_new)
        self.A = self.Kd[:self.w_s, :self.w_s]
        self.B = self.Kd[:self.w_s, self.w_s:]
        
        # 扩展全局数据
        self.X = np.concatenate([self.X, X_new], axis=0)
        self.U = np.concatenate([self.U, U_new], axis=0)
        self.Y = np.concatenate([self.Y, Y_new], axis=0)
        print(f"[KDeriv] 在线更新完成：样本数{K_old}→{K_total}")

    def compute_error_bound(self, T: float) -> float:
        """计算预测误差边界（对应KDERIVATE.pdf 3.1节定理1式22）"""
        if self.f_max_deriv is None:
            raise ValueError("请先调用estimate_f_max_deriv")
        return (T ** (self.n_deriv + 1)) / math.factorial(self.n_deriv + 1) * self.f_max_deriv

    def compute_lqr_gain(self, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """求解LQR增益（对应KDERIVATE.pdf 2.4节）"""
        if self.A is None or self.B is None:
            raise ValueError("请先调用solve_koopman")
        
        Q_lift = self.C[:self.w_s, :self.w_s].T @ Q @ self.C[:self.w_s, :self.w_s]
        R_lift = R.copy()
        P = solve_discrete_are(self.A, self.B, Q_lift, R_lift)
        K_lqr = pinv(self.B.T @ P @ self.B + R_lift) @ self.B.T @ P @ self.A
        print(f"[KDeriv] LQR增益维度：{K_lqr.shape}")
        return K_lqr

    def predict(self, s0: np.ndarray, U_seq: np.ndarray, hist_s: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        预测未来轨迹（对应KDERIVATE.pdf 3.1节式20）
        适配2*K_steps长度，支持用历史状态计算初始导数
        Args:
            s0: 初始状态 [n]
            U_seq: 控制序列 [2*K_steps, m]
            hist_s: 历史状态 [n_deriv, n]（用于计算初始导数）
        Returns:
            S_pred: 预测轨迹 [2*K_steps + 1, n]
        """
        if self.A is None or self.B is None or self.C is None:
            raise ValueError("请先调用solve_koopman")
        seq_length = U_seq.shape[0]  # 2*K_steps
        S_pred = np.zeros((seq_length + 1, self.n))
        S_pred[0] = s0
        
        # 1. 计算初始基函数（含导数）
        if hist_s is not None and len(hist_s) >= self.n_deriv:
            # 用历史状态计算初始各阶导数
            hist_s_arr = np.array(hist_s)  # [n_deriv, n]
            derivs = self.compute_derivatives(hist_s_arr[None, :, :].squeeze(0))  # [n_deriv, 1, n]
            psi_s0 = np.concatenate([s0] + [d[0] for d in derivs], axis=0)
        else:
            # 无历史数据时用零初始化导数
            psi_s0 = np.concatenate([s0] + [np.zeros_like(s0) for _ in range(self.n_deriv)], axis=0)
        
        # 2. 逐步预测（2*K_steps步）
        psi_curr = psi_s0
        for t in range(seq_length):
            psi_next = self.A @ psi_curr + self.B @ U_seq[t]
            # 重构原状态（拼接控制维度的零向量以匹配C矩阵维度）
            S_pred[t+1] = self.C @ np.concatenate([psi_next, np.zeros(self.m)], axis=0)
            psi_curr = psi_next
        
        return S_pred

    def evaluate_trajectory_prediction(
        self,
        test_data_path: str,
        num_experiments: int = 4,
        save_results: bool = True,
        result_path: str = "./results"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        新增：2*K_steps轨迹预测评估（核心功能）
        重复num_experiments次实验，计算各时间步log10误差均值并保存
        Args:
            test_data_path: 扩展测试数据路径（含2*K_steps序列）
            num_experiments: 实验次数（默认4次）
            save_results: 是否保存结果
            result_path: 结果保存路径
        Returns:
            mean_errors: 各时间步平均误差 [2*K_steps + 1]
            log10_errors: 各时间步log10误差均值 [2*K_steps + 1]
        """
        # 1. 初始化与加载数据
        os.makedirs(result_path, exist_ok=True)
        test_data = np.load(test_data_path)
        
        # 提取扩展测试数据（2*K_steps序列）
        extended_X_seq = test_data['extended_X_seq']  # [num_test_ep, 2*K_steps, n]
        extended_U_seq = test_data['extended_U_seq']  # [num_test_ep, 2*K_steps, m]
        extended_Y_seq = test_data['extended_Y_seq']  # [num_test_ep, 2*K_steps, n]
        K_steps = test_data['K_steps'].item()
        seq_length = test_data['seq_length'].item()  # 2*K_steps
        num_test_ep = extended_X_seq.shape[0]
        
        print(f"\n[KDeriv-Pred] 开始{num_experiments}次轨迹预测实验（长度={seq_length}=2*{K_steps}步）")
        print(f"[KDeriv-Pred] 测试数据规模：{num_test_ep}个序列，每个{seq_length}步")
        
        # 2. 存储所有实验的误差
        all_errors = []

        # 3. 重复实验
        for exp_idx in range(num_experiments):
            print(f"\n[KDeriv-Pred] 实验 {exp_idx+1}/{num_experiments}")
            episode_errors = []  # 存储每个测试序列的误差
            
            # 遍历所有测试序列
            for ep_idx in trange(num_test_ep, desc="处理测试序列"):
                # a. 提取当前序列的初始状态、控制序列、真实状态
                initial_state = extended_X_seq[ep_idx, 0]  # 初始状态 [n]
                control_seq = extended_U_seq[ep_idx]      # 控制序列 [2*K_steps, m]
                
                # 构建真实状态序列：[初始状态] + [extended_Y_seq] → [2*K_steps+1, n]
                true_states = np.vstack([initial_state, extended_Y_seq[ep_idx]])
                
                # b. 准备历史状态（用于计算初始导数，取前n_deriv个状态）
                if seq_length >= self.n_deriv:
                    hist_s = extended_X_seq[ep_idx, :self.n_deriv].tolist()  # [n_deriv, n]
                else:
                    # 序列过短时用初始状态重复填充（避免导数计算失败）
                    hist_s = [initial_state.copy() for _ in range(self.n_deriv)]
                
                # c. 预测轨迹
                pred_states = self.predict(
                    s0=initial_state,
                    U_seq=control_seq,
                    hist_s=hist_s
                )
                
                # d. 计算每个时间步的欧氏距离误差
                step_errors = np.linalg.norm(pred_states - true_states, axis=1)  # [2*K_steps+1]
                episode_errors.append(step_errors)
            
            # 4. 计算当前实验的平均误差（所有测试序列的均值）
            exp_errors = np.mean(episode_errors, axis=0)  # [2*K_steps+1]
            all_errors.append(exp_errors)
            print(f"[KDeriv-Pred] 实验 {exp_idx+1} 误差范围：{np.min(exp_errors):.6f} ~ {np.max(exp_errors):.6f}")

        # 5. 计算所有实验的平均误差与log10误差
        mean_errors = np.mean(all_errors, axis=0)  # 4次实验的均值
        # 加1e-10避免log(0)，确保数值稳定性
        log10_errors = np.log10(mean_errors + 1e-10)

        # 6. 保存实验结果
        if save_results:
            result_file = os.path.join(result_path, f"kderiv_pred_results_K{K_steps}.npz")
            np.savez_compressed(
                result_file,
                mean_errors=mean_errors,
                log10_errors=log10_errors,
                K_steps=K_steps,
                seq_length=seq_length,
                num_experiments=num_experiments,
                all_errors=np.array(all_errors),  # 每次实验的误差详情
                n_deriv=self.n_deriv  # 导数阶数（用于复现）
            )
            print(f"\n[KDeriv-Pred] 实验结果保存至：{result_file}")

        # 7. 绘制误差曲线（原始误差 + log10误差）
        self._plot_prediction_errors(mean_errors, log10_errors, seq_length, K_steps)

        return mean_errors, log10_errors

    def _plot_prediction_errors(
        self,
        mean_errors: np.ndarray,
        log10_errors: np.ndarray,
        seq_length: int,
        K_steps: int
    ) -> None:
        """绘制轨迹预测误差曲线（原始误差 + log10误差）"""
        plt.figure(figsize=(12, 8))
        time_steps = np.arange(seq_length + 1)  # 0 ~ 2*K_steps
        
        # 子图1：原始平均误差
        plt.subplot(2, 1, 1)
        plt.plot(time_steps, mean_errors, color='#2E86AB', linewidth=2.5, label=f'Mean Error (2*K={seq_length} steps)')
        plt.ylabel('Euclidean Distance Error', fontsize=12)
        plt.title(f'KDerivative-Koopman Trajectory Prediction Errors (K={K_steps}, n_deriv={self.n_deriv})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # 子图2：log10误差（突出误差变化趋势）
        plt.subplot(2, 1, 2)
        plt.plot(time_steps, log10_errors, color='#A23B72', linewidth=2.5, label=f'log10(Mean Error)')
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('log10(Euclidean Distance Error)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # 保存图片
        os.makedirs("./fig/lunarlander", exist_ok=True)
        plot_path = os.path.join("./fig/lunarlander", f"kderiv_pred_errors_K{K_steps}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[KDeriv-Pred] 误差曲线保存至：{plot_path}")

    def test_lander_lqr(self, K_lqr: np.ndarray, s_des: np.ndarray, num_episodes: int = 100, 
                       max_steps: int = 500, seed: int = 2, version: str = "KDeriv-LQR") -> List[float]:
        """原有LQR测试方法（保持不变）"""
        self.env = gym.make("LunarLanderContinuous-v2")
        self.env.seed(seed)
        episode_scores = []
        all_trajectories = []
        landing_positions = []
        success_count = 0

        print(f"\n[Test LQR] 开始{num_episodes}回合KDeriv-LQR测试...")
        for ep in range(num_episodes):
            s_prev = self.env.reset()[:self.n]
            hist_s = [s_prev.copy() for _ in range(self.n_deriv)]
            done = False
            total_score = 0.0
            step = 0
            trajectory = [(s_prev[0], s_prev[1])]

            while not done and step < max_steps:
                u0 = self.control_lqr(s_prev, s_des, K_lqr, hist_s)
                s_next, reward, done, _ = self.env.step(u0)
                s_next = s_next[:self.n]
                # 更新历史状态
                hist_s.pop(0)
                hist_s.append(s_prev.copy())
                # 记录数据
                total_score += reward
                trajectory.append((s_next[0], s_next[1]))
                s_prev = s_next
                step += 1

            # 统计落地信息
            landing_x, landing_y = s_prev[0], s_prev[1]
            landing_positions.append((landing_x, landing_y))
            all_trajectories.append(trajectory)
            episode_scores.append(total_score)

            # 成功着陆判断
            if abs(landing_x) <= 0.5 and -0.2 <= landing_y <= 0.2:
                success_count += 1
            print(f"[Test LQR] 回合 {ep+1:2d} | 得分：{total_score:5.1f} | 步数：{step:3d} | 落地：(x={landing_x:.3f}, y={landing_y:.3f})")

        self.env.close()

        # 误差边界验证
        T = max_steps * self.dt
        error_bound = self.compute_error_bound(T)
        actual_errors = [np.linalg.norm(pos - s_des[:2]) for pos in landing_positions]
        print(f"\n[Test LQR] 误差边界验证：")
        print(f"  预测时域T={T:.2f}s | 理论误差边界={error_bound:.4f} | 实际最大误差={np.max(actual_errors):.4f}")

        # 绘制轨迹图
        self._plot_trajectory(all_trajectories, landing_positions, s_des, version)

        # 测试总结
        avg_score = np.mean(episode_scores)
        std_score = np.std(episode_scores)
        print(f"\n[Test LQR] 测试总结：")
        print(f"  平均得分：{avg_score:.1f}±{std_score:.1f} | 成功着陆：{success_count}/{num_episodes}")
        return episode_scores

    def control_lqr(self, s_curr: np.ndarray, s_des: np.ndarray, K_lqr: np.ndarray, 
                   hist_s: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """原有LQR控制生成方法（保持不变）"""
        # 计算当前和目标的基函数
        if hist_s is None:
            psi_curr = np.concatenate([s_curr] + [np.zeros_like(s_curr) for _ in range(self.n_deriv)], axis=0)
            psi_des = np.concatenate([s_des] + [np.zeros_like(s_des) for _ in range(self.n_deriv)], axis=0)
        else:
            hist_s_arr = np.array(hist_s)
            derivs_curr = self.compute_derivatives(hist_s_arr[None, :, :].squeeze(0))
            derivs_des = [np.zeros_like(s_des) for _ in range(self.n_deriv)]
            psi_curr = np.concatenate([s_curr] + [d[0] for d in derivs_curr], axis=0)
            psi_des = np.concatenate([s_des] + derivs_des, axis=0)
        
        # LQR控制律
        u0 = -K_lqr @ (psi_curr - psi_des)
        # 控制约束裁剪
        if self.env is not None:
            u0 = np.clip(u0, self.env.action_space.low, self.env.action_space.high)
        return u0

    def _plot_trajectory(self, all_trajectories: List[List[Tuple[float, float]]], 
                        landing_positions: List[Tuple[float, float]], s_des: np.ndarray, version: str) -> None:
        """原有轨迹绘图方法（保持不变）"""
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10.colors

        # 绘制所有轨迹
        for ep, traj in enumerate(all_trajectories):
            x_coords = [p[0] for p in traj]
            y_coords = [p[1] for p in traj]
            plt.plot(x_coords, y_coords, color=colors[ep % len(colors)], alpha=0.7)

        # 标注关键信息
        mean_x = np.mean([p[0] for p in landing_positions])
        mean_y = np.mean([p[1] for p in landing_positions])
        std_x = np.std([p[0] for p in landing_positions], ddof=1)
        std_y = np.std([p[1] for p in landing_positions], ddof=1)

        plt.scatter(s_des[0], s_des[1], color="red", marker="s", s=80, label="Target Landing Pos")
        plt.scatter(mean_x, mean_y, color="blue", marker="o", s=100, label=f"Landing Mean (x={mean_x:.3f}, y={mean_y:.3f})")
        plt.gca().add_patch(plt.Rectangle(
            (mean_x - std_x, mean_y - std_y), 2*std_x, 2*std_y,
            color="blue", alpha=0.2, linestyle="--", label="Landing Std Range (±1σ)"
        ))
        plt.axhline(y=0, color="black", linestyle="--", alpha=0.8, label="Landing Pad (y=0)")

        # 图表配置
        plt.xlim(-1.5, 1.5)
        plt.ylim(0, 1.5)
        plt.xlabel("X Position (Horizontal)", fontsize=12)
        plt.ylabel("Y Position (Altitude)", fontsize=12)
        plt.title(f"Lunar Lander Trajectory Summary ({version})", fontsize=14)
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1), fontsize=10)
        plt.grid(True, alpha=0.5)

        # 保存图片
        os.makedirs("./fig/lunarlander", exist_ok=True)
        plt.savefig(f"./fig/lunarlander/kderiv_lander_trajectory_{version}.png", bbox_inches="tight", dpi=300)
        plt.close()


def main():
    """完整流程：数据加载→Koopman求解→LQR测试→2*K_steps轨迹预测评估"""
    # 1. 配置参数
    config = {
        "seed": 2,
        "num_test_episodes": 100,  # 测试回合数
        "n_deriv": 2,             # 导数阶数（匹配数据生成时的n_deriv=2）
        "dt": 0.01,               # 采样周期
        "Np": 50,                 # MPC预测时域
        "Q_weight": np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # 状态权重
        "R_weight": np.diag([0.1, 0.1]),  # 控制权重
        # 数据路径（需与数据生成脚本的输出路径一致）
        "train_data_path": "./data/train_data_LunarLanderContinuous-v2_n6_m2_deriv2_K15_seed2.npz",
        "extended_test_path": "./data/test_data_LunarLanderContinuous-v2_ep100_K15_seed2_extended.npz",
        "result_path": "./results"
    }

    # 2. 初始化KDerivativeKoopman
    kderiv = KDerivativeKoopman(
        n=6, m=2,
        n_deriv=config["n_deriv"],
        dt=config["dt"],
        Np=config["Np"]
    )

    # 3. 加载训练数据（预生成的含导数数据）
    train_data = np.load(config["train_data_path"])
    kderiv.X = train_data["X"]  # [K, n]
    kderiv.U = train_data["U"]  # [K, m]
    kderiv.Y = train_data["Y"]  # [K, n]
    kderiv.dX = train_data["dX"]  # [n_deriv, K, n]
    kderiv.estimate_f_max_deriv()  # 估计导数最大值（用于误差边界）
    print(f"[Main] 加载训练数据：X={kderiv.X.shape}, U={kderiv.U.shape}, dX={kderiv.dX.shape}")

    # 4. 求解Koopman算子（核心步骤）
    kderiv.solve_koopman()

    # 5. 计算LQR增益并测试控制性能（可选，验证模型有效性）
    K_lqr = kderiv.compute_lqr_gain(
        Q=config["Q_weight"],
        R=config["R_weight"]
    )
    s_des = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 着陆目标状态
    kderiv.test_lander_lqr(
        K_lqr=K_lqr,
        s_des=s_des,
        num_episodes=config["num_test_episodes"],
        seed=config["seed"],
        version=f"KDeriv-n{config['n_deriv']}"
    )

    # 6. 新增：2*K_steps轨迹预测评估（4次实验）
    if os.path.exists(config["extended_test_path"]):
        mean_errors, log10_errors = kderiv.evaluate_trajectory_prediction(
            test_data_path=config["extended_test_path"],
            num_experiments=4,  # 重复4次实验
            save_results=True,
            result_path=config["result_path"]
        )

        # 打印评估总结
        print("\n[KDeriv-Pred] 轨迹预测评估总结")
        print(f"  预测长度：{len(mean_errors)-1} 步（2*K_steps）")
        print(f"  平均log10误差：{np.mean(log10_errors):.4f}")
        print(f"  各时间步log10误差（每5步）：")
        for i in range(0, len(log10_errors), 5):
            print(f"    第{i:2d}步：{log10_errors[i]:.4f}")
    else:
        print(f"\n[警告] 扩展测试数据不存在：{config['extended_test_path']}，请先运行数据生成脚本")


if __name__ == "__main__":
    main()