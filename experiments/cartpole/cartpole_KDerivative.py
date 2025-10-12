import gym
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from typing import Optional, List, Tuple
from scipy.linalg import pinv, solve_discrete_are
from scipy.signal import savgol_filter


class KDerivativeKoopmanCartPole:
    """KDerivative-Koopman算法类（适配CartPole-v1环境）"""
    def __init__(self, n: int = 4, m: int = 1, n_deriv: int = 1, 
                 dt: float = 0.02, Np: int = 30):
        """初始化参数（适配CartPole状态维度：4维状态，1维离散动作）"""
        self.n = n  # CartPole状态维度：[小车位置x, 小车速度x_dot, 杆角度theta, 杆角速度theta_dot]
        self.m = m  # 动作维度：1维（离散值0=左推，1=右推）
        self.n_deriv = n_deriv  # 导数阶数
        self.dt = dt  # CartPole默认采样间隔（0.02s/步）
        self.Np = Np
        
        # Koopman算子相关矩阵
        self.Kd: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None
        self.C: Optional[np.ndarray] = None
        
        # 数据存储
        self.X: Optional[np.ndarray] = None
        self.U: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        self.dX: Optional[List[np.ndarray]] = None
        
        # 基函数参数计算（关键修复：正确计算维度）
        self.w_s: int = self.n * (self.n_deriv + 1)  # 状态+导数的基函数维度
        self.w: int = self.w_s + self.m              # 总基函数维度（状态+导数+动作）
        self.f_max_deriv: Optional[float] = None
        self.env: Optional[gym.Env] = None

    def compute_derivatives(self, X_seq: np.ndarray, filter_window: int = 5) -> List[np.ndarray]:
        """计算状态高阶导数（适配CartPole 4维状态）"""
        ep_len, n = X_seq.shape
        dX = []
        current_X = X_seq.copy()
        filter_window = filter_window if filter_window % 2 == 1 else filter_window + 1
        
        for order in range(1, self.n_deriv + 1):
            dx = np.zeros_like(current_X)
            # 中心差分（中间步）+ 前向/后向差分（边界步）
            for i in range(1, ep_len - 1):
                dx[i] = (current_X[i+1] - current_X[i-1]) / (2 * self.dt)
            dx[0] = (current_X[1] - current_X[0]) / self.dt  # 前向差分
            dx[-1] = (current_X[-1] - current_X[-2]) / self.dt  # 后向差分
            
            # 平滑滤波（避免噪声影响导数计算）
            if filter_window > 1 and ep_len > filter_window:
                dx = savgol_filter(dx, window_length=filter_window, polyorder=2, axis=0)
            
            dX.append(dx)
            current_X = dx
        return dX

    def estimate_f_max_deriv(self) -> None:
        """估计高阶导数最大值（适配CartPole 4维状态）"""
        if self.X is None or self.Y is None:
            raise ValueError("请先生成训练数据")
        
        K = self.X.shape[0]
        e1_list = []
        for k in range(K):
            psi_curr = self._psi(self.X[k], self.U[k])[:self.w_s]  # 仅取状态+导数部分
            psi_next = self._psi(self.Y[k], np.zeros(self.m))[:self.w_s]
            e1 = np.linalg.norm(psi_next - psi_curr)  # 欧氏距离衡量基函数变化
            e1_list.append(e1)
        
        e1_max = np.max(e1_list)
        # 基于阶乘和采样间隔估算导数上限
        self.f_max_deriv = e1_max * math.factorial(self.n_deriv + 1) / (self.dt ** (self.n_deriv + 1))
        print(f"[KDeriv-CartPole] 估计高阶导数最大值：{self.f_max_deriv:.4f}")

    def _psi(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        """构造基函数（适配CartPole 4维状态+1维离散动作）"""
        psi_s = [s]  # 初始状态（4维）
        # 拼接高阶导数（若已计算）
        if self.dX is not None and hasattr(self, '_current_sample_idx'):
            k = self._current_sample_idx
            for i in range(self.n_deriv):
                # 确保索引不越界
                if i < len(self.dX) and k < len(self.dX[i]):
                    psi_s.append(self.dX[i][k])
                else:
                    psi_s.append(np.zeros_like(s))
        else:
            # 无导数时补0（初始化阶段）
            for i in range(self.n_deriv):
                psi_s.append(np.zeros_like(s))
        
        # 拼接状态+导数和动作
        psi_s = np.concatenate(psi_s, axis=0)
        # 确保动作是1维的
        u_flat = u.flatten() if u.ndim > 1 else u
        psi = np.concatenate([psi_s, u_flat], axis=0)
        return psi

    def solve_koopman(self) -> None:
        """求解Koopman算子（修复基函数维度匹配问题）"""
        if self.X is None or self.U is None or self.Y is None:
            raise ValueError("请先生成训练数据")
        K = self.X.shape[0]
        
        # 打印维度信息用于调试
        print(f"[调试信息] 基函数总维度 w={self.w}, 状态+导数维度 w_s={self.w_s}")
        print(f"[调试信息] 样本数 K={K}, 状态维度 n={self.n}, 动作维度 m={self.m}")
        
        # 构造基函数矩阵 Psi_k (当前步) 和 Psi_k1 (下一步)
        Psi_k = np.zeros((K, self.w))  # 关键修复：使用计算好的w作为维度
        Psi_k1 = np.zeros((K, self.w))
        
        print(f"[KDeriv-CartPole] 构造基函数矩阵（K={K}, w={self.w}）...")
        
        for k in trange(K):
            self._current_sample_idx = k
            # 生成基函数并检查维度
            psi_k = self._psi(self.X[k], self.U[k])
            psi_k1 = self._psi(self.Y[k], np.zeros(self.m))
            
            # 调试：检查基函数维度
            if psi_k.shape[0] != self.w:
                print(f"[错误] 基函数维度不匹配: 期望 {self.w}, 实际 {psi_k.shape[0]}")
            
            Psi_k[k] = psi_k    # 当前步基函数
            Psi_k1[k] = psi_k1  # 下一步基函数（动作置0）
        
        # 正则化避免矩阵奇异
        reg_eps = 1e-5
        A = (Psi_k1.T @ Psi_k) / K
        G = (Psi_k.T @ Psi_k) / K + reg_eps * np.eye(self.w)
        self.Kd = A @ pinv(G)  # 求解Koopman算子
        
        # 分解A/B矩阵（状态转移/动作输入）
        self.A = self.Kd[:self.w_s, :self.w_s]  # 状态-状态转移
        self.B = self.Kd[:self.w_s, self.w_s:]  # 动作-状态转移
        # 观测矩阵C：仅提取原始4维状态（忽略导数和动作）
        self.C = np.hstack([np.eye(self.n), np.zeros((self.n, self.w - self.n))])
        
        # 验证A矩阵稳定性（谱半径<1为稳定）
        spectral_radius_A = np.max(np.abs(np.linalg.eigvals(self.A)))
        print(f"[KDeriv-CartPole] Koopman求解完成：A谱半径={spectral_radius_A:.4f}")

    def compute_error_bound(self, T: float) -> float:
        """计算预测误差边界"""
        if self.f_max_deriv is None:
            raise ValueError("请先调用estimate_f_max_deriv")
        return (T ** (self.n_deriv + 1)) / math.factorial(self.n_deriv + 1) * self.f_max_deriv

    def compute_lqr_gain(self, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """求解LQR增益（适配CartPole 4维状态权重）"""
        if self.A is None or self.B is None:
            raise ValueError("请先调用solve_koopman")
        
        # 提升Q矩阵维度（从原始状态到基函数空间）
        Q_lift = self.C[:self.w_s, :self.w_s].T @ Q @ self.C[:self.w_s, :self.w_s]
        R_lift = R.copy()  # 动作权重保持1维
        # 离散LQR Riccati方程求解
        P = solve_discrete_are(self.A, self.B, Q_lift, R_lift)
        K_lqr = pinv(self.B.T @ P @ self.B + R_lift) @ self.B.T @ P @ self.A
        print(f"[KDeriv-CartPole] LQR增益维度：{K_lqr.shape}")
        return K_lqr

    def predict(self, s0: np.ndarray, U_seq: np.ndarray, hist_s: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """预测轨迹（适配CartPole 4维状态）"""
        if self.A is None or self.B is None or self.C is None:
            raise ValueError("请先调用solve_koopman")
        seq_length = U_seq.shape[0]
        # 预测状态矩阵：(序列长度+1, 4维状态)
        S_pred = np.zeros((seq_length + 1, self.n))
        S_pred[0] = s0  # 初始状态
        
        # 初始化基函数（利用历史状态计算初始导数）
        if hist_s is not None and len(hist_s) >= self.n_deriv:
            hist_s_arr = np.array(hist_s)
            derivs = self.compute_derivatives(hist_s_arr[None, :, :].squeeze(0))
            psi_s0 = np.concatenate([s0] + [d[0] for d in derivs], axis=0)
        else:
            # 无历史时导数补0
            psi_s0 = np.concatenate([s0] + [np.zeros_like(s0) for _ in range(self.n_deriv)], axis=0)
        
        # 逐步预测
        psi_curr = psi_s0
        for t in range(seq_length):
            psi_next = self.A @ psi_curr + self.B @ U_seq[t].flatten()  # 确保动作是1维的
            # 映射回原始状态空间
            S_pred[t+1] = self.C @ np.concatenate([psi_next, np.zeros(self.m)], axis=0)
            psi_curr = psi_next
        
        return S_pred

    def evaluate_trajectory_prediction(
        self,
        test_data_path: str,
        num_experiments: int = 4,
        save_results: bool = True,
        result_path: str = "./results/cartpole"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """评估CartPole轨迹预测性能"""
        os.makedirs(result_path, exist_ok=True)
        test_data = np.load(test_data_path)
        
        # 加载CartPole测试数据（4维状态，1维离散动作）
        extended_X_seq = test_data['extended_X_seq']
        extended_U_seq = test_data['extended_U_seq']
        extended_Y_seq = test_data['extended_Y_seq']
        K_steps = 15
        seq_length = extended_X_seq.shape[1]
        num_test_ep = extended_X_seq.shape[0]
        
        print(f"\n[KDeriv-CartPole] 开始{num_experiments}次轨迹预测实验（长度={seq_length}=2*{K_steps}步）")
        print(f"[KDeriv-CartPole] 测试数据规模：{num_test_ep}个序列（4维状态）")
        
        all_errors = []
        for exp_idx in range(num_experiments):
            print(f"\n[KDeriv-CartPole] 实验 {exp_idx+1}/{num_experiments}")
            episode_errors = []
            
            for ep_idx in trange(num_test_ep, desc="处理测试序列"):
                initial_state = extended_X_seq[ep_idx, 0]
                control_seq = extended_U_seq[ep_idx]
                true_states = np.vstack([initial_state, extended_Y_seq[ep_idx]])
                
                if seq_length >= self.n_deriv:
                    hist_s = extended_X_seq[ep_idx, :self.n_deriv].tolist()
                else:
                    hist_s = [initial_state.copy() for _ in range(self.n_deriv)]
                
                pred_states = self.predict(s0=initial_state, U_seq=control_seq, hist_s=hist_s)
                step_errors = np.linalg.norm(pred_states - true_states, axis=1)
                episode_errors.append(step_errors)
            
            exp_errors = np.mean(episode_errors, axis=0)
            all_errors.append(exp_errors)
            print(f"[KDeriv-CartPole] 实验 {exp_idx+1} 误差范围：{np.min(exp_errors):.6f} ~ {np.max(exp_errors):.6f}")

        mean_errors = np.mean(all_errors, axis=0)
        log10_errors = np.log10(mean_errors + 1e-10)

        if save_results:
            result_file = os.path.join(result_path, f"kderiv_cartpole_pred_results_K{K_steps}.npz")
            np.savez_compressed(
                result_file,
                mean_errors=mean_errors,
                log10_errors=log10_errors,
                K_steps=K_steps,
                seq_length=seq_length,
                num_experiments=num_experiments,
                all_errors=np.array(all_errors),
                n_deriv=self.n_deriv
            )
            print(f"\n[KDeriv-CartPole] 实验结果保存至：{result_file}")

        self._plot_prediction_errors(mean_errors, log10_errors, seq_length, K_steps)
        return mean_errors, log10_errors

    def _plot_prediction_errors(
        self,
        mean_errors: np.ndarray,
        log10_errors: np.ndarray,
        seq_length: int,
        K_steps: int
    ) -> None:
        """绘制CartPole预测误差曲线"""
        plt.figure(figsize=(12, 8))
        time_steps = np.arange(seq_length + 1)
        
        plt.subplot(2, 1, 1)
        plt.plot(time_steps, mean_errors, color='#2E86AB', linewidth=2.5, 
                 label=f'Mean Error (2*K={seq_length} steps)')
        plt.ylabel('Euclidean Distance Error (4D State)', fontsize=12)
        plt.title(f'KDerivative-Koopman CartPole Prediction Errors (K={K_steps}, n_deriv={self.n_deriv})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        plt.subplot(2, 1, 2)
        plt.plot(time_steps, log10_errors, color='#A23B72', linewidth=2.5, 
                 label=f'log10(Mean Error)')
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('log10(Euclidean Distance Error)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        os.makedirs("./fig/cartpole", exist_ok=True)
        plot_path = os.path.join("./fig/cartpole", f"kderiv_cartpole_pred_errors_K{K_steps}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[KDeriv-CartPole] 误差曲线保存至：{plot_path}")

    def test_cartpole_lqr(self, K_lqr: np.ndarray, s_des: np.ndarray, num_episodes: int = 100, 
                         max_steps: int = 500, seed: int = 2, version: str = "KDeriv-CartPole") -> List[float]:
        """测试CartPole的LQR控制性能"""
        self.env = gym.make("CartPole-v1")
        self.env.seed(seed)
        episode_scores = []
        all_trajectories = []
        final_states = []
        success_count = 0

        print(f"\n[Test LQR] 开始{num_episodes}回合CartPole测试...")
        for ep in range(num_episodes):
            s_prev = self.env.reset()
            hist_s = [s_prev.copy() for _ in range(self.n_deriv)]
            done = False
            total_reward = 0.0
            step = 0
            trajectory = [s_prev[2]]  # 记录杆角度

            while not done and step < max_steps:
                u_continuous = self.control_lqr(s_prev, s_des, K_lqr, hist_s)
                u_discrete = 1 if u_continuous >= 0 else 0  # 连续→离散动作映射

                s_next, reward, done, _ = self.env.step(u_discrete)
                hist_s.pop(0)
                hist_s.append(s_prev.copy())
                total_reward += reward
                trajectory.append(s_next[2])
                s_prev = s_next
                step += 1

            # 统计结果
            final_x, final_x_dot = s_prev[0], s_prev[1]
            final_theta, final_theta_dot = s_prev[2], s_prev[3]
            final_states.append((final_x, final_theta, final_theta_dot))
            all_trajectories.append(trajectory)
            episode_scores.append(total_reward)

            # 成功稳定条件
            if (abs(final_x) <= 0.5 and abs(final_theta) <= 0.174 and 
                abs(final_theta_dot) <= 0.5):
                success_count += 1
            print(f"[Test LQR] 回合 {ep+1:2d} | 得分：{total_reward:5.1f} | 步数：{step:3d} | "
                  f"最终状态：(x={final_x:.3f}, θ={final_theta:.3f}, θ'={final_theta_dot:.3f})")

        self.env.close()

        # 误差边界验证
        T = max_steps * self.dt
        error_bound = self.compute_error_bound(T)
        actual_theta_errors = [abs(s[1]) for s in final_states]
        print(f"\n[Test LQR] 误差边界验证：")
        print(f"  预测时域T={T:.2f}s | 理论误差边界={error_bound:.4f} | 实际最大角度误差={np.max(actual_theta_errors):.4f}")

        # 绘制轨迹图
        self._plot_cartpole_trajectory(all_trajectories, final_states, s_des, version)

        # 测试总结
        avg_reward = np.mean(episode_scores)
        std_reward = np.std(episode_scores)
        print(f"\n[Test LQR] 测试总结：")
        print(f"  平均得分：{avg_reward:.1f}±{std_reward:.1f} | 成功稳定率：{success_count/num_episodes*100:.1f}%")
        return episode_scores

    def control_lqr(self, s_curr: np.ndarray, s_des: np.ndarray, K_lqr: np.ndarray, 
                   hist_s: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """生成LQR控制信号"""
        if hist_s is None:
            psi_curr = np.concatenate([s_curr] + [np.zeros_like(s_curr) for _ in range(self.n_deriv)], axis=0)
            psi_des = np.concatenate([s_des] + [np.zeros_like(s_des) for _ in range(self.n_deriv)], axis=0)
        else:
            hist_s_arr = np.array(hist_s)
            derivs_curr = self.compute_derivatives(hist_s_arr[None, :, :].squeeze(0))
            derivs_des = [np.zeros_like(s_des) for _ in range(self.n_deriv)]
            psi_curr = np.concatenate([s_curr] + [d[0] for d in derivs_curr], axis=0)
            psi_des = np.concatenate([s_des] + derivs_des, axis=0)
        
        u0 = -K_lqr @ (psi_curr - psi_des)
        return u0

    def _plot_cartpole_trajectory(self, all_trajectories: List[List[float]], 
                                 final_states: List[Tuple[float, float, float]], s_des: np.ndarray, version: str) -> None:
        """绘制CartPole杆角度轨迹"""
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10.colors
        max_traj_len = max(len(traj) for traj in all_trajectories)
        time_steps = np.arange(max_traj_len) * self.dt

        for ep, traj in enumerate(all_trajectories):
            plt.plot(time_steps[:len(traj)], traj, color=colors[ep % len(colors)], 
                     alpha=0.7, linewidth=1, label=f"Episode {ep+1}" if ep < 5 else "")

        mean_theta = np.mean([s[1] for s in final_states])
        std_theta = np.std([s[1] for s in final_states], ddof=1)
        plt.axhline(y=s_des[2], color="red", linestyle="--", linewidth=2, label="Target Angle (0 rad)")
        plt.axhline(y=mean_theta, color="blue", linestyle="-", linewidth=2, 
                    label=f"Final Mean Angle: {mean_theta:.3f} rad")
        plt.fill_between(time_steps, mean_theta - std_theta, mean_theta + std_theta, 
                        color="blue", alpha=0.1, label=f"Angle Std Range (±{std_theta:.3f})")

        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Pole Angle (rad)", fontsize=12)
        plt.ylim(-np.pi/2, np.pi/2)
        plt.title(f"CartPole Pole Angle Trajectories ({version})", fontsize=14)
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(True, alpha=0.3)

        os.makedirs("./fig/cartpole", exist_ok=True)
        plt.savefig(f"./fig/cartpole/kderiv_cartpole_trajectory_{version}.png", bbox_inches="tight", dpi=300)
        plt.close()


def generate_cartpole_data(
    env_name: str = "CartPole-v1",
    num_episodes: int = 200,
    K_steps: int = 10,
    save_path: str = "./data/cartpole",
    seed: int = 2
) -> None:
    """生成CartPole训练数据"""
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    os.makedirs(save_path, exist_ok=True)

    X_list, U_list, Y_list = [], [], []
    X_seq_list, U_seq_list, Y_seq_list = [], [], []

    print(f"[DataGen] 生成CartPole训练数据：{num_episodes}回合，K_steps={K_steps}...")
    for ep in trange(num_episodes):
        state_prev = env.reset()
        ep_states = [state_prev]
        ep_controls = []
        done = False

        while not done:
            u = env.action_space.sample()
            state_next, _, done, _ = env.step(u)
            ep_states.append(state_next)
            ep_controls.append([u])
            if len(ep_controls) >= 500:
                break

        ep_states = np.array(ep_states, dtype=np.float32)
        ep_controls = np.array(ep_controls, dtype=np.float32)
        ep_len = len(ep_controls)

        if ep_len < K_steps:
            continue

        X_list.append(ep_states[:-1])
        Y_list.append(ep_states[1:])
        U_list.append(ep_controls)

        max_start = ep_len - K_steps
        for start in range(0, max_start, 5):
            X_seq = ep_states[start:start+K_steps]
            U_seq = ep_controls[start:start+K_steps]
            Y_seq = ep_states[start+1:start+K_steps+1]
            X_seq_list.append(X_seq)
            U_seq_list.append(U_seq)
            Y_seq_list.append(Y_seq)

    X = np.concatenate(X_list, axis=0)
    U = np.concatenate(U_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    X_seq = np.array(X_seq_list)
    U_seq = np.array(U_seq_list)
    Y_seq = np.array(Y_seq_list)

    save_file = os.path.join(save_path, f"train_data_{env_name}_n4_m1_K{K_steps}_seed{seed}.npz")
    np.savez_compressed(
        save_file,
        X=X, U=U, Y=Y,
        X_seq=X_seq, U_seq=U_seq, Y_seq=Y_seq,
        K_steps=K_steps, seed=seed
    )
    print(f"[DataGen] 数据保存至：{save_file}")
    env.close()


def main():
    """CartPole实验主流程（修复参数传递错误）"""
    # 1. 配置参数
    config = {
        "seed": 2,
        "num_test_episodes": 100,
        "n_deriv": 2,               # 导数阶数
        "dt": 0.02,                 # CartPole采样间隔
        "Np": 30,
        "Q_weight": np.diag([1.0, 0.1, 10.0, 0.1]),  # 4维状态权重
        "R_weight": np.diag([0.1]),
        "K_steps": 15,
        "train_data_path": "./data/cartpole/train_data_CartPole-v1_n4_m1_deriv2_K15_seed2.npz",
        "extended_test_path": "./data/cartpole/test_data_CartPole-v1_n4_m1_K15_seed2_extended.npz",
        "result_path": "./results/cartpole"
    }

    # 2. 初始化模型（关键修复：正确传递参数，n=4是状态维度，n_deriv是导数阶数）
    kderiv = KDerivativeKoopmanCartPole(
        n=4,                # 状态维度（CartPole是4维）
        m=1,                # 动作维度
        n_deriv=config["n_deriv"],  # 导数阶数
        dt=config["dt"],
        Np=config["Np"]
    )
    
    train_data = np.load(config["train_data_path"])
    kderiv.X = train_data["X"]
    kderiv.U = train_data["U"]
    kderiv.Y = train_data["Y"]
    
    # 计算导数
    kderiv.dX = kderiv.compute_derivatives(kderiv.X)
    kderiv.estimate_f_max_deriv()
    print(f"[Main] 加载训练数据：X={kderiv.X.shape}, U={kderiv.U.shape}, Y={kderiv.Y.shape}")

    # 4. 求解Koopman算子
    kderiv.solve_koopman()

    # 5. 计算LQR增益
    K_lqr = kderiv.compute_lqr_gain(
        Q=config["Q_weight"],
        R=config["R_weight"]
    )

    # 6. LQR控制测试
    s_des = np.array([0.0, 0.0, 0.0, 0.0])  # 目标状态
    kderiv.test_cartpole_lqr(
        K_lqr=K_lqr,
        s_des=s_des,
        num_episodes=config["num_test_episodes"],
        seed=config["seed"],
        version=f"KDeriv-n{config['n_deriv']}"
    )

    # 7. 轨迹预测评估
    if os.path.exists(config["extended_test_path"]):
        mean_errors, log10_errors = kderiv.evaluate_trajectory_prediction(
            test_data_path=config["extended_test_path"],
            num_experiments=4,
            result_path=config["result_path"]
        )

        print("\n[KDeriv-CartPole] 轨迹预测评估总结")
        print(f"  预测长度：{len(mean_errors)-1} 步")
        print(f"  平均log10误差：{np.mean(log10_errors):.4f}")
    else:
        print(f"\n[警告] 扩展测试数据不存在：{config['extended_test_path']}")
        print("请先运行CartPole数据生成脚本生成测试数据")


if __name__ == "__main__":
    main()
