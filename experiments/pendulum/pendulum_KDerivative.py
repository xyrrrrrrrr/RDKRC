import gym
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from typing import Optional, List, Tuple
from scipy.linalg import pinv, solve_discrete_are
from scipy.signal import savgol_filter


class KDerivativeKoopmanPendulum:
    """KDerivative-Koopman算法类（适配pendulum环境）"""
    def __init__(self, n: int = 3, m: int = 1, n_deriv: int = 1, 
                 dt: float = 0.05, Np: int = 30):
        """初始化参数（适配pendulum状态维度：3维状态，1维动作）"""
        self.n = n  # pendulum状态维度：[theta, theta_dot, cos(theta), sin(theta)]实际有效3维
        self.m = m  # 动作维度：1维力矩
        self.n_deriv = n_deriv
        self.dt = dt  # pendulum默认采样间隔
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
        
        # 基函数参数
        self.w_s: int = n * (n_deriv + 1)
        self.w: int = self.w_s + m
        self.f_max_deriv: Optional[float] = None
        self.env: Optional[gym.Env] = None

    def compute_derivatives(self, X_seq: np.ndarray, filter_window: int = 5) -> List[np.ndarray]:
        """计算状态高阶导数（适配pendulum状态特性）"""
        ep_len, n = X_seq.shape
        dX = []
        current_X = X_seq.copy()
        filter_window = filter_window if filter_window % 2 == 1 else filter_window + 1
        
        for order in range(1, self.n_deriv + 1):
            dx = np.zeros_like(current_X)
            for i in range(1, ep_len - 1):
                dx[i] = (current_X[i+1] - current_X[i-1]) / (2 * self.dt)
            dx[0] = (current_X[1] - current_X[0]) / self.dt
            dx[-1] = (current_X[-1] - current_X[-2]) / self.dt
            
            if filter_window > 1 and ep_len > filter_window:
                dx = savgol_filter(dx, window_length=filter_window, polyorder=2, axis=0)
            
            dX.append(dx)
            current_X = dx
        return dX

    def estimate_f_max_deriv(self) -> None:
        """估计高阶导数最大值"""
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
        print(f"[KDeriv-Pendulum] 估计高阶导数最大值：{self.f_max_deriv:.4f}")

    def _psi(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        """构造基函数（适配pendulum状态）"""
        psi_s = [s]
        if self.dX is not None and hasattr(self, '_current_sample_idx'):
            k = self._current_sample_idx
            for i in range(self.n_deriv):
                psi_s.append(self.dX[i][k] if len(self.dX[i]) > k else np.zeros_like(s))
        else:
            for i in range(self.n_deriv):
                psi_s.append(np.zeros_like(s))
        
        psi_s = np.concatenate(psi_s, axis=0)
        psi = np.concatenate([psi_s, u], axis=0)
        return psi

    def solve_koopman(self) -> None:
        """求解Koopman算子"""
        if self.X is None or self.U is None or self.Y is None:
            raise ValueError("请先生成训练数据")
        K = self.X.shape[0]
        w = self.w
        
        Psi_k = np.zeros((K, w))
        Psi_k1 = np.zeros((K, w))
        print(f"[KDeriv-Pendulum] 构造基函数矩阵（K={K}, w={w}）...")
        
        for k in trange(K):
            self._current_sample_idx = k
            Psi_k[k] = self._psi(self.X[k], self.U[k])
            Psi_k1[k] = self._psi(self.Y[k], np.zeros(self.m))
        
        reg_eps = 1e-5
        A = (Psi_k1.T @ Psi_k) / K
        G = (Psi_k.T @ Psi_k) / K + reg_eps * np.eye(w)
        self.Kd = A @ pinv(G)
        
        self.A = self.Kd[:self.w_s, :self.w_s]
        self.B = self.Kd[:self.w_s, self.w_s:]
        self.C = np.hstack([np.eye(self.n), np.zeros((self.n, self.w - self.n))])
        
        spectral_radius_A = np.max(np.abs(np.linalg.eigvals(self.A)))
        print(f"[KDeriv-Pendulum] Koopman求解完成：A谱半径={spectral_radius_A:.4f}")

    def compute_error_bound(self, T: float) -> float:
        """计算预测误差边界"""
        if self.f_max_deriv is None:
            raise ValueError("请先调用estimate_f_max_deriv")
        return (T ** (self.n_deriv + 1)) / math.factorial(self.n_deriv + 1) * self.f_max_deriv

    def compute_lqr_gain(self, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """求解LQR增益（适配pendulum控制目标）"""
        if self.A is None or self.B is None:
            raise ValueError("请先调用solve_koopman")
        
        Q_lift = self.C[:self.w_s, :self.w_s].T @ Q @ self.C[:self.w_s, :self.w_s]
        R_lift = R.copy()
        P = solve_discrete_are(self.A, self.B, Q_lift, R_lift)
        K_lqr = pinv(self.B.T @ P @ self.B + R_lift) @ self.B.T @ P @ self.A
        print(f"[KDeriv-Pendulum] LQR增益维度：{K_lqr.shape}")
        return K_lqr

    def predict(self, s0: np.ndarray, U_seq: np.ndarray, hist_s: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """预测轨迹（适配pendulum）"""
        if self.A is None or self.B is None or self.C is None:
            raise ValueError("请先调用solve_koopman")
        seq_length = U_seq.shape[0]
        S_pred = np.zeros((seq_length + 1, self.n))
        S_pred[0] = s0
        
        if hist_s is not None and len(hist_s) >= self.n_deriv:
            hist_s_arr = np.array(hist_s)
            derivs = self.compute_derivatives(hist_s_arr[None, :, :].squeeze(0))
            psi_s0 = np.concatenate([s0] + [d[0] for d in derivs], axis=0)
        else:
            psi_s0 = np.concatenate([s0] + [np.zeros_like(s0) for _ in range(self.n_deriv)], axis=0)
        
        psi_curr = psi_s0
        for t in range(seq_length):
            psi_next = self.A @ psi_curr + self.B @ U_seq[t]
            S_pred[t+1] = self.C @ np.concatenate([psi_next, np.zeros(self.m)], axis=0)
            psi_curr = psi_next
        
        return S_pred

    def evaluate_trajectory_prediction(
        self,
        test_data_path: str,
        num_experiments: int = 4,
        save_results: bool = True,
        result_path: str = "./results/pendulum"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """评估pendulum轨迹预测性能"""
        os.makedirs(result_path, exist_ok=True)
        test_data = np.load(test_data_path)
        
        extended_X_seq = test_data['extended_X_seq']
        extended_U_seq = test_data['extended_U_seq']
        extended_Y_seq = test_data['extended_Y_seq']
        K_steps = 15
        seq_length = extended_X_seq.shape[1]
        num_test_ep = extended_X_seq.shape[0]
        
        print(f"\n[KDeriv-Pendulum] 开始{num_experiments}次轨迹预测实验（长度={seq_length}=2*{K_steps}步）")
        print(f"[KDeriv-Pendulum] 测试数据规模：{num_test_ep}个序列")
        
        all_errors = []
        for exp_idx in range(num_experiments):
            print(f"\n[KDeriv-Pendulum] 实验 {exp_idx+1}/{num_experiments}")
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
            print(f"[KDeriv-Pendulum] 实验 {exp_idx+1} 误差范围：{np.min(exp_errors):.6f} ~ {np.max(exp_errors):.6f}")

        mean_errors = np.mean(all_errors, axis=0)
        log10_errors = np.log10(mean_errors + 1e-10)

        if save_results:
            result_file = os.path.join(result_path, f"kderiv_pendulum_pred_results_K{K_steps}.npz")
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
            print(f"\n[KDeriv-Pendulum] 实验结果保存至：{result_file}")

        self._plot_prediction_errors(mean_errors, log10_errors, seq_length, K_steps)
        return mean_errors, log10_errors

    def _plot_prediction_errors(
        self,
        mean_errors: np.ndarray,
        log10_errors: np.ndarray,
        seq_length: int,
        K_steps: int
    ) -> None:
        """绘制pendulum预测误差曲线"""
        plt.figure(figsize=(12, 8))
        time_steps = np.arange(seq_length + 1)
        
        plt.subplot(2, 1, 1)
        plt.plot(time_steps, mean_errors, color='#2E86AB', linewidth=2.5, label=f'Mean Error (2*K={seq_length} steps)')
        plt.ylabel('Euclidean Distance Error', fontsize=12)
        plt.title(f'KDerivative-Koopman Pendulum Prediction Errors (K={K_steps}, n_deriv={self.n_deriv})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        plt.subplot(2, 1, 2)
        plt.plot(time_steps, log10_errors, color='#A23B72', linewidth=2.5, label=f'log10(Mean Error)')
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('log10(Euclidean Distance Error)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        os.makedirs("./fig/pendulum", exist_ok=True)
        plot_path = os.path.join("./fig/pendulum", f"kderiv_pendulum_pred_errors_K{K_steps}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[KDeriv-Pendulum] 误差曲线保存至：{plot_path}")

    def test_pendulum_lqr(self, K_lqr: np.ndarray, s_des: np.ndarray, num_episodes: int = 100, 
                         max_steps: int = 200, seed: int = 2, version: str = "KDeriv-Pendulum") -> List[float]:
        """测试pendulum的LQR控制性能"""
        self.env = gym.make("Pendulum-v0")
        self.env.seed(seed)
        episode_scores = []
        all_trajectories = []
        final_states = []
        success_count = 0

        print(f"\n[Test LQR] 开始{num_episodes}回合pendulum测试...")
        for ep in range(num_episodes):
            s_prev = self.env.reset()
            hist_s = [s_prev.copy() for _ in range(self.n_deriv)]
            done = False
            total_reward = 0.0
            step = 0
            trajectory = [s_prev[0]]  # 记录角度轨迹

            while not done and step < max_steps:
                u0 = self.control_lqr(s_prev, s_des, K_lqr, hist_s)
                s_next, reward, done, _ = self.env.step(u0)
                # 更新历史状态
                hist_s.pop(0)
                hist_s.append(s_prev.copy())
                # 记录数据
                total_reward += reward
                trajectory.append(s_next[0])
                s_prev = s_next
                step += 1

            # 统计结果
            final_theta = s_prev[0]
            final_states.append(final_theta)
            all_trajectories.append(trajectory)
            episode_scores.append(total_reward)

            # 成功标准：角度接近0（垂直向上）
            if abs(final_theta) <= 0.1:
                success_count += 1
            print(f"[Test LQR] 回合 {ep+1:2d} | 奖励：{total_reward:5.1f} | 步数：{step:3d} | 最终角度：{final_theta:.3f}")

        self.env.close()

        # 误差边界验证
        T = max_steps * self.dt
        error_bound = self.compute_error_bound(T)
        actual_errors = [abs(theta) for theta in final_states]
        print(f"\n[Test LQR] 误差边界验证：")
        print(f"  预测时域T={T:.2f}s | 理论误差边界={error_bound:.4f} | 实际最大误差={np.max(actual_errors):.4f}")

        # 绘制轨迹图
        self._plot_pendulum_trajectory(all_trajectories, final_states, s_des, version)

        # 测试总结
        avg_reward = np.mean(episode_scores)
        std_reward = np.std(episode_scores)
        print(f"\n[Test LQR] 测试总结：")
        print(f"  平均奖励：{avg_reward:.1f}±{std_reward:.1f} | 成功平衡：{success_count}/{num_episodes}")
        return episode_scores

    def control_lqr(self, s_curr: np.ndarray, s_des: np.ndarray, K_lqr: np.ndarray, 
                   hist_s: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """生成LQR控制信号（适配pendulum动作范围）"""
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
        # Pendulum动作范围：[-2, 2]
        if self.env is not None:
            u0 = np.clip(u0, self.env.action_space.low, self.env.action_space.high)
        return u0

    def _plot_pendulum_trajectory(self, all_trajectories: List[List[float]], 
                                 final_states: List[float], s_des: np.ndarray, version: str) -> None:
        """绘制pendulum角度轨迹"""
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10.colors

        # 绘制所有轨迹
        for ep, traj in enumerate(all_trajectories):
            time_steps = np.arange(len(traj)) * self.dt
            plt.plot(time_steps, traj, color=colors[ep % len(colors)], alpha=0.7, linewidth=1)

        # 标注目标与统计信息
        mean_theta = np.mean(final_states)
        std_theta = np.std(final_states, ddof=1)
        plt.axhline(y=s_des[0], color="red", linestyle="--", linewidth=2, label="Target Angle (0 rad)")
        plt.axhline(y=mean_theta, color="blue", linestyle="-", linewidth=2, label=f"Final Mean: {mean_theta:.3f} rad")

        # 图表配置
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Pendulum Angle (rad)", fontsize=12)
        plt.ylim(-np.pi, np.pi)
        plt.title(f"Pendulum Angle Trajectories ({version})", fontsize=14)
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(True, alpha=0.3)

        # 保存图片
        os.makedirs("./fig/pendulum", exist_ok=True)
        plt.savefig(f"./fig/pendulum/kderiv_pendulum_trajectory_{version}.png", bbox_inches="tight", dpi=300)
        plt.close()


def generate_pendulum_data(
    env_name: str = "Pendulum-v0",
    num_episodes: int = 200,
    K_steps: int = 10,
    save_path: str = "./data/pendulum",
    seed: int = 2
) -> None:
    """生成pendulum训练数据"""
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    os.makedirs(save_path, exist_ok=True)

    X_list, U_list, Y_list = [], [], []
    X_seq_list, U_seq_list, Y_seq_list = [], [], []

    print(f"[DataGen] 生成pendulum训练数据：{num_episodes}回合，K_steps={K_steps}...")
    for ep in trange(num_episodes):
        state_prev = env.reset()
        ep_states = [state_prev]
        ep_controls = []
        done = False

        while not done:
            u = env.action_space.sample()
            state_next, _, done, _ = env.step(u)
            ep_states.append(state_next)
            ep_controls.append(u)
            if len(ep_controls) >= 500:  # 限制最大步数
                break

        ep_states = np.array(ep_states, dtype=np.float32)
        ep_controls = np.array(ep_controls, dtype=np.float32)
        ep_len = len(ep_controls)

        if ep_len < K_steps:
            continue

        # 收集单步数据
        X_list.append(ep_states[:-1])
        Y_list.append(ep_states[1:])
        U_list.append(ep_controls)

        # 收集K步序列数据
        max_start = ep_len - K_steps
        for start in range(0, max_start, 5):  # 间隔采样减少冗余
            X_seq = ep_states[start:start+K_steps]
            U_seq = ep_controls[start:start+K_steps]
            Y_seq = ep_states[start+1:start+K_steps+1]
            X_seq_list.append(X_seq)
            U_seq_list.append(U_seq)
            Y_seq_list.append(Y_seq)

    # 保存数据
    X = np.concatenate(X_list, axis=0)
    U = np.concatenate(U_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    X_seq = np.array(X_seq_list)
    U_seq = np.array(U_seq_list)
    Y_seq = np.array(Y_seq_list)

    save_file = os.path.join(save_path, f"train_data_{env_name}_n3_m1_K{K_steps}_seed{seed}.npz")
    np.savez_compressed(
        save_file,
        X=X, U=U, Y=Y,
        X_seq=X_seq, U_seq=U_seq, Y_seq=Y_seq,
        K_steps=K_steps, seed=seed
    )
    print(f"[DataGen] 数据保存至：{save_file}")
    env.close()


def main():
    """pendulum实验主流程"""
    # 1. 配置参数
    config = {
        "seed": 2,
        "num_test_episodes": 50,
        "n_deriv": 2,
        "dt": 0.05,  # Pendulum默认采样间隔
        "Np": 30,
        "Q_weight": np.diag([1.0, 1.0, 1.0]),  # 角度权重更高
        "R_weight": np.diag([0.1]),
        "K_steps": 10,
        "train_data_path": "./data/pendulum/train_data_Pendulum-v0_n3_m1_deriv2_K15_seed2.npz",
        "extended_test_path": "./data/pendulum/test_data_Pendulum-v0_n3_m1_K15_seed2_extended.npz",
        "result_path": "./results/pendulum"
    }


    # 3. 初始化模型
    kderiv = KDerivativeKoopmanPendulum(
        n=3, m=1,
        n_deriv=config["n_deriv"],
        dt=config["dt"],
        Np=config["Np"]
    )

    # 4. 加载训练数据
    train_data = np.load(config["train_data_path"])
    kderiv.X = train_data["X"]
    kderiv.U = train_data["U"]
    kderiv.Y = train_data["Y"]
    # 计算导数
    kderiv.dX = kderiv.compute_derivatives(kderiv.X)
    kderiv.estimate_f_max_deriv()
    print(f"[Main] 加载训练数据：X={kderiv.X.shape}, U={kderiv.U.shape}")

    # 5. 求解Koopman算子
    kderiv.solve_koopman()

    # 6. LQR控制测试
    K_lqr = kderiv.compute_lqr_gain(
        Q=config["Q_weight"],
        R=config["R_weight"]
    )
    s_des = np.array([0.0, 0.0, 1.0])  # 目标状态：垂直向上（theta=0, theta_dot=0, cos(theta)=1）
    kderiv.test_pendulum_lqr(
        K_lqr=K_lqr,
        s_des=s_des,
        num_episodes=config["num_test_episodes"],
        seed=config["seed"],
        version=f"KDeriv-n{config['n_deriv']}"
    )

    # 7. 轨迹预测评估（需先生成扩展测试数据）
    if os.path.exists(config["extended_test_path"]):
        mean_errors, log10_errors = kderiv.evaluate_trajectory_prediction(
            test_data_path=config["extended_test_path"],
            num_experiments=4,
            result_path=config["result_path"]
        )

        print("\n[KDeriv-Pendulum] 轨迹预测评估总结")
        print(f"  预测长度：{len(mean_errors)-1} 步")
        print(f"  平均log10误差：{np.mean(log10_errors):.4f}")
    else:
        print(f"\n[警告] 扩展测试数据不存在：{config['extended_test_path']}")


if __name__ == "__main__":
    main()