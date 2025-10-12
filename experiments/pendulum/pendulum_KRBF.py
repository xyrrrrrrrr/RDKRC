import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from typing import List, Tuple

# 导入自定义模块（假设已有适用于单摆的数据工具和KRBF模型）
from rdkrc.models.KRBF import KRBFModel


def test_krbf_pendulum_lqr(
    krbf: KRBFModel,
    K_lqr: np.ndarray,
    x_star: np.ndarray,
    num_episodes: int = 10,
    max_steps: int = 200,
    seed: int = 2,
    version: str = "KRBF-LQR"
) -> List[float]:
    """
    KRBF+LQR单摆控制测试
    状态空间: [theta, theta_dot, cos(theta)]（实际取决于环境输出，通常为3维）
    目标: 控制单摆稳定在竖直向上位置（theta=0）
    """
    env = gym.make("Pendulum-v0")
    env.seed(seed)
    episode_scores: List[float] = []
    all_trajectories: List[List[Tuple[float, float]]] = []  # theta-theta_dot轨迹
    final_states: List[Tuple[float, float]] = []            # 最终状态
    success_count = 0  # 成功稳定计数（theta∈[-0.1,0.1], theta_dot∈[-0.2,0.2]）

    print(f"\n[Test LQR] 开始{num_episodes}回合KRBF+LQR单摆测试...")
    for ep in range(num_episodes):
        x_prev = env.reset()  # 单摆环境返回(状态, 空字典)
        done = False
        total_score = 0.0
        step = 0
        trajectory = []  # 当前episode轨迹（角度-角速度）

        while not done and step < max_steps:
            # 记录角度-角速度轨迹
            theta = x_prev[0]
            theta_dot = x_prev[1]
            trajectory.append((theta, theta_dot))

            # 1. KRBF提升：z = ψ(x) - ψ(x*)
            z_prev = krbf._psi(x_prev) - krbf._psi(x_star)

            # 2. LQR控制计算：u = -K_lqr · z
            u_t = -K_lqr @ z_prev
            u_t = np.clip(u_t, env.action_space.low, env.action_space.high)

            # 3. 环境交互
            x_next, reward, done, _ = env.step(u_t)  # 单摆环境step返回5元组
            total_score += reward
            x_prev = x_next
            step += 1

        # 记录结果
        final_theta, final_theta_dot = x_prev[0], x_prev[1]
        final_states.append((final_theta, final_theta_dot))
        trajectory.append((final_theta, final_theta_dot))
        all_trajectories.append(trajectory)
        episode_scores.append(total_score)

        # 成功稳定判断
        if abs(final_theta) <= 0.1 and abs(final_theta_dot) <= 0.2:
            success_count += 1
        print(f"[Test LQR] 回合 {ep+1:2d} | 得分：{total_score:5.1f} | 步数：{step:3d} | "
              f"最终状态：(θ={final_theta:.3f}, θ'={final_theta_dot:.3f})")

    env.close()

    # -------------------------- 最终状态统计 --------------------------
    final_thetas = np.array([s[0] for s in final_states])
    final_thetas_dot = np.array([s[1] for s in final_states])
    mean_theta, mean_theta_dot = np.mean(final_thetas), np.mean(final_thetas_dot)
    std_theta, std_theta_dot = np.std(final_thetas, ddof=1), np.std(final_thetas_dot, ddof=1)

    print(f"\n[Test LQR] === 最终状态统计 ===")
    print(f"目标状态：(θ={x_star[0]:.3f}, θ'={x_star[1]:.3f})")
    print(f"均值状态：(θ={mean_theta:.3f}, θ'={mean_theta_dot:.3f})")
    print(f"标准差：std_θ={std_theta:.3f}, std_θ'={std_theta_dot:.3f}")
    print(f"成功稳定：{success_count}/{num_episodes} 次")

    # -------------------------- 轨迹绘图 --------------------------
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10.colors

    # 绘制所有轨迹
    for ep, traj in enumerate(all_trajectories):
        theta_coords = [p[0] for p in traj]
        theta_dot_coords = [p[1] for p in traj]
        plt.plot(theta_coords, theta_dot_coords, color=colors[ep % len(colors)], alpha=0.7,
                 label=f"Episode {ep+1}" if ep < 5 else "")  # 只显示前5条轨迹标签

    # 标注目标/均值/稳定区
    plt.scatter(x_star[0], x_star[1], color="red", marker="s", s=80, label="Target State")
    plt.scatter(mean_theta, mean_theta_dot, color="blue", marker="o", s=100, 
                label=f"Final Mean (θ={mean_theta:.3f}, θ'={mean_theta_dot:.3f})")
    plt.gca().add_patch(plt.Rectangle(
        (mean_theta - std_theta, mean_theta_dot - std_theta_dot), 
        2*std_theta, 2*std_theta_dot,
        color="blue", alpha=0.2, linestyle="--", label="State Std Range (±1σ)"
    ))
    # 绘制稳定区域
    plt.gca().add_patch(plt.Rectangle(
        (-0.1, -0.2), 0.2, 0.4,
        color="green", alpha=0.1, linestyle="-", label="Success Region"
    ))

    # 图表配置
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-8, 8)
    plt.xlabel("Angle θ (rad)", fontsize=12)
    plt.ylabel("Angular Velocity θ' (rad/s)", fontsize=12)
    plt.title(f"Pendulum Trajectory Summary ({version})", fontsize=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1), fontsize=10)
    plt.grid(True, alpha=0.5)

    # 保存图片
    os.makedirs("./fig/pendulum", exist_ok=True)
    plt.savefig(f"./fig/pendulum/krbf_pendulum_trajectory_{version}.png", bbox_inches="tight", dpi=300)
    plt.close()

    # -------------------------- 测试总结 --------------------------
    avg_score = np.mean(episode_scores)
    std_score = np.std(episode_scores)
    print(f"\n[Test LQR] === 测试总结 ===")
    print(f"平均得分：{avg_score:.1f}±{std_score:.1f} | 成功稳定率：{success_count/num_episodes*100:.1f}%")

    return episode_scores


def predict_trajectory(
    krbf: KRBFModel,
    initial_state: np.ndarray,
    control_sequence: np.ndarray,
    horizon: int
) -> np.ndarray:
    """使用KRBF模型预测单摆轨迹"""
    predicted_states = krbf.predict(
        x0=initial_state,
        U_seq=control_sequence.T)
    return predicted_states.T


def evaluate_trajectory_prediction(
    krbf: KRBFModel,
    test_data_path: str,
    num_experiments: int = 4,
    save_results: bool = True,
    result_path: str = "./results/pendulum",
) -> Tuple[np.ndarray, np.ndarray]:
    """评估单摆轨迹预测性能"""
    os.makedirs(result_path, exist_ok=True)
    
    # 加载测试数据
    test_data = np.load(test_data_path)
    extended_X_seq = test_data['extended_X_seq']  # [num_episodes, 2*K_steps, n]
    extended_U_seq = test_data['extended_U_seq']  # [num_episodes, 2*K_steps, m]
    extended_Y_seq = test_data['extended_Y_seq']  # [num_episodes, 2*K_steps, n]
    K_steps = 15
    seq_length = extended_X_seq.shape[1]
    
    print(f"\n[轨迹预测评估] 开始{num_experiments}次实验，预测长度={seq_length}步")
    print(f"[轨迹预测评估] 测试数据规模：{extended_X_seq.shape[0]}个序列")
    
    all_errors = []
    for exp_idx in range(num_experiments):
        print(f"\n[实验 {exp_idx+1}/{num_experiments}]")
        episode_errors = []
        
        for ep_idx in trange(extended_X_seq.shape[0], desc="处理测试序列"):
            initial_state = extended_X_seq[ep_idx, 0]
            control_sequence = extended_U_seq[ep_idx]
            
            # 真实状态序列
            true_states = np.vstack((initial_state, extended_Y_seq[ep_idx]))
            # 预测轨迹
            predicted_states = predict_trajectory(
                krbf=krbf,
                initial_state=initial_state,
                control_sequence=control_sequence,
                horizon=seq_length
            )
            
            # 计算每一步的预测误差（欧氏距离）
            step_errors = np.linalg.norm(predicted_states - true_states, axis=1)
            episode_errors.append(step_errors)
        
        exp_errors = np.mean(episode_errors, axis=0)
        all_errors.append(exp_errors)
        print(f"[实验 {exp_idx+1}] 平均误差范围: {np.min(exp_errors):.6f} - {np.max(exp_errors):.6f}")
    
    # 计算统计量
    mean_errors = np.mean(all_errors, axis=0)
    log10_errors = np.log10(mean_errors + 1e-10)  # 避免log(0)
    
    # 保存结果
    if save_results:
        result_file = os.path.join(result_path, f"krbf_pendulum_pred_results_K{K_steps}.npz")
        np.savez_compressed(
            result_file,
            mean_errors=mean_errors,
            log10_errors=log10_errors,
            K_steps=K_steps,
            seq_length=seq_length,
            num_experiments=num_experiments,
            all_errors=all_errors
        )
        print(f"\n[轨迹预测评估] 结果保存至：{result_file}")
    
    # 绘制误差曲线
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(seq_length + 1)
    
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, mean_errors, 'b-', linewidth=2)
    plt.ylabel('Mean Prediction Error')
    plt.title(f'Pendulum Trajectory Prediction Errors (2*K={seq_length} steps)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, log10_errors, 'r-', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('log10(Mean Prediction Error)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    error_plot_path = os.path.join("./fig/pendulum", f"prediction_errors_K{K_steps}.png")
    plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[轨迹预测评估] 误差曲线图保存至：{error_plot_path}")
    
    return mean_errors, log10_errors


def main():
    """单摆KRBF完整实验流程"""
    # -------------------------- 1. 配置参数 --------------------------
    config = {
        "seed": 2,
        "num_data_episodes": 200,
        "num_test_episodes": 100,
        "N_rbf": 10,
        "Np_mpc": 50,
        "K_steps_data": 15,
        "data_load_path": "./data/pendulum/train_data_Pendulum-v0_n3_m1_deriv2_K15_seed2.npz",
        "extended_test_path": "./data/pendulum/test_data_Pendulum-v0_n3_m1_K15_seed2_extended.npz",
        "koopman_save_path": "./data/pendulum/krbf_pendulum_koopman_matrix.npz",
        "num_prediction_experiments": 4
    }
    
    # 加载训练数据
    data = np.load(config["data_load_path"])
    x_seq = data['X_seq']
    u_seq = data['U_seq']
    x_next_seq = data['Y_seq']
    X_single = data['X']
    U_single = data['U']
    Y_single = data['Y']

    # -------------------------- 2. KRBF模型初始化与训练 --------------------------
    krbf = KRBFModel(
        n=3,  # 单摆状态维度为3
        m=1,  # 单摆动作维度为1
        N_rbf=config["N_rbf"],
        Np=config["Np_mpc"],
        state_low=[-1.0, -1.0, -8.0],
        state_high=[1.0, 1.0, 8.0],
        action_low=[-2.0],
        action_high=[2.0]
    )
    krbf.set_data(X_single, U_single, Y_single)
    krbf.solve_koopman()

    # -------------------------- 3. LQR增益计算 --------------------------
    Q_lqr = np.diag([1.0, 1.0, 1.0])  # 角度权重最高
    R_lqr = np.array([[0.1]])          # 控制权重
    K_lqr = krbf.compute_lqr_gain(Q_lqr, R_lqr)

    # -------------------------- 4. 目标状态与LQR测试 --------------------------
    x_star = np.array([1.0, 0.0, 0.0])  # 单摆竖直向上状态（theta=0, theta_dot=0, cos(theta)=1）
    test_krbf_pendulum_lqr(
        krbf=krbf,
        K_lqr=K_lqr,
        x_star=x_star,
        num_episodes=config["num_test_episodes"],
        seed=config["seed"],
        version="KRBF-LQR_v0"
    )
    
    # -------------------------- 5. 轨迹预测评估 --------------------------
    if os.path.exists(config["extended_test_path"]):
        mean_errors, log10_errors = evaluate_trajectory_prediction(
            krbf=krbf,
            test_data_path=config["extended_test_path"],
            num_experiments=config["num_prediction_experiments"],
            save_results=True
        )
        
        print("\n[轨迹预测评估总结]")
        print(f"预测长度: {len(mean_errors)-1} 步 (2*K_steps)")
        print(f"平均log10误差: {np.mean(log10_errors):.4f}")
        print(f"各时间步log10误差均值:")
        for i in range(0, len(log10_errors), 5):
            print(f"  第{i}步: {log10_errors[i]:.4f}")
    else:
        print(f"\n警告：未找到扩展测试数据，请先运行数据生成脚本生成 {config['extended_test_path']}")


if __name__ == "__main__":
    main()