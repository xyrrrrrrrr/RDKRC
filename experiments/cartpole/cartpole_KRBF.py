import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from typing import List, Tuple

# 导入自定义模块（假设已有适用于CartPole的数据工具和KRBF模型）
from rdkrc.models.KRBF import KRBFModel


def test_krbf_cartpole_lqr(
    krbf: KRBFModel,
    K_lqr: np.ndarray,
    x_star: np.ndarray,
    num_episodes: int = 10,
    max_steps: int = 500,  # CartPole-v1默认最大步数
    seed: int = 2,
    version: str = "KRBF-LQR"
) -> List[float]:
    """
    KRBF+LQR CartPole控制测试
    状态空间: [小车位置x, 小车速度x_dot, 杆角度theta, 杆角速度theta_dot]（4维）
    目标: 控制CartPole稳定在「小车居中、杆竖直」状态（x=0, theta=0, 速度均为0）
    """
    env = gym.make("CartPole-v1")
    env.seed(seed)
    episode_scores: List[float] = []
    all_trajectories: List[List[Tuple[float, float]]] = []  # 杆角度-角速度轨迹
    final_states: List[Tuple[float, float, float, float]] = []  # 完整最终状态
    success_count = 0  # 成功稳定计数：x∈[-0.5,0.5], theta∈[-10°=0.174rad,10°], 速度∈[-0.5,0.5]

    print(f"\n[Test LQR] 开始{num_episodes}回合KRBF+LQR CartPole测试...")
    for ep in range(num_episodes):
        x_prev = env.reset()  # CartPole重置返回4维状态：[x, x_dot, theta, theta_dot]
        done = False
        total_score = 0.0
        step = 0
        trajectory = []  # 当前episode轨迹（杆角度-角速度）

        while not done and step < max_steps:
            # 记录杆角度-角速度轨迹（CartPole状态索引：theta=2, theta_dot=3）
            theta = x_prev[2]
            theta_dot = x_prev[3]
            trajectory.append((theta, theta_dot))

            # 1. KRBF提升：z = ψ(x) - ψ(x*)（适配4维状态）
            z_prev = krbf._psi(x_prev) - krbf._psi(x_star)

            # 2. LQR控制计算：连续输出映射为CartPole离散动作（0=左推，1=右推）
            u_t = -K_lqr @ z_prev  # LQR连续输出
            u_action = 1 if u_t >= 0 else 0  # 离散动作映射：正→右推，负→左推

            # 3. 环境交互（CartPole输入离散动作）
            x_next, reward, done, _ = env.step(u_action)
            total_score += reward
            x_prev = x_next
            step += 1

        # 记录结果（CartPole完整状态：x=0, x_dot=1, theta=2, theta_dot=3）
        final_x, final_x_dot = x_prev[0], x_prev[1]
        final_theta, final_theta_dot = x_prev[2], x_prev[3]
        final_states.append((final_x, final_x_dot, final_theta, final_theta_dot))
        trajectory.append((final_theta, final_theta_dot))
        all_trajectories.append(trajectory)
        episode_scores.append(total_score)

        # 成功稳定判断（CartPole专属条件）
        if (abs(final_x) <= 0.5 and abs(final_x_dot) <= 0.5 and
            abs(final_theta) <= 0.174 and abs(final_theta_dot) <= 0.5):
            success_count += 1
        print(f"[Test LQR] 回合 {ep+1:2d} | 得分：{total_score:5.1f} | 步数：{step:3d} | "
              f"最终状态：(x={final_x:.3f}, θ={final_theta:.3f}, θ'={final_theta_dot:.3f})")

    env.close()

    # -------------------------- 最终状态统计 --------------------------
    final_xs = np.array([s[0] for s in final_states])
    final_thetas = np.array([s[2] for s in final_states])
    final_thetas_dot = np.array([s[3] for s in final_states])
    
    mean_x, mean_theta, mean_theta_dot = np.mean(final_xs), np.mean(final_thetas), np.mean(final_thetas_dot)
    std_x, std_theta, std_theta_dot = np.std(final_xs, ddof=1), np.std(final_thetas, ddof=1), np.std(final_thetas_dot, ddof=1)

    print(f"\n[Test LQR] === 最终状态统计 ===")
    print(f"目标状态：(x={x_star[0]:.3f}, θ={x_star[2]:.3f}, θ'={x_star[3]:.3f})")
    print(f"均值状态：(x={mean_x:.3f}, θ={mean_theta:.3f}, θ'={mean_theta_dot:.3f})")
    print(f"标准差：std_x={std_x:.3f}, std_θ={std_theta:.3f}, std_θ'={std_theta_dot:.3f}")
    print(f"成功稳定：{success_count}/{num_episodes} 次")

    # -------------------------- 轨迹绘图 --------------------------
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10.colors

    # 绘制所有轨迹（杆角度-角速度）
    for ep, traj in enumerate(all_trajectories):
        theta_coords = [p[0] for p in traj]
        theta_dot_coords = [p[1] for p in traj]
        plt.plot(theta_coords, theta_dot_coords, color=colors[ep % len(colors)], alpha=0.7,
                 label=f"Episode {ep+1}" if ep < 5 else "")  # 只显示前5条轨迹标签

    # 标注目标/均值/稳定区（CartPole专属）
    plt.scatter(x_star[2], x_star[3], color="red", marker="s", s=80, label="Target State (θ=0, θ'=0)")
    plt.scatter(mean_theta, mean_theta_dot, color="blue", marker="o", s=100, 
                label=f"Final Mean (θ={mean_theta:.3f}, θ'={mean_theta_dot:.3f})")
    # 状态标准差区域（杆角度-角速度）
    plt.gca().add_patch(plt.Rectangle(
        (mean_theta - std_theta, mean_theta_dot - std_theta_dot), 
        2*std_theta, 2*std_theta_dot,
        color="blue", alpha=0.2, linestyle="--", label="θ Std Range (±1σ)"
    ))
    # CartPole稳定区域（10°内+角速度限制）
    plt.gca().add_patch(plt.Rectangle(
        (-0.174, -0.5), 0.348, 1.0,  # theta∈[-0.174,0.174], theta_dot∈[-0.5,0.5]
        color="green", alpha=0.1, linestyle="-", label="Success Region (θ±10°, θ'±0.5)"
    ))

    # 图表配置（适配CartPole）
    plt.xlim(-np.pi/2, np.pi/2)  # CartPole杆角度范围（超过则结束）
    plt.ylim(-5, 5)              # 合理角速度范围
    plt.xlabel("Pole Angle θ (rad)", fontsize=12)
    plt.ylabel("Pole Angular Velocity θ' (rad/s)", fontsize=12)
    plt.title(f"CartPole Trajectory Summary ({version})", fontsize=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1), fontsize=10)
    plt.grid(True, alpha=0.5)

    # 保存图片（路径适配CartPole）
    os.makedirs("./fig/cartpole", exist_ok=True)
    plt.savefig(f"./fig/cartpole/krbf_cartpole_trajectory_{version}.png", bbox_inches="tight", dpi=300)
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
    """使用KRBF模型预测CartPole轨迹（逻辑不变，适配4维状态）"""
    predicted_states = krbf.predict(
        x0=initial_state,
        U_seq=control_sequence.T)
    return predicted_states.T


def evaluate_trajectory_prediction(
    krbf: KRBFModel,
    test_data_path: str,
    num_experiments: int = 4,
    save_results: bool = True,
    result_path: str = "./results/cartpole",  # 路径适配CartPole
) -> Tuple[np.ndarray, np.ndarray]:
    """评估CartPole轨迹预测性能（逻辑不变，适配4维状态数据）"""
    os.makedirs(result_path, exist_ok=True)
    
    # 加载CartPole测试数据（4维状态）
    test_data = np.load(test_data_path)
    extended_X_seq = test_data['extended_X_seq']  # [num_episodes, 2*K_steps, 4]
    extended_U_seq = test_data['extended_U_seq']  # [num_episodes, 2*K_steps, 1]（离散动作0/1）
    extended_Y_seq = test_data['extended_Y_seq']  # [num_episodes, 2*K_steps, 4]
    K_steps = 15
    seq_length = extended_X_seq.shape[1]
    
    print(f"\n[轨迹预测评估] 开始{num_experiments}次实验，预测长度={seq_length}步")
    print(f"[轨迹预测评估] 测试数据规模：{extended_X_seq.shape[0]}个序列（4维状态）")
    
    all_errors = []
    for exp_idx in range(num_experiments):
        print(f"\n[实验 {exp_idx+1}/{num_experiments}]")
        episode_errors = []
        
        for ep_idx in trange(extended_X_seq.shape[0], desc="处理测试序列"):
            initial_state = extended_X_seq[ep_idx, 0]  # 4维初始状态
            control_sequence = extended_U_seq[ep_idx]  # 离散控制序列（0/1）
            
            # 真实状态序列（4维）
            true_states = np.vstack((initial_state, extended_Y_seq[ep_idx]))
            # 预测轨迹（KRBF输出4维状态）
            predicted_states = predict_trajectory(
                krbf=krbf,
                initial_state=initial_state,
                control_sequence=control_sequence,
                horizon=seq_length
            )
            
            # 计算每一步的预测误差（欧氏距离，4维状态）
            step_errors = np.linalg.norm(predicted_states - true_states, axis=1)
            episode_errors.append(step_errors)
        
        exp_errors = np.mean(episode_errors, axis=0)
        all_errors.append(exp_errors)
        print(f"[实验 {exp_idx+1}] 平均误差范围: {np.min(exp_errors):.6f} - {np.max(exp_errors):.6f}")
    
    # 计算统计量
    mean_errors = np.mean(all_errors, axis=0)
    log10_errors = np.log10(mean_errors + 1e-10)  # 避免log(0)
    
    # 保存结果（路径适配CartPole）
    if save_results:
        result_file = os.path.join(result_path, f"krbf_cartpole_pred_results_K{K_steps}.npz")
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
    
    # 绘制误差曲线（标题适配CartPole）
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(seq_length + 1)
    
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, mean_errors, 'b-', linewidth=2)
    plt.ylabel('Mean Prediction Error (4D State)')
    plt.title(f'CartPole Trajectory Prediction Errors (2*K={seq_length} steps)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, log10_errors, 'r-', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('log10(Mean Prediction Error)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    error_plot_path = os.path.join("./fig/cartpole", f"prediction_errors_K{K_steps}.png")  # 路径适配
    plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[轨迹预测评估] 误差曲线图保存至：{error_plot_path}")
    
    return mean_errors, log10_errors


def main():
    """CartPole KRBF完整实验流程（适配4维状态+离散动作）"""
    # -------------------------- 1. 配置参数（适配CartPole） --------------------------
    config = {
        "seed": 2,
        "num_data_episodes": 200,
        "num_test_episodes": 100,
        "N_rbf": 10,
        "Np_mpc": 50,
        "K_steps_data": 15,
        # CartPole训练数据路径（需与之前生成的CartPole数据匹配）
        "data_load_path": "./data/cartpole/train_data_CartPole-v1_n4_m1_deriv2_K15_seed2.npz",
        # CartPole测试数据路径
        "extended_test_path": "./data/cartpole/test_data_CartPole-v1_n4_m1_K15_seed2_extended.npz",
        # CartPole Koopman矩阵保存路径
        "koopman_save_path": "./data/cartpole/krbf_cartpole_koopman_matrix.npz",
        "num_prediction_experiments": 4
    }
    
    # 加载CartPole训练数据（4维状态）
    data = np.load(config["data_load_path"])
    x_seq = data['X_seq']          # [num, K_steps, 4]
    u_seq = data['U_seq']          # [num, K_steps, 1]（离散动作0/1）
    x_next_seq = data['Y_seq']     # [num, K_steps, 4]
    X_single = data['X']           # [num_samples, 4]
    U_single = data['U']           # [num_samples, 1]
    Y_single = data['Y']           # [num_samples, 4]

    # -------------------------- 2. KRBF模型初始化（适配CartPole） --------------------------
    krbf = KRBFModel(
        n=4,  # CartPole状态维度：x, x_dot, theta, theta_dot
        m=1,  # CartPole动作维度：离散动作（0/1），按1维处理
        N_rbf=config["N_rbf"],
        Np=config["Np_mpc"],
        # CartPole状态范围（参考环境默认限制）
        state_low=[-2.4, -5.0, -np.pi/2, -10.0],  # x∈[-2.4,2.4], theta∈[-pi/2,pi/2]
        state_high=[2.4, 5.0, np.pi/2, 10.0],
        # CartPole动作范围（离散0/1）
        action_low=[0.0],
        action_high=[1.0]
    )
    krbf.set_data(X_single, U_single, Y_single)  # 传入4维状态数据
    krbf.solve_koopman()

    # -------------------------- 3. LQR增益计算（适配CartPole状态权重） --------------------------
    # Q矩阵：杆角度权重最高（优先稳定杆），小车位置次之，速度权重最小
    Q_lqr = np.diag([1.0, 0.1, 10.0, 0.1])  # [x, x_dot, theta, theta_dot]权重
    R_lqr = np.array([[0.1]])               # 动作权重（避免控制过大）
    K_lqr = krbf.compute_lqr_gain(Q_lqr, R_lqr)

    # -------------------------- 4. 目标状态与LQR测试（CartPole专属目标） --------------------------
    x_star = np.array([0.0, 0.0, 0.0, 0.0])  # 目标：小车居中(x=0)、杆竖直(theta=0)、速度为0
    test_krbf_cartpole_lqr(
        krbf=krbf,
        K_lqr=K_lqr,
        x_star=x_star,
        num_episodes=config["num_test_episodes"],
        seed=config["seed"],
        version="KRBF-LQR_v0"
    )
    
    # -------------------------- 5. 轨迹预测评估（适配CartPole数据） --------------------------
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
        print(f"\n警告：未找到CartPole扩展测试数据，请先运行数据生成脚本生成 {config['extended_test_path']}")


if __name__ == "__main__":
    main()