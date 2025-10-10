import gym
import os
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from typing import List, Tuple

# 导入自定义模块
from rdkrc.utils.data_utils import generate_lunar_lander_data_ksteps, load_lunar_lander_data
from rdkrc.models.KRBF import KRBFModel


def test_krbf_lander_lqr(
    krbf: KRBFModel,
    K_lqr: np.ndarray,
    x_star: np.ndarray,
    num_episodes: int = 10,
    max_steps: int = 500,
    seed: int = 2,
    version: str = "KRBF-LQR"
) -> List[float]:
    """
    KRBF+LQR月球着陆器测试（适配原有测试框架，保留落地统计与轨迹绘图）
    对应`KRBF.pdf` 8节数值示例
    """
    env = gym.make("LunarLanderContinuous-v2")
    env.seed(seed)
    episode_scores: List[float] = []
    all_trajectories: List[List[Tuple[float, float]]] = []  # x-y轨迹
    landing_positions: List[Tuple[float, float]] = []       # 落地位置
    success_count = 0  # 成功着陆计数（x∈[-0.5,0.5], y∈[-0.2,0.2]）

    print(f"\n[Test LQR] 开始{num_episodes}回合KRBF+LQR测试...")
    for ep in range(num_episodes):
        x_prev = env.reset()[:6]  # 取6维状态
        done = False
        total_score = 0.0
        step = 0
        trajectory = []  # 当前episode x-y轨迹

        while not done and step < max_steps:
            # 记录x-y轨迹（原有逻辑）
            trajectory.append((x_prev[0], x_prev[1]))

            # 1. KRBF提升：z = ψ(x) - ψ(x*)（`KRBF.pdf` 5节控制偏移）
            z_prev = krbf._psi(x_prev) - krbf._psi(x_star)

            # 2. LQR控制计算：u = -K_lqr · z（`KRBF.pdf` 5节线性控制律）
            u_t = -K_lqr @ z_prev
            u_t = np.clip(u_t, env.action_space.low, env.action_space.high)

            # 3. 环境交互
            x_next, reward, done, _ = env.step(u_t)
            total_score += reward
            x_prev = x_next[:6]
            step += 1

        # 记录结果（原有统计逻辑）
        landing_x, landing_y = x_prev[0], x_prev[1]
        landing_positions.append((landing_x, landing_y))
        trajectory.append((landing_x, landing_y))  # 补充落地位置
        all_trajectories.append(trajectory)
        episode_scores.append(total_score)

        # 成功着陆判断
        if abs(landing_x) <= 0.5 and -0.2 <= landing_y <= 0.2:
            success_count += 1
        print(f"[Test LQR] 回合 {ep+1:2d} | 得分：{total_score:5.1f} | 步数：{step:3d} | 落地：(x={landing_x:.3f}, y={landing_y:.3f})")

    env.close()

    # -------------------------- 落地位置统计（原有逻辑） --------------------------
    landing_xs = np.array([p[0] for p in landing_positions])
    landing_ys = np.array([p[1] for p in landing_positions])
    mean_x, mean_y = np.mean(landing_xs), np.mean(landing_ys)
    std_x, std_y = np.std(landing_xs, ddof=1), np.std(landing_ys, ddof=1)

    print(f"\n[Test LQR] === 落地位置统计 ===")
    print(f"目标位置：(x={x_star[0]:.3f}, y={x_star[1]:.3f})")
    print(f"均值位置：(x={mean_x:.3f}, y={mean_y:.3f})")
    print(f"标准差：std_x={std_x:.3f}, std_y={std_y:.3f}")
    print(f"成功着陆：{success_count}/{num_episodes} 次")

    # -------------------------- 轨迹绘图（原有逻辑） --------------------------
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10.colors

    # 绘制所有轨迹
    for ep, traj in enumerate(all_trajectories):
        x_coords = [p[0] for p in traj]
        y_coords = [p[1] for p in traj]
        plt.plot(x_coords, y_coords, color=colors[ep % len(colors)], alpha=0.7)

    # 标注目标/均值/着陆区
    plt.scatter(x_star[0], x_star[1], color="red", marker="s", s=80, label="Target Landing Pos")
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
    os.makedirs("./fig", exist_ok=True)
    plt.savefig(f"./fig/krbf_lander_trajectory_{version}.png", bbox_inches="tight", dpi=300)
    plt.close()

    # -------------------------- 测试总结 --------------------------
    avg_score = np.mean(episode_scores)
    std_score = np.std(episode_scores)
    print(f"\n[Test LQR] === 测试总结 ===")
    print(f"平均得分：{avg_score:.1f}±{std_score:.1f} | 成功着陆率：{success_count/num_episodes*100:.1f}%")

    return episode_scores

def predict_trajectory(
    krbf: KRBFModel,
    initial_state: np.ndarray,
    control_sequence: np.ndarray,
    horizon: int
) -> np.ndarray:
    """
    使用KRBF模型预测轨迹
    Args:
        krbf: KRBF模型
        initial_state: 初始状态
        control_sequence: 控制序列 [horizon, m]
        horizon: 预测步数
    Returns:
        predicted_states: 预测的状态序列 [horizon+1, n]
    """
    predicted_states = krbf.predict(
        x0=initial_state,
        U_seq=control_sequence.T)
    
    
        
    return predicted_states.T


def evaluate_trajectory_prediction(
    krbf: KRBFModel,
    test_data_path: str,
    num_experiments: int = 4,
    save_results: bool = True,
    result_path: str = "./results"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    评估2*K_steps长度的轨迹预测性能
    重复多次实验，计算每个时间步的预测误差log10值的均值
    """
    # 创建结果保存目录
    os.makedirs(result_path, exist_ok=True)
    
    # 加载扩展测试数据（2*K_steps长度）
    test_data = np.load(test_data_path)
    extended_X_seq = test_data['extended_X_seq']  # [num_episodes, 2*K_steps, n]
    extended_U_seq = test_data['extended_U_seq']  # [num_episodes, 2*K_steps, m]
    extended_Y_seq = test_data['extended_Y_seq']  # [num_episodes, 2*K_steps, n]
    K_steps = test_data['K_steps'].item()
    seq_length = test_data['seq_length'].item()  # 应该是2*K_steps
    
    print(f"\n[轨迹预测评估] 开始{num_experiments}次实验，预测长度={seq_length}步")
    print(f"[轨迹预测评估] 测试数据规模：{extended_X_seq.shape[0]}个序列")
    
    # 存储每次实验的误差
    all_errors = []
    
    for exp_idx in range(num_experiments):
        print(f"\n[实验 {exp_idx+1}/{num_experiments}]")
        episode_errors = []
        
        # 对每个测试序列进行预测
        for ep_idx in trange(extended_X_seq.shape[0], desc="处理测试序列"):
            # 获取初始状态和控制序列
            initial_state = extended_X_seq[ep_idx, 0]
            control_sequence = extended_U_seq[ep_idx]
            
            # 真实状态序列（包含初始状态）
            # 计算extended_Y_seq的Z值
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
        
        # 计算该次实验的平均误差
        exp_errors = np.mean(episode_errors, axis=0)
        all_errors.append(exp_errors)
        print(f"[实验 {exp_idx+1}] 平均误差范围: {np.min(exp_errors):.6f} - {np.max(exp_errors):.6f}")
    
    # 计算所有实验的平均误差
    mean_errors = np.mean(all_errors, axis=0)
    
    # 计算每个时间步的log10误差（添加小常数避免log(0)）
    log10_errors = np.log10(mean_errors + 1e-10)
    
    # 保存结果
    if save_results:
        result_file = os.path.join(result_path, f"trajectory_prediction_results_K{K_steps}.npz")
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
    plt.title(f'Trajectory Prediction Errors (2*K={seq_length} steps)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, log10_errors, 'r-', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('log10(Mean Prediction Error)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    error_plot_path = os.path.join("./fig", f"prediction_errors_K{K_steps}.png")
    plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[轨迹预测评估] 误差曲线图保存至：{error_plot_path}")
    
    return mean_errors, log10_errors


def main():
    """完整测试流程：数据生成→KRBF训练→LQR测试→轨迹预测评估"""
    # -------------------------- 1. 配置参数 --------------------------
    config = {
        "seed": 2,
        "num_data_episodes": 200,
        "num_test_episodes": 100,
        "N_rbf": 100,
        "Np_mpc": 30,
        "K_steps_data": 15,
        "data_load_path": "./data/train_data_LunarLanderContinuous-v2_n6_m2_deriv2_K15_seed2.npz",
        "extended_test_path": "./data/test_data_LunarLanderContinuous-v2_ep100_K15_seed2_extended.npz",  # 新增
        "koopman_save_path": "./data/krbf_koopman_matrix.npz",
        "num_prediction_experiments": 4  # 新增：轨迹预测实验次数
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
        n=6, m=2,
        N_rbf=config["N_rbf"],
        Np=config["Np_mpc"]
    )
    # 设置数据并求解Koopman矩阵
    krbf.set_data(X_single, U_single, Y_single)

    krbf.solve_koopman()


    # -------------------------- 3. LQR增益计算 --------------------------
    Q_lqr = np.eye(6) * 1.0
    R_lqr = np.eye(2) * 0.01
    K_lqr = krbf.compute_lqr_gain(Q_lqr, R_lqr)

    # -------------------------- 4. 目标状态与LQR测试 --------------------------
    x_star = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    test_krbf_lander_lqr(
        krbf=krbf,
        K_lqr=K_lqr,
        x_star=x_star,
        num_episodes=config["num_test_episodes"],
        seed=config["seed"],
        version="KRBF-LQR_v1"
    )
    
    # -------------------------- 5. 新增：轨迹预测评估实验 --------------------------
    if os.path.exists(config["extended_test_path"]):
        mean_errors, log10_errors = evaluate_trajectory_prediction(
            krbf=krbf,
            test_data_path=config["extended_test_path"],
            num_experiments=config["num_prediction_experiments"],
            save_results=True
        )
        
        # 打印轨迹预测结果摘要
        print("\n[轨迹预测评估总结]")
        print(f"预测长度: {len(mean_errors)-1} 步 (2*K_steps)")
        print(f"平均log10误差: {np.mean(log10_errors):.4f}")
        print(f"各时间步log10误差均值:")
        for i in range(0, len(log10_errors), 5):  # 每5步打印一次
            print(f"  第{i}步: {log10_errors[i]:.4f}")
    else:
        print(f"\n警告：未找到扩展测试数据，请先运行数据生成脚本生成 {config['extended_test_path']}")



if __name__ == "__main__":
    main()