import torch
import gym
import os
import torch.optim as optim
import numpy as np
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange
from typing import List, Tuple
from torch.utils.data import TensorDataset, DataLoader
# 假设原有模型/控制器模块路径正确
from rdkrc.models.DKN import KStepsPredictor
from rdkrc.controller.lqr_controller import solve_discrete_lqr
from rdkrc.controller.mpc_controller import DKRCMPCController


def plot_loss_curve(losses: List[float], version: str) -> None:
    """绘制训练损失曲线（路径适配CartPole）"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"DKN Training Loss ({version})")
    plt.legend()
    os.makedirs("./fig/cartpole", exist_ok=True)  # 路径改为CartPole专属
    plt.savefig(f"./fig/cartpole/dkn_loss_{version}.png", dpi=300)
    plt.close()


def test_cartpole_lqr(
    psi: KStepsPredictor,
    K_lqr: np.ndarray,
    x_star: torch.Tensor,  # 目标状态：[0,0,0,0]（小车居中、杆竖直）
    num_episodes: int = 10,
    max_steps: int = 500,  # CartPole默认最大步数
    version: str = "DKN-LQR",
    seed: int = 2
) -> List[float]:
    """DKN+LQR控制测试（CartPole稳定控制）"""
    env = gym.make("CartPole-v1")  # 切换为CartPole环境
    env.seed(seed)
    device = next(psi.parameters()).device
    episode_scores: List[float] = []
    all_trajectories: List[List[Tuple[float, float]]] = []  # (θ, θ_dot)轨迹
    success_count = 0  # 成功条件：x∈[-0.5,0.5]、θ∈[-0.174,0.174]、θ_dot∈[-0.5,0.5]

    print(f"\n[Test LQR] 开始{num_episodes}回合测试...")
    psi.eval()
    with torch.no_grad():
        for ep in range(num_episodes):
            x_prev = env.reset()  # CartPole直接返回4维状态：[x, x_dot, θ, θ_dot]
            done = False
            total_score = 0.0
            step = 0
            trajectory = []

            while not done and step < max_steps:
                # 记录杆角度（θ）和角速度（θ_dot）
                theta = x_prev[2]
                theta_dot = x_prev[3]
                trajectory.append((theta, theta_dot))

                # 状态嵌入与控制计算
                x_prev_tensor = torch.tensor(x_prev, device=device, dtype=torch.float32).unsqueeze(0)
                z_prev = psi.StateEmbedding(x_prev_tensor) - psi.StateEmbedding(x_star.unsqueeze(0))
                z_prev_np = z_prev.squeeze(0).cpu().numpy()

                u_t_ = -K_lqr @ z_prev_np.T  # LQR连续控制输出
                u_t_ = torch.tensor(u_t_.T, device=device, dtype=torch.float32)
                u_t = psi.decode_control(u_t_).squeeze(0).cpu().numpy()
                
                # 关键：CartPole离散动作映射（0=左推，1=右推）
                u_discrete = 1 if u_t >= 0.5 else 0  # 连续输出→离散动作
                u_discrete = np.clip(u_discrete, 0, 1)  # 确保动作在合法范围

                # 环境交互（CartPole输入离散动作，无需列表包装）
                x_next, reward, done, _ = env.step(u_discrete)
                total_score += reward
                x_prev = x_next  # CartPole状态直接更新
                step += 1

            # 成功判断（CartPole专属条件）
            final_x = x_prev[0]
            final_theta = x_prev[2]
            final_theta_dot = x_prev[3]
            if (abs(final_x) <= 0.5 and abs(final_theta) <= 0.174 and 
                abs(final_theta_dot) <= 0.5):
                success_count += 1

            all_trajectories.append(trajectory)
            episode_scores.append(total_score)
            print(f"回合 {ep+1:2d} | 得分：{total_score:5.1f} | 步数：{step:3d} | "
                  f"最终x：{final_x:.3f} | 最终θ：{final_theta:.3f} rad")

    env.close()

    # 轨迹可视化（杆角度-时间曲线）
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    for ep, traj in enumerate(all_trajectories):
        thetas = [t[0] for t in traj]
        plt.plot(thetas, color=colors[ep % len(colors)], alpha=0.7, label=f"Episode {ep+1}" if ep < 5 else "")

    # 标注目标与稳定区
    plt.axhline(0, color='r', linestyle='--', label="Target θ=0")
    plt.axhline(0.174, color='k', linestyle=':', alpha=0.5, label="θ±10°")
    plt.axhline(-0.174, color='k', linestyle=':', alpha=0.5)
    plt.ylabel("Pole Angle θ (rad)")
    plt.xlabel("Step")
    plt.title(f"CartPole Trajectories ({version})")  # 标题适配CartPole
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    os.makedirs("./fig/cartpole", exist_ok=True)
    plt.savefig(f"./fig/cartpole/trajectory_{version}.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 测试总结
    avg_score = np.mean(episode_scores)
    std_score = np.std(episode_scores)
    print(f"\n=== 测试总结 ===")
    print(f"平均得分：{avg_score:.1f}±{std_score:.1f} | 成功稳定：{success_count}/{num_episodes}")
    return episode_scores


def train_psi_cartpole(
    x_prev: np.ndarray,
    u_prev: np.ndarray,
    x_next: np.ndarray,
    z_dim: int = 8,  # 嵌入维度（CartPole复杂度适中，无需过大）
    epochs: int = 300,
    batch_size: int = 128,
    lr: float = 1e-4,
    K_steps: int = 10,
    version: str = "v0"
) -> KStepsPredictor:
    """训练CartPole的KStepsPredictor模型（适配4维状态）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CartPole状态范围（参考环境物理限制）
    state_low = [-2.4, -5.0, -np.pi/2, -10.0]  # x, x_dot, θ, θ_dot
    state_high = [2.4, 5.0, np.pi/2, 10.0]
    x_dim = 4  # CartPole状态维度从3→4
    control_dim = 1  # 动作维度仍为1（离散0/1按连续值处理）

    # 转换数据为tensor（确保输入为4维状态）
    x_true_series_tensor = torch.tensor(x_next, device=device, dtype=torch.float32)
    u_series_tensor = torch.tensor(u_prev, device=device, dtype=torch.float32)
    x_prev_batch = torch.tensor(x_prev, device=device, dtype=torch.float32)

    # 初始化模型（适配4维状态）
    psi = KStepsPredictor(
        x_dim=x_dim,
        control_dim=control_dim,
        z_dim=z_dim,
        hidden_dim=128,
        low=state_low,
        high=state_high,
        K_steps=K_steps,
        device=device
    ).to(device)

    optimizer = optim.Adam(psi.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    avg_loss_list: List[float] = []

    # 训练循环（逻辑不变，适配4维数据）
    psi.train()
    for epoch in range(epochs):
        total_epoch_loss = 0.0
        dataset = TensorDataset(x_prev_batch, u_series_tensor, x_true_series_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        for batch in dataloader:
            x_prev_b, u_series_b, x_true_series_b = batch
            z_pred_series, u_decode_series = psi(x_prev_b, u_series_b)
            z_true_series = psi.StateEmbedding(x_true_series_b)  # 4维状态嵌入

            # 损失计算：嵌入损失 + 控制解码损失（逻辑不变）
            loss1 = loss_function(z_pred_series, z_true_series)
            loss2 = loss_function(u_decode_series, u_series_b)
            batch_loss = loss1 + loss2

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_epoch_loss += batch_loss.item() * x_prev_b.shape[0]

        # 学习率衰减（保持原策略）
        if (epoch + 1) % 50 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

        avg_epoch_loss = total_epoch_loss / len(x_prev_batch)
        avg_loss_list.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1:3d}/{epochs}] | 平均损失：{avg_epoch_loss:.6f}", end='\r')

    plot_loss_curve(avg_loss_list, version)
    return psi


def evaluate_trajectory_prediction(
    model: KStepsPredictor,
    test_data_path: str,
    num_experiments: int = 4,
    save_results: bool = True,
    result_path: str = "./results/cartpole",  # 路径适配CartPole
    seed: int = 2,
    z_dim: int = 16,
    x_dim: int = 4  # CartPole状态维度从3→4
) -> Tuple[np.ndarray, np.ndarray]:
    """评估KStepsPredictor的2*K_steps轨迹预测性能（适配CartPole 4维状态）"""
    # 1. 初始化与数据加载
    os.makedirs(result_path, exist_ok=True)
    test_data = np.load(test_data_path)
    
    # 提取CartPole扩展测试数据（4维状态）
    extended_X_seq = test_data['extended_X_seq']  # [num_test_ep, 2*K_steps, 4]
    extended_U_seq = test_data['extended_U_seq']  # [num_test_ep, 2*K_steps, 1]（离散动作0/1）
    extended_Y_seq = test_data['extended_Y_seq']  # [num_test_ep, 2*K_steps, 4]
    K_steps = 15
    seq_length = 2 * K_steps
    num_test_ep = extended_X_seq.shape[0]
    device = next(model.parameters()).device
    control_dim = 1

    print(f"\n=== 开始{num_experiments}次2*K_steps轨迹预测实验 ===")
    print(f"测试数据：{num_test_ep}个序列，每个序列长度={seq_length}=2*{K_steps}步（4维状态）")

    # 2. 获取Koopman矩阵A、B与重构矩阵C（适配4维状态）
    A, B, Q_, R_ = calculate_parameter(model, x_dim, z_dim, control_dim)
    A = A.cpu().detach().numpy()
    B = B.cpu().detach().numpy()
    # 重构矩阵C：从z_dim映射回4维状态
    C = np.hstack([np.eye(x_dim), np.zeros((x_dim, z_dim))])  # [4, z_dim]

    # 3. 存储所有实验的误差
    all_experiment_errors = []
    torch.manual_seed(seed)
    np.random.seed(seed)

    for exp_idx in range(num_experiments):
        print(f"\n--- 实验 {exp_idx+1}/{num_experiments} ---")
        per_episode_errors = []

        # 模型切换到推理模式
        model.eval()
        with torch.no_grad():
            for ep_idx in trange(num_test_ep, desc="处理测试序列"):
                # a. 提取当前序列的初始状态、控制序列、真实轨迹
                initial_state = extended_X_seq[ep_idx, 0]  # [4]
                control_seq = extended_U_seq[ep_idx]       # [2*K_steps, 1]
                # 真实轨迹：[初始状态] + [extended_Y_seq] → [2*K_steps + 1, 4]
                true_trajectory = np.vstack([initial_state, extended_Y_seq[ep_idx]])
                # b. 多步预测：手动实现2*K_steps演化
                pred_trajectory = np.zeros_like(true_trajectory)  # [2*K_steps + 1, 4]
                pred_trajectory[0] = initial_state  # 初始状态

                # 初始状态嵌入（适配4维输入）
                z_curr = model.StateEmbedding(
                    torch.tensor(initial_state, device=device, dtype=torch.float32).unsqueeze(0)
                ).squeeze(0).cpu().numpy()  # [z_dim]

                # 逐步预测2*K_steps
                for t in range(seq_length):
                    # 当前控制输入（离散动作0/1，按连续值处理）
                    u_t = control_seq[t].flatten()  # [1]
                    # Koopman线性演化：z_{t+1} = A·z_t + B·u_t
                    z_next = A @ z_curr + B @ u_t  # [z_dim]
                    # 重构原4维状态：x_{t+1} = C·z_{t+1}
                    x_next = C @ z_next  # [4]
                    # 记录预测状态
                    pred_trajectory[t+1] = x_next[:x_dim]
                    # 更新当前z
                    z_curr = z_next

                # c. 计算每个时间步的欧氏距离误差（4维状态）
                step_errors = np.linalg.norm(pred_trajectory - true_trajectory, axis=1)  # [2*K_steps + 1]
                per_episode_errors.append(step_errors)

        # 4. 计算当前实验的平均误差
        exp_average_errors = np.mean(per_episode_errors, axis=0)  # [2*K_steps + 1]
        all_experiment_errors.append(exp_average_errors)
        print(f"实验 {exp_idx+1} 误差范围：{np.min(exp_average_errors):.6f} ~ {np.max(exp_average_errors):.6f}")

    # 5. 计算所有实验的统计指标
    mean_errors = np.mean(all_experiment_errors, axis=0)
    log10_errors = np.log10(mean_errors + 1e-10)  # 避免log(0)

    # 6. 保存实验结果（CartPole路径）
    if save_results:
        result_file = os.path.join(result_path, f"dkn_pred_results_K{K_steps}_exp{num_experiments}.npz")
        np.savez_compressed(
            result_file,
            mean_errors=mean_errors,
            log10_errors=log10_errors,
            K_steps=K_steps,
            seq_length=seq_length,
            num_experiments=num_experiments,
            all_experiment_errors=np.array(all_experiment_errors),
            x_dim=x_dim,
            z_dim=z_dim,
            control_dim=control_dim
        )
        print(f"\n=== 实验结果保存至：{result_file} ===")

    # 7. 绘制误差曲线（适配CartPole）
    plot_prediction_errors(mean_errors, log10_errors, seq_length, K_steps)

    return mean_errors, log10_errors


def calculate_parameter(psi: KStepsPredictor, x_dim: int, z_dim: int, control_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """适配CartPole的参数计算：获取Koopman矩阵A、B及LQR权重Q_、R_（4维状态）"""
    device = next(psi.parameters()).device
    # KStepsPredictor的KoopmanOperator权重（维度适配z_dim）
    A_lander = psi.KoopmanOperator.A.weight  # [z_dim, z_dim]
    B_lander = psi.KoopmanOperator.B.weight  # [z_dim, control_dim]
    
    # 构造4维状态重构矩阵C（从z_dim映射回CartPole状态）
    C = np.hstack([np.eye(x_dim), np.zeros((x_dim, z_dim - x_dim))])  # [4, z_dim]
    
    # 构造LQR权重（CartPole：杆角度权重最高，优先稳定杆）
    Q = np.diag([1.0, 1.0, 1.0, 1.0])  # [x, x_dot, θ, θ_dot]权重
    Q_ = C.T @ Q @ C                    # 映射到高维空间：[z_dim, z_dim]
    R_ = 0.1 * np.eye(control_dim)      # 控制权重（抑制频繁动作）
    print(f"[参数计算] A维度：{A_lander.shape}, B维度：{B_lander.shape}, C维度：{C.shape}")
    return A_lander, B_lander, Q_, R_


def plot_prediction_errors(
    mean_errors: np.ndarray,
    log10_errors: np.ndarray,
    seq_length: int,
    K_steps: int
) -> None:
    """绘制轨迹预测误差曲线（适配CartPole路径与标题）"""
    plt.figure(figsize=(12, 8))
    time_steps = np.arange(seq_length + 1)  # 0 ~ 2*K_steps

    # 子图1：原始平均误差
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, mean_errors, color="#2E86AB", linewidth=2.5, label=f"Mean Euclidean Error (4D State)")
    plt.ylabel("Error (Euclidean Distance)", fontsize=12)
    plt.title(f"DKN CartPole Trajectory Prediction Errors (2*K={seq_length} Steps)", fontsize=14)  # 标题适配
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # 子图2：log10误差
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, log10_errors, color="#A23B72", linewidth=2.5, label=f"log10(Mean Error)")
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("log10(Error)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # 保存图片（CartPole专属路径）
    os.makedirs("./fig/cartpole", exist_ok=True)
    plot_path = os.path.join("./fig/cartpole", f"dkn_pred_errors_K{K_steps}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"误差曲线保存至：{plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_dim", type=int, default=4)  # 状态维度改为4（CartPole）
    parser.add_argument("--control_dim", type=int, default=1)
    parser.add_argument("--K_steps", type=int, default=15)
    parser.add_argument("--version", type=str, default="dkn_cartpole_v0")  # 版本名适配
    args = parser.parse_args()

    # 加载CartPole训练数据（需提前生成4维状态数据）
    data_path = f"./data/cartpole/train_data_CartPole-v1_n4_m1_deriv2_K{args.K_steps}_seed2.npz"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CartPole训练数据不存在，请先运行数据生成脚本：{data_path}")
    data = np.load(data_path)
    x_prev, u_prev, x_next = data["X_seq"], data["U_seq"], data["Y_seq"]
    # 训练CartPole的DKN模型
    psi = train_psi_cartpole(
        x_prev=x_prev,
        u_prev=u_prev,
        x_next=x_next,
        z_dim=16,
        epochs=300,
        K_steps=args.K_steps,
        version=args.version
    )

    # 定义CartPole目标状态：小车居中(x=0)、杆竖直(θ=0)、速度为0
    device = next(psi.parameters()).device
    x_star = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    # 求解LQR增益（适配4维状态权重）
    A, B, _, _ = calculate_parameter(psi, args.x_dim, z_dim=16, control_dim=args.control_dim)
    K_lqr = solve_discrete_lqr(A, B)  # 传入Q/R确保LQR适配CartPole

    # 测试DKN+LQR控制性能
    test_cartpole_lqr(
        psi=psi,
        K_lqr=K_lqr,
        x_star=x_star,
        num_episodes=100,
        version=args.version
    )

    # 评估轨迹预测性能（加载CartPole扩展测试数据）
    test_data_path = "./data/cartpole/test_data_CartPole-v1_n4_m1_K15_seed2_extended.npz"
    if os.path.exists(test_data_path):
        evaluate_trajectory_prediction(
            model=psi,
            test_data_path=test_data_path,
            z_dim=16,
            x_dim=args.x_dim
        )
    else:
        print(f"\n[警告] CartPole扩展测试数据不存在：{test_data_path}")
        print("请先运行CartPole数据生成脚本生成测试数据")