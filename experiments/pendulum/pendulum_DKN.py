import torch
import gym
import os
import torch.optim as optim
import numpy as np
import argparse
import math
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
    """绘制训练损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"DKN Training Loss ({version})")
    plt.legend()
    os.makedirs("./fig/pendulum", exist_ok=True)
    plt.savefig(f"./fig/pendulum/dkn_loss_{version}.png", dpi=300)
    plt.close()


def test_pendulum_lqr(
    psi: KStepsPredictor,
    K_lqr: np.ndarray,
    x_star: torch.Tensor,  # 目标状态：[1, 0, 0]（θ=0）
    num_episodes: int = 10,
    max_steps: int = 200,
    version: str = "DKN-LQR",
    seed: int = 2
) -> List[float]:
    """DKN+LQR控制测试（Pendulum倒立控制）"""
    env = gym.make("Pendulum-v0")
    env.seed(seed)
    device = next(psi.parameters()).device
    episode_scores: List[float] = []
    all_trajectories: List[List[Tuple[float, float]]] = []  # (θ, θ_dot)轨迹
    success_count = 0  # 成功条件：θ ≤ 0.1rad且θ_dot ≤ 1

    print(f"\n[Test LQR] 开始{num_episodes}回合测试...")
    psi.eval()
    with torch.no_grad():
        for ep in range(num_episodes):
            x_prev = env.reset()[:3]  # 取3维状态
            done = False
            total_score = 0.0
            step = 0
            trajectory = []

            while not done and step < max_steps:
                # 计算θ（从cosθ和sinθ反推）
                theta = np.arctan2(x_prev[1], x_prev[0])
                trajectory.append((theta, x_prev[2]))  # 记录角度和角速度

                # 状态嵌入与控制计算
                x_prev_tensor = torch.tensor(x_prev, device=device, dtype=torch.float32).unsqueeze(0)
                z_prev = psi.StateEmbedding(x_prev_tensor) - psi.StateEmbedding(x_star.unsqueeze(0))
                z_prev_np = z_prev.squeeze(0).cpu().numpy()

                u_t_ = -K_lqr @ z_prev_np.T  # LQR控制律
                u_t_ = torch.tensor(u_t_.T, device=device, dtype=torch.float32)
                u_t = psi.decode_control(u_t_).squeeze(0).cpu().numpy()
                u_t = np.clip(u_t, -2.0, 2.0)  # 限制力矩范围

                x_next, reward, done, _ = env.step([u_t])
                total_score += reward
                x_prev = x_next[:3]
                step += 1

            # 成功判断
            final_theta = np.arctan2(x_prev[1], x_prev[0])
            if abs(final_theta) <= 0.1 and abs(x_prev[2]) <= 1.0:
                success_count += 1

            all_trajectories.append(trajectory)
            episode_scores.append(total_score)
            print(f"回合 {ep+1:2d} | 得分：{total_score:5.1f} | 步数：{step:3d} | 最终θ：{final_theta:.3f} rad")

    env.close()

    # 轨迹可视化（角度-时间曲线）
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    for ep, traj in enumerate(all_trajectories):
        thetas = [t[0] for t in traj]
        plt.plot(thetas, color=colors[ep % len(colors)], alpha=0.7, label=f"Episode {ep+1}")

    plt.axhline(0, color='r', linestyle='--', label="Target θ=0")
    plt.axhline(0.1, color='k', linestyle=':', alpha=0.5)
    plt.axhline(-0.1, color='k', linestyle=':', alpha=0.5)
    plt.ylabel("Theta (rad)")
    plt.xlabel("Step")
    plt.title(f"Pendulum Trajectories ({version})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    os.makedirs("./fig/pendulum", exist_ok=True)
    plt.savefig(f"./fig/pendulum/trajectory_{version}.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 测试总结
    avg_score = np.mean(episode_scores)
    std_score = np.std(episode_scores)
    print(f"\n=== 测试总结 ===")
    print(f"平均得分：{avg_score:.1f}±{std_score:.1f} | 成功倒立：{success_count}/{num_episodes}")
    return episode_scores


def train_psi_pendulum(
    x_prev: np.ndarray,
    u_prev: np.ndarray,
    x_next: np.ndarray,
    z_dim: int = 8,  # 嵌入维度（Pendulum较简单，可减小）
    epochs: int = 300,
    batch_size: int = 128,
    lr: float = 1e-4,
    K_steps: int = 10,
    version: str = "v0"
) -> KStepsPredictor:
    """训练Pendulum的KStepsPredictor模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Pendulum状态范围：[cosθ, sinθ, θ_dot]
    state_low = [-1.0, -1.0, -8.0]
    state_high = [1.0, 1.0, 8.0]
    x_dim = 3
    control_dim = 1

    # 转换数据为tensor
    x_true_series_tensor = torch.tensor(x_next, device=device, dtype=torch.float32)
    u_series_tensor = torch.tensor(u_prev, device=device, dtype=torch.float32)
    x_prev_batch = torch.tensor(x_prev, device=device, dtype=torch.float32)

    # 初始化模型
    psi = KStepsPredictor(
        x_dim=x_dim,
        control_dim=control_dim,
        z_dim=z_dim,
        hidden_dim=128,  # 简化网络
        low=state_low,
        high=state_high,
        K_steps=K_steps,
        device=device
    ).to(device)

    optimizer = optim.Adam(psi.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    avg_loss_list: List[float] = []

    # 训练循环
    psi.train()
    for epoch in range(epochs):
        total_epoch_loss = 0.0
        dataset = TensorDataset(x_prev_batch, u_series_tensor, x_true_series_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        for batch in dataloader:
            x_prev_b, u_series_b, x_true_series_b = batch
            z_pred_series, u_decode_series = psi(x_prev_b, u_series_b)
            z_true_series = psi.StateEmbedding(x_true_series_b)  # 真实状态嵌入

            # 损失计算：嵌入损失 + 控制解码损失
            loss1 = loss_function(z_pred_series, z_true_series)
            loss2 = loss_function(u_decode_series, u_series_b)
            batch_loss = loss1 + loss2

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_epoch_loss += batch_loss.item() * x_prev_b.shape[0]

        # 学习率衰减
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
    result_path: str = "./results/pendulum",
    seed: int = 2,
    z_dim: int = 16,
    x_dim: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    评估KStepsPredictor的2*K_steps轨迹预测性能
    功能：重复num_experiments次实验，计算每个时间步的预测误差log10均值并保存
    Args:
        model: 训练好的KStepsPredictor模型
        test_data_path: 扩展测试数据路径（含2*K_steps序列）
        num_experiments: 实验次数（默认4次）
        save_results: 是否保存结果
        result_path: 结果保存路径
        seed: 随机种子（确保可复现）
    Returns:
        mean_errors: 所有实验的平均误差 [2*K_steps + 1]
        log10_errors: 平均误差的log10值 [2*K_steps + 1]
    """
    # 1. 初始化与数据加载
    os.makedirs(result_path, exist_ok=True)
    test_data = np.load(test_data_path)
    
    # 提取扩展测试数据（2*K_steps长度）
    extended_X_seq = test_data['extended_X_seq']  # [num_test_ep, 2*K_steps, x_dim]
    extended_U_seq = test_data['extended_U_seq']  # [num_test_ep, 2*K_steps, u_dim]
    extended_Y_seq = test_data['extended_Y_seq']  # [num_test_ep, 2*K_steps, x_dim]
    K_steps = 15
    seq_length =  2*K_steps
    num_test_ep = extended_X_seq.shape[0]
    device = next(model.parameters()).device
    control_dim = 1

    print(f"\n=== 开始{num_experiments}次2*K_steps轨迹预测实验 ===")
    print(f"测试数据：{num_test_ep}个序列，每个序列长度={seq_length}（2*{K_steps}）步")

    # 2. 获取Koopman矩阵A、B与重构矩阵C（用于多步预测）
    A, B, Q_, R_ = calculate_parameter(model, x_dim, z_dim, control_dim)
     # 构造状态重构矩阵C（文档Equation 9：从高维z恢复原状态x）
    A = A.cpu().detach().numpy()
    B = B.cpu().detach().numpy()
    I_n = torch.eye(x_dim+z_dim, x_dim, device=device)
    zero_mat = torch.zeros(x_dim+z_dim, z_dim, device=device)
    C = torch.cat([I_n, zero_mat], dim=1).cpu().detach().numpy()  # [x_dim, z_dim]

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
                initial_state = extended_X_seq[ep_idx, 0]  # [x_dim]
                control_seq = extended_U_seq[ep_idx] # [2*K_steps, u_dim]
                # 真实轨迹：[初始状态] + [extended_Y_seq] → [2*K_steps + 1, x_dim]
                true_trajectory = np.vstack([initial_state, extended_Y_seq[ep_idx]])
                # b. 多步预测：手动实现2*K_steps演化（适配KStepsPredictor的Koopman逻辑）
                pred_trajectory = np.zeros_like(true_trajectory)  # [2*K_steps + 1, x_dim]
                pred_trajectory[0] = initial_state  # 初始状态

                # 初始状态嵌入（适配model.StateEmbedding的batch输入）
                z_curr = model.StateEmbedding(
                    torch.tensor(initial_state, device=device, dtype=torch.float32).unsqueeze(0)
                ).squeeze(0).cpu().numpy()  # [z_dim]

                # 逐步预测2*K_steps
                for t in range(seq_length):
                    # 当前控制输入
                    u_t = control_seq[t]   # [u_dim]
                    # Koopman线性演化：z_{t+1} = A·z_t + B·u_t（注意矩阵维度对齐）
                    z_next = A @ z_curr + B @ u_t  # [z_dim]
                    # 重构原状态：x_{t+1} = C·z_{t+1}
                    x_next = C @ z_next  # [x_dim]
                    # 记录预测状态
                    pred_trajectory[t+1] = x_next[:x_dim]
                    # 更新当前z
                    z_curr = z_next

                # c. 计算每个时间步的欧氏距离误差
                step_errors = np.linalg.norm(pred_trajectory - true_trajectory, axis=1)  # [2*K_steps + 1]
                per_episode_errors.append(step_errors)

        # 4. 计算当前实验的平均误差（所有测试序列的均值）
        exp_average_errors = np.mean(per_episode_errors, axis=0)  # [2*K_steps + 1]
        all_experiment_errors.append(exp_average_errors)
        print(f"实验 {exp_idx+1} 误差范围：{np.min(exp_average_errors):.6f} ~ {np.max(exp_average_errors):.6f}")

    # 5. 计算所有实验的统计指标
    mean_errors = np.mean(all_experiment_errors, axis=0)  # 4次实验的平均误差
    log10_errors = np.log10(mean_errors + 1e-10)  # 避免log(0)

    # 6. 保存实验结果
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

    # 7. 绘制误差曲线
    plot_prediction_errors(mean_errors, log10_errors, seq_length, K_steps)

    return mean_errors, log10_errors

def calculate_parameter(psi: KStepsPredictor, x_dim: int, z_dim: int, control_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """适配KStepsPredictor的参数计算：获取Koopman矩阵A、B及LQR权重Q_、R_"""
    device = next(psi.parameters()).device
    # KStepsPredictor的KoopmanOperator是Linear层，权重维度为[z_dim, z_dim]（A）、[z_dim, control_dim]（B）
    A_lander = psi.KoopmanOperator.A.weight # [z_dim, z_dim]
    B_lander = psi.KoopmanOperator.B.weight # [z_dim, control_dim]
    
    # 构造状态重构矩阵C（文档Equation 9：从高维z恢复原状态x）
    I_n = torch.eye(x_dim+z_dim, x_dim, device=device)
    zero_mat = torch.zeros(x_dim+z_dim, z_dim, device=device)
    C = torch.cat([I_n, zero_mat], dim=1).cpu().detach().numpy()  # [x_dim, z_dim]
    
    # 构造LQR权重（文档III节：状态权重Q聚焦位置，控制权重R抑制过大输入）
    Q = np.eye(x_dim+z_dim)
    Q_ = C.T @ Q @ C          # 映射到高维空间：[z_dim, z_dim]
    R_ = 0.1 * np.eye(control_dim)  # 控制权重
    print(A_lander.shape, B_lander.shape, C.shape, Q_.shape, R_.shape)
    return A_lander, B_lander, Q_, R_


def plot_prediction_errors(
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
    plt.plot(time_steps, mean_errors, color="#2E86AB", linewidth=2.5, label=f"Mean Euclidean Error")
    plt.ylabel("Error (Euclidean Distance)", fontsize=12)
    plt.title(f"DKN Trajectory Prediction Errors (2*K={seq_length} Steps)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # 子图2：log10误差（突出误差变化趋势）
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, log10_errors, color="#A23B72", linewidth=2.5, label=f"log10(Mean Error)")
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("log10(Error)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # 保存图片
    os.makedirs("./fig/lunarlander", exist_ok=True)
    plot_path = os.path.join("./fig/lunarlander", f"dkn_pred_errors_K{K_steps}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"误差曲线保存至：{plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_dim", type=int, default=3)
    parser.add_argument("--control_dim", type=int, default=1)
    parser.add_argument("--K_steps", type=int, default=15)
    parser.add_argument("--version", type=str, default="dkn_pendulum_v0")
    args = parser.parse_args()

    # 加载训练数据（需先运行generate_pendulum_data.py）
    data = np.load(f"./data/pendulum/train_data_Pendulum-v0_n3_m1_deriv2_K{args.K_steps}_seed2.npz")
    x_prev, u_prev, x_next = data["X_seq"], data["U_seq"], data["Y_seq"]

    # 训练模型
    psi = train_psi_pendulum(
        x_prev=x_prev,
        u_prev=u_prev,
        x_next=x_next,
        z_dim=16,
        epochs=100,
        K_steps=args.K_steps,
        version=args.version
    )

    # 定义目标状态（θ=0：cosθ=1, sinθ=0, θ_dot=0）
    x_star = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=next(psi.parameters()).device)

    # 求解LQR增益（示例：需根据实际线性化模型调整Q/R矩阵）
    Q = np.diag([1.0, 1.0, 1.0])  # 状态权重（角度相关项权重更高）
    R = np.diag([0.1])  # 控制权重
    A = psi.KoopmanOperator.A.weight
    B = psi.KoopmanOperator.B.weight
    K_lqr = solve_discrete_lqr(A, B)  # 示例A/B需替换为实际线性化矩阵

    # 测试LQR控制
    test_pendulum_lqr(
        psi=psi,
        K_lqr=K_lqr,
        x_star=x_star,
        num_episodes=100,
        version=args.version
    )

    evaluate_trajectory_prediction(
        model=psi,
        test_data_path="./data/pendulum/test_data_Pendulum-v0_n3_m1_K15_seed2_extended.npz",
        z_dim=16
    )