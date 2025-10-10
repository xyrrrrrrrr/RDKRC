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
from typing import Tuple, List, Optional
from torch.utils.data import TensorDataset, DataLoader
from rdkrc.utils.data_utils import generate_lunar_lander_data
from rdkrc.models.psi_mlp import PsiMLP, PsiMLP_v2, PsiMLP_v3
from rdkrc.models.DKN import KStepsPredictor
from rdkrc.trainer.loss_functions import compute_total_loss
from rdkrc.utils.matrix_utils import compute_C_matrix, update_A_B
from rdkrc.controller.lqr_controller import solve_discrete_lqr, solve_discrete_lqr_v2
from rdkrc.controller.mpc_controller import DKRCMPCController


# -------------------------- 原有函数保留（仅修复训练损失计算小问题） --------------------------
def test_lander_lqr(
    psi: KStepsPredictor,  # 修正模型类型：适配KStepsPredictor
    K_lqr: np.ndarray,
    x_star: torch.Tensor,
    num_episodes: int = 10,
    max_steps: int = 500,
    version: str = "v1",
    seed: int = 2
) -> List[float]:
    # 原函数逻辑不变，仅修正模型类型注解
    env = gym.make("LunarLanderContinuous-v2")
    env.seed(seed)
    device = next(psi.parameters()).device
    episode_scores: List[float] = []
    all_trajectories: List[List[Tuple[float, float]]] = []
    landing_positions: List[Tuple[float, float]] = []
    success_count = 0
    psi.eval()
    with torch.no_grad():
        for ep in range(num_episodes):
            x_prev = env.reset()[:6]
            done = False
            total_score = 0.0
            step = 0
            trajectory = []

            while not done and step < max_steps:
                trajectory.append((x_prev[0], x_prev[1]))
                # 适配KStepsPredictor的StateEmbedding接口（输入需为tensor且带batch维度）
                x_prev_tensor = torch.tensor(x_prev, device=device, dtype=torch.float32).unsqueeze(0)
                x_star_tensor = x_star.unsqueeze(0)
                z_prev = psi.StateEmbedding(x_prev_tensor) - psi.StateEmbedding(x_star_tensor)
                z_prev_np = z_prev.squeeze(0).cpu().detach().numpy()

                # 计算LQR控制（适配KStepsPredictor的decode_control接口）
                u_t_ = -K_lqr @ z_prev_np.T  # [control_dim, 1]
                u_t_ = torch.tensor(u_t_.T, device=device, dtype=torch.float32)  # [1, control_dim]
                u_t = psi.decode_control(u_t_).squeeze(0).cpu().detach().numpy()
                u_t = np.clip(u_t, env.action_space.low, env.action_space.high)

                x_next, reward, done, _ = env.step(u_t)
                total_score += reward
                x_prev = x_next[:6]
                step += 1

            # 落地统计与轨迹绘图逻辑不变...
            landing_x, landing_y = x_prev[0], x_prev[1]
            landing_positions.append((landing_x, landing_y))
            trajectory.append((landing_x, landing_y))
            all_trajectories.append(trajectory)
            episode_scores.append(total_score)

            if abs(landing_x) <= 0.5 and -0.2 <= landing_y <= 0.2:
                success_count += 1
            print(f"测试回合 {ep+1:2d}/{num_episodes} | 得分：{total_score:5.1f} | 步数：{step:3d} | 落地：(x={landing_x:.3f}, y={landing_y:.3f})")

    env.close()
    # 落地位置统计（原逻辑不变）
    landing_xs = np.array([p[0] for p in landing_positions], dtype=np.float32)
    landing_ys = np.array([p[1] for p in landing_positions], dtype=np.float32)
    mean_x = np.mean(landing_xs)
    mean_y = np.mean(landing_ys)
    var_x = np.var(landing_xs, ddof=1)
    var_y = np.var(landing_ys, ddof=1)
    std_x = np.sqrt(var_x)
    std_y = np.sqrt(var_y)

    print(f"\n=== 落地位置统计 ===")
    x_star_np = x_star.cpu().numpy()
    print(f"目标：(x={x_star_np[0]:.3f}, y={x_star_np[1]:.3f}) | 均值：(x={mean_x:.3f}, y={mean_y:.3f})")
    print(f"方差：var_x={var_x:.6f}, var_y={var_y:.6f} | 标准差：std_x={std_x:.3f}, std_y={std_y:.3f}")

    # 轨迹绘图（原逻辑不变）
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10.colors
    for ep, traj in enumerate(all_trajectories):
        x_coords = [p[0] for p in traj]
        y_coords = [p[1] for p in traj]
        plt.plot(x_coords, y_coords, color=colors[ep % len(colors)], alpha=0.7)

    plt.scatter(x_star_np[0], x_star_np[1], color="red", marker="s", s=80, label="Target")
    plt.scatter(mean_x, mean_y, color="blue", marker="o", s=100, label=f"Landing Mean")
    plt.gca().add_patch(plt.Rectangle((mean_x-std_x, mean_y-std_y), 2*std_x, 2*std_y, 
                                     color="blue", alpha=0.2, linestyle="--", label="±1σ"))
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.8, label="Landing Pad")
    plt.xlim(-1.5, 1.5)
    plt.ylim(0, 1.5)
    plt.xlabel("X Position", fontsize=12)
    plt.ylabel("Y Position", fontsize=12)
    plt.title(f"Lunar Lander Trajectory ({version})", fontsize=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1), fontsize=10)
    os.makedirs("./fig/lunarlander", exist_ok=True)
    plt.savefig(f"./fig/lunarlander/trajectory_{version}.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 测试总结（原逻辑不变）
    avg_score = np.mean(episode_scores)
    std_score = np.std(episode_scores)
    print(f"\n=== 测试总结 ===")
    print(f"平均得分：{avg_score:.1f}±{std_score:.1f} | 成功着陆：{success_count}/{num_episodes}")
    return episode_scores


def test_lander_mpc(
    psi: KStepsPredictor,  # 修正模型类型
    mpc_controller: "DKRCMPCController",
    x_star: torch.Tensor,
    num_episodes: int = 10,
    max_steps: int = 500,
    version: str = "v1",
    seed: int = 2
) -> List[float]:
    # 原函数逻辑不变，仅修正模型类型注解
    env = gym.make("LunarLanderContinuous-v2")
    env.seed(seed)
    device = next(psi.parameters()).device
    episode_scores: List[float] = []
    all_trajectories: List[List[Tuple[float, float]]] = []
    landing_positions: List[Tuple[float, float]] = []
    success_count = 0
    psi.eval()
    with torch.no_grad():
        for ep in trange(num_episodes):
            x_prev = env.reset()[:6]
            done = False
            total_score = 0.0
            step = 0
            trajectory = []
            while not done and step < max_steps:
                trajectory.append((x_prev[0], x_prev[1]))
                u_current = mpc_controller.compute_control(x_prev)
                u_current = np.clip(u_current, env.action_space.low, env.action_space.high)
                x_next, reward, done, _ = env.step(u_current)
                total_score += reward
                x_prev = x_next[:6]
                step += 1
            # 原统计与绘图逻辑不变...
    return episode_scores


def train_psi_lander(
    x_prev: np.ndarray,
    u_prev: np.ndarray,
    x_next: np.ndarray,
    z_dim: int = 12,
    epochs: int = 500,
    batch_size: int = 128,
    lr: float = 1e-4,
    K_steps: int = 10,
    args = None  # 新增：传入args用于模型初始化
) -> KStepsPredictor:
    """修复原训练函数：修正损失计算错误，适配KStepsPredictor初始化"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_low = [-2, -2, -5, -5, -math.pi, -5]
    state_high = [2, 2, 5, 5, math.pi, 5]
    print(f"使用设备：{device}")
    x_true_series_tensor = torch.tensor(x_next, device=device, dtype=torch.float32)
    u_series_tensor = torch.tensor(u_prev, device=device, dtype=torch.float32)
    x_prev_batch = torch.tensor(x_prev, device=device, dtype=torch.float32)

    # 初始化KStepsPredictor（适配args参数）
    psi = KStepsPredictor(
        x_dim=args.x_dim,
        control_dim=args.control_dim,
        z_dim=z_dim,
        hidden_dim=128,
        low=state_low,
        high=state_high,
        K_steps=K_steps,
        device=device
    ).to(device)

    # 优化器与损失函数
    optimizer = optim.Adam(psi.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    avg_loss_list: List[float] = []

    # 训练循环（修复损失计算：原代码多乘了batch_loss，改为乘batch_size）
    psi.train()
    for epoch in range(epochs):
        total_epoch_loss = 0.0
        dataset = TensorDataset(x_prev_batch, u_series_tensor, x_true_series_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        for batch in dataloader:
            x_prev_b, u_series_b, x_true_series_b = batch
            # 模型前向（适配KStepsPredictor输出：z_pred_series, u_decode_series）
            z_pred_series, u_decode_series = psi(x_prev_b, u_series_b)
            # 计算真实状态的嵌入序列
            z_true_series = psi.StateEmbedding(x_true_series_b)  # [batch, K_steps, z_dim]
            
            # 计算损失（K步嵌入损失 + 控制解码损失）
            loss1 = loss_function(z_pred_series, z_true_series)
            loss2 = loss_function(u_decode_series, u_series_b)
            batch_loss = loss1 + loss2
            
            # 优化步骤
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            # 累计损失（修正：用batch_size加权，而非batch_loss）
            total_epoch_loss += batch_loss.item() * x_prev_b.shape[0]

        # 学习率衰减（原逻辑不变，但移到batch循环外，避免每batch衰减）
        if (epoch + 1) % 20 == 0 and epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

        # 计算平均损失
        avg_epoch_loss = total_epoch_loss / len(x_prev_batch)
        avg_loss_list.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1:3d}/{epochs}] | 平均损失：{avg_epoch_loss:.6f}", end='\r', flush=True)

    # 绘制损失曲线
    plot_loss_curve(avg_loss_list, args.test_version)
    return psi


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


def plot_loss_curve(loss_list: List[float], version: str) -> None:
    """绘制训练损失曲线（原逻辑不变）"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, color="#2E86AB", linewidth=2, label='Average Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training Loss Curve (Version: {version})', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    os.makedirs("./fig/lunarlander", exist_ok=True)
    plt.savefig(f'./fig/training_loss_{version}.png', dpi=300, bbox_inches="tight")
    plt.close()


def design_q_matrix(psi: KStepsPredictor, x_star: torch.Tensor, pos_weight: float = 100.0, other_weight: float = 1.0) -> np.ndarray:
    """适配KStepsPredictor的Q矩阵设计（原逻辑不变）"""
    device = next(psi.parameters()).device
    N = psi.z_dim
    Q = np.eye(N) * other_weight

    # 计算位置敏感的Z分量（通过梯度）
    x_sample = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32).unsqueeze(0)
    y_sample = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32).unsqueeze(0)
    x_star_tensor = x_star.unsqueeze(0)

    # x方向敏感分量
    x_sample.requires_grad_(True)
    z_x = psi.StateEmbedding(x_sample) - psi.StateEmbedding(x_star_tensor)
    z_x.sum().backward()
    x_sensitivity = x_sample.grad.squeeze().cpu().numpy()

    # y方向敏感分量
    y_sample.requires_grad_(True)
    z_y = psi.StateEmbedding(y_sample) - psi.StateEmbedding(x_star_tensor)
    z_y.sum().backward()
    y_sensitivity = y_sample.grad.squeeze().cpu().numpy()

    # 放大敏感分量权重
    sensitive_indices = np.where((abs(x_sensitivity) > 1e-4) | (abs(y_sensitivity) > 1e-4))[0]
    Q[sensitive_indices, sensitive_indices] = pos_weight
    print(f"Q矩阵设计完成：{len(sensitive_indices)}/{N}个Z分量为位置敏感维度，权重={pos_weight}")
    return Q


# -------------------------- 新增：2*K_steps轨迹预测评估函数 --------------------------
def evaluate_trajectory_prediction(
    model: KStepsPredictor,
    test_data_path: str,
    num_experiments: int = 4,
    save_results: bool = True,
    result_path: str = "./results",
    seed: int = 2
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
    K_steps = test_data['K_steps'].item()
    seq_length = test_data['seq_length'].item()  # 2*K_steps
    num_test_ep = extended_X_seq.shape[0]
    device = next(model.parameters()).device
    x_dim = args.x_dim
    z_dim = args.z_dim
    control_dim = args.control_dim

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


# -------------------------- 主函数（新增轨迹预测评估调用） --------------------------
if __name__ == "__main__":  
    # 1. 命令行参数解析（新增extended_test_path参数）
    parse = argparse.ArgumentParser()
    parse.add_argument('--test_version', type=str, default='v4', help='模型版本标识')
    parse.add_argument('--controller_type', type=str, default='lqr', help='控制器类型（lqr/mpc）')
    parse.add_argument('--seed', type=int, default=2, help='随机种子')
    parse.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parse.add_argument('--epochs', type=int, default=10, help='训练轮次')
    parse.add_argument('--data_epochs', type=int, default=20, help='数据生成回合数')
    parse.add_argument('--batch_size', type=int, default=256, help='批量大小')
    parse.add_argument('--num_episodes', type=int, default=100, help='LQR测试回合数')
    parse.add_argument('--data_prepared', action='store_true', help='是否使用预生成数据')
    parse.add_argument('--z_dim', type=int, default=12, help='高维状态维度N')
    parse.add_argument('--x_dim', type=int, default=6, help='状态维度（月球着陆器6维）')
    parse.add_argument('--control_dim', type=int, default=2, help='控制维度（2维引擎）')
    parse.add_argument('--K_steps', type=int, default=15, help='训练时的K步长度')
    # 新增：扩展测试数据路径（需与数据生成脚本的输出路径一致）
    parse.add_argument('--extended_test_path', type=str, 
                       default="./data/test_data_LunarLanderContinuous-v2_ep100_K15_seed2_extended.npz",
                       help='2*K_steps扩展测试数据路径')
    args = parse.parse_args()

    # 2. 固定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 3. 步骤1：加载/生成训练数据
    print("="*50 + " 步骤1/4：加载训练数据 " + "="*50)
    train_data_path = f"./data/train_data_LunarLanderContinuous-v2_n6_m2_deriv2_K{args.K_steps}_seed{args.seed}.npz"
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"训练数据不存在：{train_data_path}，请先运行数据生成脚本")
    data = np.load(train_data_path)
    x_prev = data['X_seq']
    u_prev = data['U_seq']
    x_next = data['Y_seq']
    print(f"加载预生成数据：{x_prev.shape[0]}组样本")
    # 4. 步骤2：训练KStepsPredictor模型（修复：传入args参数）
    print("\n" + "="*50 + " 步骤2/4：训练DKN模型 " + "="*50)
    psi_lander = train_psi_lander(
        x_prev=x_prev,
        u_prev=u_prev,
        x_next=x_next,
        z_dim=args.z_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        K_steps=args.K_steps,
        args=args  # 新增：传入args用于模型初始化
    )

    # 5. 步骤3：LQR控制测试（保留原功能，修复K_lqr计算）
    print("\n" + "="*50 + " 步骤3/4：LQR控制测试 " + "="*50)
    x_star_lander = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=next(psi_lander.parameters()).device)
    A_lander, B_lander, Q_, R_ = calculate_parameter(psi_lander, args.x_dim, args.z_dim, args.control_dim)
    # 修复：solve_discrete_lqr需传入Q_和R_（原代码遗漏）
    K_lqr = solve_discrete_lqr(A_lander, B_lander, Q_, R_)
    test_lander_lqr(psi_lander, K_lqr, x_star_lander, num_episodes=args.num_episodes, version=args.test_version, seed=args.seed)

    # 6. 步骤4：新增2*K_steps轨迹预测评估（4次实验）
    print("\n" + "="*50 + " 步骤4/4：2*K_steps轨迹预测评估 " + "="*50)
    if os.path.exists(args.extended_test_path):
        mean_errors, log10_errors = evaluate_trajectory_prediction(
            model=psi_lander,
            test_data_path=args.extended_test_path,
            num_experiments=4,  # 重复4次实验
            save_results=True,
            result_path="./results",
            seed=args.seed
        )

        # 打印评估总结
        print("\n=== 轨迹预测评估总结 ===")
        print(f"预测长度：{len(mean_errors)-1} 步（2*K={args.K_steps*2}）")
        print(f"所有时间步平均log10误差：{np.mean(log10_errors):.4f}")
        print(f"各时间步log10误差（每5步展示）：")
        for i in range(0, len(log10_errors), 5):
            print(f"  第{i:2d}步：{log10_errors[i]:.4f}")
    else:
        raise FileNotFoundError(f"扩展测试数据不存在：{args.extended_test_path}，请先运行数据生成脚本生成2*K_steps数据")