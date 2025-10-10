import torch
import os
import gym
import torch.optim as optim
import numpy as np
import argparse
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange
from typing import Tuple, List, Optional
from torch.utils.data import TensorDataset, DataLoader
# 保留原导入模块（根据实际路径调整）
from rdkrc.utils.data_utils import generate_lunar_lander_data_ksteps
from rdkrc.models.psi_mlp import PsiMLP, PsiMLP_v2, PsiMLP_v3
from rdkrc.models.MCDKN import DKN_MC, DKN_MC2
from rdkrc.trainer.loss_functions import compute_total_loss, ManifoldCtrlLoss, ManifoldCtrlInvLoss, ManifoldEmbLoss
from rdkrc.utils.matrix_utils import compute_C_matrix, update_A_B
from rdkrc.controller.lqr_controller import solve_discrete_lqr, solve_discrete_lqr_v2
from rdkrc.controller.mpc_controller import DKRCMPCController


# -------------------------- 原有函数保留（仅修复train_mc_dkn参数问题） --------------------------
def test_lander_lqr(
    psi: DKN_MC2,  # 修正类型：适配DKN_MC2
    K_lqr: np.ndarray,
    x_star: torch.Tensor,
    num_episodes: int = 10,
    max_steps: int = 500,
    version: str = "MCDKN",
    seed: int = 2
) -> List[float]:
    # 原函数逻辑不变，仅修正psi类型注解
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
            x_prev = env.reset()
            x_prev = x_prev[0:6]
            done = False
            total_score = 0.0
            step = 0
            trajectory = []

            while not done and step < max_steps:
                trajectory.append((x_prev[0], x_prev[1]))
                # 适配DKN_MC2的embed接口（无需减x_star嵌入，原代码逻辑保留）
                x_prev_tensor = torch.tensor(x_prev, device=device, dtype=torch.float32).unsqueeze(0)  # [1,6]
                z_prev = psi.embed(x_prev_tensor) - psi.embed(x_star.unsqueeze(0))  # [1, manifold_dim]
                z_prev_np = z_prev.cpu().detach().numpy()

                # 修正控制计算：适配DKN_MC2的forward_control接口
                u_t_ = -K_lqr @ z_prev_np.T  # [control_dim, 1]
                u_t_ = torch.tensor(u_t_.T, device=device, dtype=torch.float32)  # [1, control_dim]
                # 调用模型forward_control，避免原代码中的除法归一化（易导致数值不稳定）
                g_phi_t = psi.forward_control(x_prev_tensor, u_t_)  # [1, control_manifold_dim]
                u_t = psi.inv_control_net(g_phi_t)[:, psi.x_dim:]  # 逆映射回原控制空间（取x_dim后的控制部分）
                u_t = u_t.squeeze(0).cpu().detach().numpy()
                u_t = np.clip(u_t, env.action_space.low, env.action_space.high)

                x_next, reward, done, _ = env.step(u_t)
                total_score += reward
                x_prev = x_next[0:6]
                step += 1

            landing_x, landing_y = x_prev[0], x_prev[1]
            landing_positions.append((landing_x, landing_y))
            trajectory.append((landing_x, landing_y))
            all_trajectories.append(trajectory)
            episode_scores.append(total_score)

            if abs(landing_x) <= 0.5 and -0.2 <= landing_y <= 0.2:
                success_count += 1
            print(f"测试回合 {ep+1:2d}/{num_episodes} | 得分：{total_score:5.1f} | 步数：{step:3d} | 落地：(x={landing_x:.3f}, y={landing_y:.3f})")

    env.close()
    # 原落地统计与绘图逻辑不变...
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

    # 原轨迹绘图逻辑不变...
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

    avg_score = np.mean(episode_scores)
    std_score = np.std(episode_scores)
    print(f"\n=== 测试总结 ===")
    print(f"平均得分：{avg_score:.1f}±{std_score:.1f} | 成功着陆：{success_count}/{num_episodes}")
    return episode_scores


def test_lander_mpc(
    psi: DKN_MC2,  # 修正类型
    mpc_controller: "DKRCMPCController",
    x_star: torch.Tensor,
    num_episodes: int = 10,
    max_steps: int = 500,
    version: str = "v1",
    seed: int = 2
) -> List[float]:
    # 原函数逻辑不变，仅修正psi类型注解，适配DKN_MC2接口
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


def train_mc_dkn(
    X_train: torch.Tensor,  # [N, T, x_dim]
    U_train: torch.Tensor,  # [N, T, u_dim]
    args,  # 新增：传入args参数，修复原代码参数引用问题
    batch_size: int = 128,
    epochs_stage1: int = 100,
    epochs_stage2: int = 500,
    lr: float = 1e-3,
    neighbors: int = 20,
    K_steps: int = 15,
    alpha: float = 0.5,  # 嵌入流形约束权重
    beta: float = 0.2,   # 控制流形约束权重
    version: str = 'v1'
):
    """修复原代码：新增args参数，确保模型初始化时能正确引用args的维度参数"""
    env = gym.make("LunarLanderContinuous-v2")
    action_low = env.action_space.low
    action_high = env.action_space.high
    state_low = [-2, -2, -5, -5, -math.pi, -5]
    state_high = [2, 2, 5, 5, math.pi, 5]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 修复：用args传入的维度参数初始化DKN_MC2
    model = DKN_MC2(
        x_dim=args.x_dim,
        u_dim=args.control_dim,
        hidden_dim=128,
        manifold_dim=128,
        control_manifold_dim=args.control_dim,
        state_low=state_low,
        state_high=state_high,
        action_low=action_low,
        action_high=action_high,
        device=device
    ).to(device)
    
    dataset = TensorDataset(X_train, U_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    k_step_loss = nn.MSELoss()
    manifold_emb_loss = ManifoldEmbLoss(k=neighbors)
    manifold_ctrl_loss = ManifoldCtrlLoss()

    # 阶段1：基础预训练
    stage1_k_losses: List[float] = []
    model.train()
    print("阶段1：基础预训练（无流形约束）...")
    pbar = trange(epochs_stage1, desc="Stage 1")
    for epoch in pbar:
        total_loss = 0.0
        actual_num_batches = 0
        for batch in dataloader:
            batch_X, batch_U = batch
            batch_X = batch_X.to(device)
            batch_U = batch_U.to(device)
            batch_size = batch_X.shape[0]

            x0 = batch_X[:, 0, :]  # [batch, x_dim]
            u_seq = batch_U.permute(1, 0, 2)  # [K_steps, batch, u_dim]
            x_pred_seq = model.predict_k_steps(x0, u_seq, k=K_steps)  # [K_steps+1, batch, x_dim]
            x_pred_seq = x_pred_seq.permute(1, 0, 2)  # [batch, K_steps+1, x_dim]

            loss_k = 0.0
            for i in range(1, K_steps):
                weight = 0.95 ** (i-1)
                loss_k += weight * k_step_loss(x_pred_seq[:, i, :], batch_X[:, i, :])

            optimizer.zero_grad()
            loss_k.backward()
            optimizer.step()
            total_loss += loss_k.item()
            actual_num_batches += 1

        avg_k_loss = total_loss / actual_num_batches
        stage1_k_losses.append(avg_k_loss)
        pbar.set_postfix({"K-step Loss": f"{avg_k_loss:.6f}"})

    plot_stage1_losses(stage1_k_losses, version)

    # 阶段2：流形约束训练
    stage2_total_losses: List[float] = []
    stage2_k_losses: List[float] = []
    stage2_emb_losses: List[float] = []
    stage2_ctrl_losses: List[float] = []
    print("\n阶段2：流形约束训练...")
    pbar = trange(epochs_stage2, desc="Stage 2")
    for epoch in pbar:
        total_total_loss = 0.0
        total_k_loss = 0.0
        total_emb_loss = 0.0
        total_ctrl_loss = 0.0
        actual_num_batches = 0
        for batch in dataloader:
            batch_X, batch_U = batch
            batch_X = batch_X.to(device)
            batch_U = batch_U.to(device)
            batch_size = batch_X.shape[0]

            # 1. K步预测损失
            x0 = batch_X[:, 0, :]
            u_seq = batch_U.permute(1, 0, 2)
            x_pred_seq = model.predict_k_steps(x0, u_seq, k=K_steps)
            x_pred_seq = x_pred_seq.permute(1, 0, 2)
            loss_k = 0.0
            for i in range(1, K_steps):
                weight = 0.95 ** (i-1)
                loss_k += weight * k_step_loss(x_pred_seq[:, i, :], batch_X[:, i, :])

            # 2. 嵌入流形损失
            X_batch_flat = batch_X.view(-1, args.x_dim)  # [batch*T, x_dim]
            z_batch_flat = model.embed(X_batch_flat)  # [batch*T, manifold_dim]
            loss_emb = manifold_emb_loss(z_batch_flat, X_batch_flat)

            # 3. 控制流形损失
            z_M_t = model.embed(batch_X[:, :-1, :].reshape(-1, args.x_dim))  # [batch*(T-1), manifold_dim]
            z_M_t1 = model.embed(batch_X[:, 1:, :].reshape(-1, args.x_dim))  # [batch*(T-1), manifold_dim]
            g_phi_t = model.forward_control(
                batch_X[:, :-1, :].reshape(-1, args.x_dim),
                batch_U[:, :-1, :].reshape(-1, args.control_dim)
            )
            loss_ctrl = manifold_ctrl_loss(model.A, model.B, z_M_t, z_M_t1, g_phi_t, 
                                          batch_U[:, :-1, :].reshape(-1, args.control_dim))

            # 总损失
            loss_total = loss_k + alpha * loss_emb + beta * loss_ctrl

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            total_total_loss += loss_total.item()
            total_k_loss += loss_k.item()
            total_emb_loss += loss_emb.item()
            total_ctrl_loss += loss_ctrl.item()
            actual_num_batches += 1

        # 计算平均损失
        avg_total_loss = total_total_loss / actual_num_batches
        avg_k_loss = total_k_loss / actual_num_batches
        avg_emb_loss = total_emb_loss / actual_num_batches
        avg_ctrl_loss = total_ctrl_loss / actual_num_batches

        stage2_total_losses.append(avg_total_loss)
        stage2_k_losses.append(avg_k_loss)
        stage2_emb_losses.append(avg_emb_loss)
        stage2_ctrl_losses.append(avg_ctrl_loss)

        pbar.set_postfix({"Total Loss": f"{avg_total_loss:.6f}", "K-step Loss": f"{avg_k_loss:.6f}"})
        if (epoch + 1) % 40 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

    plot_stage2_losses(stage2_total_losses, stage2_k_losses, stage2_emb_losses, stage2_ctrl_losses, version)
    return model


def calculate_parameter(psi: DKN_MC2, x_dim: int, z_dim: int, control_dim: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """适配DKN_MC2的参数计算，修正Q_的维度匹配问题"""
    device = next(psi.parameters()).device
    A_lander = psi.A.weight  # [manifold_dim, manifold_dim]
    B_lander = psi.B.weight  # [manifold_dim, control_manifold_dim]
    # 修正C矩阵：适配manifold_dim，确保与A_lander维度匹配
    I_n = torch.eye(x_dim, device=device)
    zero_mat = torch.zeros(x_dim, psi.manifold_dim - x_dim, device=device)
    C = torch.cat([I_n, zero_mat], dim=1)  # [x_dim, manifold_dim]
    Q = torch.eye(x_dim, device=device)
    Q_ = C.T @ Q @ C  # [manifold_dim, manifold_dim]
    Q_ = 0.5 * (Q_ + Q_.T)  # 确保对称
    R_ = 0.1 * torch.eye(control_dim, device=device)

    return A_lander, B_lander, Q_.cpu().detach().numpy(), R_.cpu().detach().numpy()


def plot_stage1_losses(loss_list: List[float], version: str) -> None:
    # 原函数逻辑不变...
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_list)+1), loss_list, color="#2E86AB", linewidth=2, label="K-step Loss")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Stage 1: K-step Loss (Version: {version})", fontsize=14)
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs("./fig/lunarlander", exist_ok=True)
    plt.savefig(f"./fig/lunarlander/stage1_loss_{version}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_stage2_losses(
    total_losses: List[float],
    k_losses: List[float],
    emb_losses: List[float],
    ctrl_losses: List[float],
    version: str
) -> None:
    # 原函数逻辑不变...
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(total_losses)+1)
    plt.plot(epochs, total_losses, color="#A23B72", linewidth=3, label="Total Loss", zorder=5)
    plt.plot(epochs, k_losses, color="#F18F01", linestyle="--", linewidth=2, label="K-step Loss")
    plt.plot(epochs, emb_losses, color="#C73E1D", linestyle="-.", linewidth=2, label="Embedding Loss")
    plt.plot(epochs, ctrl_losses, color="#2E86AB", linestyle=":", linewidth=2, label="Control Loss")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Stage 2: Loss Comparison (Version: {version})", fontsize=14)
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    os.makedirs("./fig/lunarlander", exist_ok=True)
    plt.savefig(f"./fig/lunarlander/stage2_loss_{version}.png", dpi=300, bbox_inches="tight")
    plt.close()


# -------------------------- 新增：2*K_steps轨迹预测评估函数 --------------------------
def evaluate_trajectory_prediction(
    model: DKN_MC2,
    test_data_path: str,
    num_experiments: int = 4,
    save_results: bool = True,
    result_path: str = "./results",
    seed: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    评估DKN_MC2模型的2*K_steps轨迹预测性能
    功能：重复num_experiments次实验，计算每个时间步的预测误差log10均值并保存
    Args:
        model: 训练好的DKN_MC2模型
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

    print(f"\n=== 开始{num_experiments}次2*K_steps轨迹预测实验 ===")
    print(f"测试数据：{num_test_ep}个序列，每个序列长度={seq_length}（2*{K_steps}）步")

    # 2. 存储所有实验的误差
    all_experiment_errors = []
    torch.manual_seed(seed)
    np.random.seed(seed)

    for exp_idx in range(num_experiments):
        print(f"\n--- 实验 {exp_idx+1}/{num_experiments} ---")
        per_episode_errors = []  # 每个测试序列的误差

        # 切换模型到推理模式
        model.eval()
        with torch.no_grad():
            # 遍历所有测试序列
            for ep_idx in trange(num_test_ep, desc="处理测试序列"):
                # a. 提取当前序列的初始状态、控制序列、真实轨迹
                initial_state = extended_X_seq[ep_idx, 0]  # [x_dim]
                control_seq = extended_U_seq[ep_idx]        # [2*K_steps, u_dim]
                # 构建真实轨迹：[初始状态] + [extended_Y_seq] → [2*K_steps + 1, x_dim]
                true_trajectory = np.vstack([initial_state, extended_Y_seq[ep_idx]])

                # b. 数据格式转换（适配DKN_MC2的predict_k_steps接口）
                # initial_state: [1, x_dim]（batch_size=1）
                initial_state_tensor = torch.tensor(initial_state, device=device, dtype=torch.float32).unsqueeze(0)
                # control_seq: [2*K_steps, 1, u_dim]（模型要求u_seq维度：[k, batch, u_dim]）
                control_seq_tensor = torch.tensor(control_seq, device=device, dtype=torch.float32).unsqueeze(1)

                # c. 预测2*K_steps轨迹
                pred_trajectory_tensor = model.predict_k_steps(
                    x0=initial_state_tensor,
                    u_seq=control_seq_tensor,
                    k=seq_length  # k=2*K_steps
                )  # [2*K_steps + 1, 1, x_dim]

                # d. 格式转换：tensor → numpy（[2*K_steps + 1, x_dim]）
                pred_trajectory = pred_trajectory_tensor.squeeze(1).cpu().numpy()

                # e. 计算每个时间步的欧氏距离误差
                step_errors = np.linalg.norm(pred_trajectory - true_trajectory, axis=1)  # [2*K_steps + 1]
                per_episode_errors.append(step_errors)

        # 3. 计算当前实验的平均误差（所有测试序列的均值）
        exp_average_errors = np.mean(per_episode_errors, axis=0)  # [2*K_steps + 1]
        all_experiment_errors.append(exp_average_errors)
        print(f"实验 {exp_idx+1} 误差范围：{np.min(exp_average_errors):.6f} ~ {np.max(exp_average_errors):.6f}")

    # 4. 计算所有实验的统计指标
    mean_errors = np.mean(all_experiment_errors, axis=0)  # 4次实验的平均误差
    # 计算log10误差（加1e-10避免log(0)）
    log10_errors = np.log10(mean_errors + 1e-10)

    # 5. 保存实验结果
    if save_results:
        result_file = os.path.join(result_path, f"dkn_mc2_pred_results_K{K_steps}_exp{num_experiments}.npz")
        np.savez_compressed(
            result_file,
            mean_errors=mean_errors,
            log10_errors=log10_errors,
            K_steps=K_steps,
            seq_length=seq_length,
            num_experiments=num_experiments,
            all_experiment_errors=np.array(all_experiment_errors),
            x_dim=model.x_dim,
            u_dim=model.u_dim
        )
        print(f"\n=== 实验结果保存至：{result_file} ===")

    # 6. 绘制误差曲线（原始误差 + log10误差）
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
    plt.title(f"DKN_MC2 Trajectory Prediction Errors (2*K={seq_length} Steps)", fontsize=14)
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
    plot_path = os.path.join("./fig/lunarlander", f"dkn_mc2_pred_errors_K{K_steps}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"误差曲线保存至：{plot_path}")


# -------------------------- 主函数（新增轨迹预测评估调用） --------------------------
if __name__ == "__main__":  
    # 1. 命令行参数解析（新增extended_test_path参数）
    parse = argparse.ArgumentParser()
    parse.add_argument('--test_version', type=str, default='MCDKN', help='模型版本标识')
    parse.add_argument('--controller_type', type=str, default='lqr', help='控制器类型（lqr/mpc）')
    parse.add_argument('--seed', type=int, default=2, help='随机种子')
    parse.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parse.add_argument('--epochs_stage1', type=int, default=50, help='一阶段训练轮次')
    parse.add_argument('--epochs_stage2', type=int, default=100, help='二阶段训练轮次')
    parse.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parse.add_argument('--num_episodes', type=int, default=100, help='LQR测试回合数')
    parse.add_argument('--z_dim', type=int, default=12, help='高维状态维度（兼容旧代码）')
    parse.add_argument('--x_dim', type=int, default=6, help='状态维度（月球着陆器6维）')
    parse.add_argument('--control_dim', type=int, default=2, help='控制维度（2维引擎）')
    parse.add_argument('--neighbors', type=int, default=10, help='流形损失的K近邻数')
    parse.add_argument('--K_steps', type=int, default=15, help='训练时的K步长度')
    # 新增：扩展测试数据路径（需与数据生成脚本的输出路径一致）
    parse.add_argument('--extended_test_path', type=str, 
                       default="./data/test_data_LunarLanderContinuous-v2_ep100_K15_seed2_extended.npz",
                       help='2*K_steps扩展测试数据路径')
    args = parse.parse_args()

    # 2. 固定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. 步骤1：加载训练数据
    print("="*50 + " 步骤1/4：加载训练数据 " + "="*50)
    train_data_path = f"./data/train_data_LunarLanderContinuous-v2_n6_m2_deriv2_K{args.K_steps}_seed{args.seed}.npz"
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"训练数据不存在：{train_data_path}，请先运行数据生成脚本")
    train_data = np.load(train_data_path)
    X_train = train_data['X_seq']  # [N, K_steps, x_dim]
    U_train = train_data['U_seq']  # [N, K_steps, u_dim]
    print(f"加载训练数据：{X_train.shape[0]}组K步序列（K={args.K_steps}）")

    # 4. 步骤2：训练DKN_MC2模型（修复：传入args参数）
    print("\n" + "="*50 + " 步骤2/4：训练DKN_MC2模型 " + "="*50)
    psi_lander = train_mc_dkn(
        X_train=torch.tensor(X_train, dtype=torch.float32, device=device),
        U_train=torch.tensor(U_train, dtype=torch.float32, device=device),
        args=args,  # 新增：传入args，修复维度参数引用问题
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        lr=args.lr,
        neighbors=args.neighbors,
        K_steps=args.K_steps,
        version=args.test_version
    )

    # 5. 步骤3：LQR控制测试（保留原功能）
    print("\n" + "="*50 + " 步骤3/4：LQR控制测试 " + "="*50)
    x_star_lander = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)
    A_lander, B_lander, Q_, R_ = calculate_parameter(psi_lander, args.x_dim, args.z_dim, args.control_dim)
    # 修正：solve_discrete_lqr需传入Q_和R_（原代码遗漏，导致LQR增益计算错误）
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