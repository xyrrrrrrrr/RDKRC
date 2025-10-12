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
# 假设MCDKN模型定义正确
from rdkrc.models.MCDKN import DKN_MC2
from rdkrc.trainer.loss_functions import ManifoldEmbLoss, ManifoldCtrlLoss
from rdkrc.controller.lqr_controller import solve_discrete_lqr


def train_mc_dkn_pendulum(
    X_train: torch.Tensor,  # [N, T, x_dim]
    U_train: torch.Tensor,  # [N, T, u_dim]
    x_dim: int = 3,
    u_dim: int = 1,
    batch_size: int = 128,
    epochs_stage1: int = 0,
    epochs_stage2: int = 100,
    lr: float = 1e-4,
    K_steps: int = 15,
    version: str = "mcdkn_v0",
    alpha: float = 0.4,
    beta: float = 0.2
) -> DKN_MC2:
    """训练Pendulum的MCDKN模型（带流形约束）"""
    # 状态/动作范围
    state_low = [-1.0, -1.0, -8.0]
    state_high = [1.0, 1.0, 8.0]
    action_low = [-2.0]
    action_high = [2.0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = DKN_MC2(
        x_dim=x_dim,
        u_dim=u_dim,
        hidden_dim=128,
        manifold_dim=16,
        control_manifold_dim=u_dim,
        state_low=state_low,
        state_high=state_high,
        action_low=action_low,
        action_high=action_high,
        device=device
    ).to(device)

    # 数据加载
    dataset = TensorDataset(X_train, U_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    k_step_loss = nn.MSELoss()
    manifold_emb_loss = ManifoldEmbLoss(k=20)  # 流形嵌入损失
    manifold_ctrl_loss = ManifoldCtrlLoss()    # 控制流形损失

    # 阶段1：基础预训练（无流形约束）
    print("阶段1：基础预训练...")
    model.train()
    pbar = trange(epochs_stage1)
    for epoch in pbar:
        total_loss = 0.0
        for batch in dataloader:
            batch_X, batch_U = batch[0].to(device), batch[1].to(device)
            x0 = batch_X[:, 0, :]  # 初始状态
            u_seq = batch_U.permute(1, 0, 2)  # [K_steps, batch, u_dim]
            
            # K步预测
            x_pred_seq = model.predict_k_steps(x0, u_seq, k=K_steps)
            x_pred_seq = x_pred_seq.permute(1, 0, 2)  # [batch, K_steps+1, x_dim]

            # 计算K步损失（带权重）
            loss_k = 0.0
            for i in range(1, K_steps):
                weight = 0.95 ** (i-1)  # 远期预测权重衰减
                loss_k += weight * k_step_loss(x_pred_seq[:, i, :], batch_X[:, i, :])
            pbar.set_postfix({"Epoch" : f"{epoch+1}/{epochs_stage1}","K-step Loss": f"{loss_k.item():.6f}"})
            optimizer.zero_grad()
            loss_k.backward()
            optimizer.step()
            total_loss += loss_k.item()

    # 阶段2：流形约束训练
    print("阶段2：流形约束训练...")
    pbar = trange(epochs_stage2)
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
                batch_U[:, :-1, :].reshape(-1, args.u_dim)
            )
            loss_ctrl = manifold_ctrl_loss(model.A, model.B, z_M_t, z_M_t1, g_phi_t, 
                                          batch_U[:, :-1, :].reshape(-1, args.u_dim))

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

        pbar.set_postfix({"Total Loss": f"{avg_total_loss:.6f}", "K-step Loss": f"{avg_k_loss:.6f}",
                          "Emb Loss": f"{avg_emb_loss:.6f}", "Ctrl Loss": f"{avg_ctrl_loss:.6f}"})
        if (epoch + 1) % 40 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

    return model


def test_mcdkn_pendulum_lqr(
    psi: DKN_MC2,
    K_lqr: np.ndarray,
    x_star: torch.Tensor,
    num_episodes: int = 10,
    max_steps: int = 200,
    version: str = "MCDKN-LQR",
    seed: int = 2
) -> List[float]:
    """MCDKN+LQR测试（复用DKN测试逻辑，适配模型接口）"""
    env = gym.make("Pendulum-v0")
    env.seed(seed)
    device = next(psi.parameters()).device
    episode_scores: List[float] = []
    all_trajectories: List[List[Tuple[float, float]]] = []
    success_count = 0

    print(f"\n[Test MCDKN-LQR] 开始测试...")
    psi.eval()
    with torch.no_grad():
        for ep in range(num_episodes):
            x_prev = env.reset()[:3]
            done = False
            total_score = 0.0
            step = 0
            trajectory = []

            while not done and step < max_steps:
                theta = np.arctan2(x_prev[1], x_prev[0])
                trajectory.append((theta, x_prev[2]))

                # MCDKN嵌入
                x_prev_tensor = torch.tensor(x_prev, device=device, dtype=torch.float32).unsqueeze(0)
                z_prev = psi.embed(x_prev_tensor) - psi.embed(x_star.unsqueeze(0))
                z_prev_np = z_prev.squeeze(0).cpu().numpy()

                # 控制计算
                u_t_ = -K_lqr @ z_prev_np.T
                u_t_ = torch.tensor(u_t_.T, device=device, dtype=torch.float32)
                g_phi_t = psi.forward_control(x_prev_tensor, u_t_)
                u_t = psi.inv_control_net(g_phi_t)[:, x_star.shape[0]:].squeeze(0).cpu().numpy()
                u_t = np.clip(u_t, -2.0, 2.0)

                x_next, reward, done, _ = env.step(u_t)
                total_score += reward
                x_prev = x_next[:3]
                step += 1

            # 成功判断
            final_theta = np.arctan2(x_prev[1], x_prev[0])
            if abs(final_theta) <= 0.1 and abs(x_prev[2]) <= 1.0:
                success_count += 1

            all_trajectories.append(trajectory)
            episode_scores.append(total_score)
            print(f"回合 {ep+1:2d} | 得分：{total_score:5.1f} | 最终θ：{final_theta:.3f} rad")
        # 测试总结
        avg_reward = np.mean(episode_scores)
        std_reward = np.std(episode_scores)
        print(f"\n[Test LQR] 测试总结：")
        print(f"  平均奖励：{avg_reward:.1f}±{std_reward:.1f} | 成功平衡：{success_count}/{num_episodes}")
    # 轨迹绘图与总结（同DKN测试，略）
    return episode_scores


def evaluate_trajectory_prediction(
    model: DKN_MC2,
    test_data_path: str,
    num_experiments: int = 4,
    save_results: bool = True,
    result_path: str = "./results/pendulum",
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
    K_steps = 15
    seq_length = 2*K_steps
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
    os.makedirs("./fig/pendulum", exist_ok=True)
    plot_path = os.path.join("./fig/pendulum", f"dkn_mc2_pred_errors_K{K_steps}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"误差曲线保存至：{plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_dim", type=int, default=3)
    parser.add_argument("--u_dim", type=int, default=1)
    parser.add_argument("--K_steps", type=int, default=15)
    args = parser.parse_args()

    # 加载训练数据（K步序列）
    data = np.load(f"./data/pendulum/train_data_Pendulum-v0_n3_m1_deriv2_K{args.K_steps}_seed2.npz")
    X_train = torch.tensor(data["X_seq"], dtype=torch.float32)  # [N, K, 3]
    U_train = torch.tensor(data["U_seq"], dtype=torch.float32)  # [N, K, 1]

    # 训练MCDKN模型
    model = train_mc_dkn_pendulum(
        X_train=X_train,
        U_train=U_train,
        x_dim=args.x_dim,
        u_dim=args.u_dim,
        K_steps=args.K_steps
    )

    # 测试
    x_star = torch.tensor([1.0, 0.0, 0.0], device=next(model.parameters()).device)
    A = model.A.weight
    B = model.B.weight
    K_lqr = solve_discrete_lqr(A=A, B=B)
    test_mcdkn_pendulum_lqr(
        psi=model,
        K_lqr=K_lqr,
        x_star=x_star,
        num_episodes=100
    )
    evaluate_trajectory_prediction(
        model=model,
        test_data_path="./data/pendulum/test_data_Pendulum-v0_n3_m1_K15_seed2_extended.npz"
    )