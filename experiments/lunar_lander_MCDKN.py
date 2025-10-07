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
from typing import Tuple, List
from torch.utils.data import TensorDataset, DataLoader
from rdkrc.utils.data_utils import generate_lunar_lander_data_ksteps
from rdkrc.models.psi_mlp import PsiMLP, PsiMLP_v2, PsiMLP_v3
from rdkrc.models.MCDKN import DKN_MC
from rdkrc.trainer.loss_functions import compute_total_loss, ManifoldCtrlLoss, ManifoldEmbLoss
from rdkrc.utils.matrix_utils import compute_C_matrix, update_A_B
from rdkrc.controller.lqr_controller import solve_discrete_lqr, solve_discrete_lqr_v2
from rdkrc.controller.mpc_controller import DKRCMPCController


def test_lander_lqr(
    psi: PsiMLP,
    K_lqr: np.ndarray,
    x_star: torch.Tensor,
    num_episodes: int = 10,
    max_steps: int = 500,
    version: str = "MCDKN",
    seed: int = 2
) -> List[float]:
    """
    月球着陆器LQR控制测试（含落地位置均值/方差统计与轨迹汇总图）
    依据文档IV.D节：通过10次独立测试验证DKRC鲁棒性，新增落地位置统计以量化着陆精度（🔶1-83、🔶1-87）。
    
    Args:
        psi: 训练好的PsiMLP网络（含u₀参数，文档II.36节）
        K_lqr: LQR控制增益，shape=[2, 256]（文档III节离散LQR求解）
        x_star: 目标状态（着陆区，文档IV.D节定义：x、y对应着陆位置），shape=[6]
        num_episodes: 测试回合数（文档指定10次，确保统计鲁棒性）
        max_steps: 每回合最大步数（避免无限循环，文档未指定时默认500）
        version: PsiMLP版本标识（用于区分结果文件，不影响算法逻辑）
        seed: 随机种子（确保结果可复现，文档IV.D节隐含要求）
    Returns:
        episode_scores: 每回合得分列表
    """

    env = gym.make("LunarLanderContinuous-v2")
    env.seed(seed)
    device = next(psi.parameters()).device
    episode_scores: List[float] = []
    all_trajectories: List[List[Tuple[float, float]]] = []  # 存储所有episode的x-y轨迹（文档核心位置维度，🔶1-80）
    landing_positions: List[Tuple[float, float]] = []  # 新增：存储所有episode的落地位置（最终x-y坐标）
    success_count = 0  # 成功着陆计数（文档IV.D节隐含评估标准：x∈[-0.5,0.5]且y∈[0,0.1]）
    psi.eval()  # 推理模式（禁用梯度，文档测试阶段要求，🔶1-28）
    with torch.no_grad():
        for ep in range(num_episodes):
            # 初始化环境（文档IV.D节：随机初始扰动，确保测试鲁棒性）
            x_prev = env.reset()
            x_prev = x_prev[0:6]  # 取文档定义的6维状态（x,y,θ,ẋ,ẏ,θ̇），仅x-y用于轨迹与落地统计（🔶1-80）
            done = False
            total_score = 0.0
            step = 0
            trajectory = []  # 记录当前episode的x-y轨迹（文档图8核心维度，🔶1-87）

            while not done and step < max_steps:
                # 记录当前位置（仅保留文档关注的x-y维度，🔶1-80、🔶1-87）
                trajectory.append((x_prev[0], x_prev[1]))
                # 1. 计算高维线性状态z（文档Equation 4：z=Ψ(x)-Ψ(x*)，核心线性化步骤）
                x_prev_tensor = torch.tensor(x_prev, device=device, dtype=torch.float32)
                z_prev = psi.embed(x_prev_tensor) - psi.embed(x_star)
                z_prev_np = z_prev.cpu().detach().numpy()

                # 2. 计算LQR控制输入（文档III节：v_t=-K_lqr z_t，u_t=v_t+u₀，控制律设计）
                u_t_ = -K_lqr @ z_prev_np.T  # 变换后控制输入（适配高维线性模型）
                u_t_ = torch.tensor(u_t_.T, device=device, dtype=torch.float32)
                # u_t = psi.decode_control(u_t_)[6: ].cpu().detach().numpy()
                u_t = psi.forward_inv_control(x_prev_tensor, u_t_).squeeze(0).cpu().detach().numpy()
                u_t = np.clip(u_t, env.action_space.low, env.action_space.high)  # 文档隐含控制约束（物理执行器限制）
                # 3. 环境交互（文档IV.D节：获取下一状态与奖励，完成状态迭代）
                x_next, reward, done, _ = env.step(u_t)
                total_score += reward
                x_prev = x_next[0:6]
                step += 1

            # 记录当前episode的落地位置（最终x-y坐标，文档关注的着陆精度核心指标）
            landing_x = x_prev[0]
            landing_y = x_prev[1]
            landing_positions.append((landing_x, landing_y))
            # 记录最终位置以完善轨迹（确保“初始→落地”完整路径，文档图8要求，🔶1-87）
            trajectory.append((landing_x, landing_y))
            all_trajectories.append(trajectory)
            episode_scores.append(total_score)

            # 成功着陆判断（文档IV.D节隐含评估标准：落地位置在着陆区附近）
            if abs(landing_x) <= 0.5 and -0.2 <= landing_y <= 0.2:
                success_count += 1
            print(f"测试回合 {ep+1:2d}/{num_episodes} | 得分：{total_score:5.1f} | 步数：{step:3d} | 落地位置：(x={landing_x:.3f}, y={landing_y:.3f})")

    env.close()
    # -------------------------- 新增：落地位置均值/方差计算（文档IV.D节量化评估延伸） --------------------------
    # 提取落地位置的x、y坐标数组
    landing_xs = np.array([pos[0] for pos in landing_positions], dtype=np.float32)
    landing_ys = np.array([pos[1] for pos in landing_positions], dtype=np.float32)
    # 计算均值（反映落地位置的集中趋势，量化着陆精度）
    mean_x = np.mean(landing_xs)
    mean_y = np.mean(landing_ys)
    # 计算方差（反映落地位置的离散程度，量化DKRC鲁棒性，文档IV.D节“多回合一致性”要求）
    var_x = np.var(landing_xs, ddof=1)  # ddof=1：样本方差（适配有限测试回合，更贴合文档10次测试场景）
    var_y = np.var(landing_ys, ddof=1)
    # 计算标准差（便于图表标注，直观反映离散范围）
    std_x = np.sqrt(var_x)
    std_y = np.sqrt(var_y)

    # -------------------------- 新增：落地位置统计结果打印（对齐文档评估报告风格） --------------------------
    x_star_np = x_star.cpu().numpy()  # 目标状态（文档IV.D节着陆区，🔶1-80）
    print(f"\n=== 落地位置统计结果（文档IV.D节量化评估） ===")
    print(f"目标着陆位置（x_star）：(x={x_star_np[0]:.3f}, y={x_star_np[1]:.3f})")
    print(f"实际落地位置均值：(x={mean_x:.3f}, y={mean_y:.3f})")
    print(f"实际落地位置方差（样本方差）：var_x={var_x:.6f}, var_y={var_y:.6f}")
    print(f"实际落地位置标准差：std_x={std_x:.3f}, std_y={std_y:.3f}")
    print(f"落地位置相对于目标的偏移：Δx={mean_x - x_star_np[0]:.3f}, Δy={mean_y - x_star_np[1]:.3f}")

    # -------------------------- 轨迹汇总图绘制（含落地位置均值/方差标注，严格对齐文档图8） --------------------------
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10.colors  # 多轨迹颜色区分（避免重叠遮挡，文档图8风格，🔶1-87）

    # 1. 绘制所有episode的轨迹（文档图8核心内容）
    for ep, trajectory in enumerate(all_trajectories):
        x_coords = [p[0] for p in trajectory]
        y_coords = [p[1] for p in trajectory]
        color = colors[ep % len(colors)]  # 循环分配颜色，适配文档10次测试回合
        plt.plot(x_coords, y_coords, color=color, alpha=0.7)

    # 2. 标注目标着陆位置（文档IV.D节定义的着陆区，🔶1-80）
    plt.scatter(
        x_star_np[0], x_star_np[1], 
        color="red", marker="s", s=80, edgecolor="black", 
        label=f"Target Landing Pos (x={x_star_np[0]:.1f}, y={x_star_np[1]:.1f})"
    )

    # 3. 新增：标注落地位置均值（反映集中趋势，文档量化评估可视化）
    plt.scatter(
        mean_x, mean_y, 
        color="blue", marker="o", s=100, edgecolor="black", 
        label=f"Landing Mean (x={mean_x:.3f}, y={mean_y:.3f})"
    )

    # 4. 新增：标注落地位置方差范围（用矩形框表示±1倍标准差，直观反映离散程度）
    # x方向范围：mean_x ± std_x，y方向范围：mean_y ± std_y
    plt.gca().add_patch(
        plt.Rectangle(
            (mean_x - std_x, mean_y - std_y),  # 矩形左下角
            2 * std_x, 2 * std_y,  # 矩形宽（2*std_x）、高（2*std_y）
            color="blue", alpha=0.2, edgecolor="blue", linestyle="--",
            label=f"Landing Std Range (±1σ)"
        )
    )

    # 5. 标注着陆区（文档IV.D节：着陆平台位置，y对应目标高度，🔶1-82）
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.8, label="Landing Pad (y=0)")

    # 6. 坐标轴设置（匹配文档状态空间：x∈[-1.5,1.5]，y∈[0,1.5]，🔶1-80）
    plt.xlim(-1.5, 1.5)
    plt.ylim(0, 1.5)

    # 7. 标签与标题（文档图8规范：明确位置维度与实验对象，🔶1-87）
    plt.xlabel("X Position (Horizontal)", fontsize=12)
    plt.ylabel("Y Position (Altitude)", fontsize=12)
    if version == "v1":
        plt.title("Lunar Lander Trajectory Summary (DKRC + LQR) with Landing Stats", fontsize=14)
    elif version == "v2":
        plt.title("Lunar Lander Trajectory Summary (RDKRC + LQR) with Landing Stats", fontsize=14)
    elif version == "v3":
        plt.title("Lunar Lander Trajectory Summary (RRDKRC + LQR) with Landing Stats", fontsize=14)

    # 8. 图例（避免遮挡轨迹，文档图8右侧布局，包含新增的均值/方差标注）
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1), fontsize=10)
    plt.grid(True, alpha=0.5)

    # 9. 保存汇总图（确保完整显示图例，符合文档实验结果保存要求，🔶1-87）
    plt.savefig(f"./fig/lunar_lander_trajectory_summary_{version}_with_stats.png", bbox_inches="tight", dpi=300)
    plt.close()

    # -------------------------- 测试结果总统计（文档IV.D节评估标准，补充均值/方差信息） --------------------------
    avg_score = np.mean(episode_scores)
    std_score = np.std(episode_scores)
    print(f"\n=== 测试总总结（文档IV.D节评估框架） ===")
    print(f"平均得分：{avg_score:.1f}±{std_score:.1f} | 成功着陆：{success_count}/{num_episodes} 次")
    print(f"落地位置均值：(x={mean_x:.3f}, y={mean_y:.3f}) | 落地位置标准差：(x={std_x:.3f}, y={std_y:.3f})")

    return episode_scores


def test_lander_mpc(
    psi: PsiMLP,
    mpc_controller: "DKRCMPCController",  # MPC控制器实例（替换LQR的增益矩阵K_lqr）
    x_star: torch.Tensor,
    num_episodes: int = 10,
    max_steps: int = 500,
    version: str = "v1",
    seed: int = 2
) -> List[float]:
    """
    月球着陆器MPC控制测试（含落地位置均值/方差统计与轨迹汇总图）
    依据文档III节“Koopman-based MPC”与IV.D节：通过10次独立测试验证MPC鲁棒性，量化着陆精度（🔶1-45、🔶1-83、🔶1-87）。
    
    Args:
        psi: 训练好的PsiMLP网络（MPC控制器内部依赖其计算高维状态z，文档II.36节）
        mpc_controller: DKRCMPCController实例（封装MPC优化逻辑，文档III节）
        x_star: 目标状态（着陆区，文档IV.D节定义：x、y对应着陆位置），shape=[6]
        num_episodes: 测试回合数（文档指定10次，确保统计鲁棒性）
        max_steps: 每回合最大步数（避免无限循环，文档未指定时默认500）
        version: PsiMLP版本标识（用于区分结果文件，不影响算法逻辑）
        seed: 随机种子（确保结果可复现，文档IV.D节隐含要求）
    Returns:
        episode_scores: 每回合得分列表（Gym内置得分，>200为成功着陆，文档IV.D节评估标准）
    """
    env = gym.make("LunarLanderContinuous-v2")
    env.seed(seed)  # 固定随机种子，确保测试可复现（文档实验可复现性隐含要求）
    device = next(psi.parameters()).device
    episode_scores: List[float] = []
    all_trajectories: List[List[Tuple[float, float]]] = []  # 存储所有episode的x-y轨迹（文档核心位置维度，🔶1-80）
    landing_positions: List[Tuple[float, float]] = []  # 存储所有episode的落地位置（最终x-y坐标，量化精度核心）
    success_count = 0  # 成功着陆计数（文档IV.D节隐含评估标准：x∈[-0.5,0.5]且y∈[0,0.1]）

    psi.eval()  # 推理模式（禁用梯度，文档测试阶段要求，🔶1-28）
    with torch.no_grad():
        for ep in trange(num_episodes):
            # 初始化环境（文档IV.D节：随机初始扰动，验证MPC对扰动的鲁棒性）
            x_prev = env.reset()  # Gym接口：返回初始状态（含随机位置/速度扰动）
            x_prev = x_prev[0:6]  # 取文档定义的6维状态（x,y,θ,ẋ,ẏ,θ̇），仅x-y用于轨迹与落地统计（🔶1-80）
            done = False
            total_score = 0.0
            step = 0
            trajectory = []  # 记录当前episode的x-y轨迹（文档图8核心维度，直观展示路径，🔶1-87）

            while not done and step < max_steps:
                # 记录当前位置（仅保留文档关注的x-y维度，忽略姿态/速度，聚焦着陆位置，🔶1-80、🔶1-87）
                trajectory.append((x_prev[0], x_prev[1]))

                # 1. 计算MPC最优控制输入（核心差异：替换LQR的增益矩阵计算，文档III节MPC逻辑）
                # MPC控制器直接接收原状态x_prev，内部自动完成高维状态z计算与优化（封装文档Equation 5与11）
                u_current = mpc_controller.compute_control(x_prev)  # shape=[2]（主引擎+侧引擎，🔶1-80）

                # 2. 控制输入双重裁剪（确保在环境动作空间内，MPC内部已裁剪，此处双重保险符合文档物理约束，🔶1-82）
                u_current = np.clip(u_current, env.action_space.low, env.action_space.high)

                # 3. 环境交互（文档IV.D节：获取下一状态与奖励，完成状态迭代，与LQR测试逻辑完全一致）
                x_next, reward, done, _ = env.step(u_current)
                total_score += reward
                x_prev = x_next[0:6]  # 更新状态，保留前6维核心状态
                step += 1

            # 记录当前episode的关键结果（落地位置+完整轨迹）
            landing_x, landing_y = x_prev[0], x_prev[1]
            landing_positions.append((landing_x, landing_y))
            trajectory.append((landing_x, landing_y))  # 补充最终落地位置，确保轨迹完整（文档图8要求，🔶1-87）
            all_trajectories.append(trajectory)
            episode_scores.append(total_score)

            # 成功着陆判断（文档IV.D节隐含评估标准：落地位置在着陆平台附近，量化MPC控制精度）
            if abs(landing_x) <= 0.5 and -0.1 <= landing_y <= 0.1:
                success_count += 1
            # 打印单回合结果（实时监控测试过程，符合文档实验日志风格）
            print(f"测试回合 {ep+1:2d}/{num_episodes} | 得分：{total_score:5.1f} | 步数：{step:3d} | 落地位置：(x={landing_x:.3f}, y={landing_y:.3f})")

    env.close()  # 关闭环境，释放资源

    # -------------------------- 落地位置量化统计（文档IV.D节量化评估延伸，与LQR测试完全一致） --------------------------
    # 提取落地位置的x、y坐标数组（用于计算统计量）
    landing_xs = np.array([pos[0] for pos in landing_positions], dtype=np.float32)
    landing_ys = np.array([pos[1] for pos in landing_positions], dtype=np.float32)
    # 1. 均值：反映落地位置的集中趋势，量化MPC的着陆精度（越接近x_star越优，🔶1-80）
    mean_x = np.mean(landing_xs)
    mean_y = np.mean(landing_ys)
    # 2. 样本方差：反映落地位置的离散程度，量化MPC的鲁棒性（越小越优，文档IV.D节“多回合一致性”要求，🔶1-83）
    var_x = np.var(landing_xs, ddof=1)  # ddof=1：样本方差，适配10次有限测试回合
    var_y = np.var(landing_ys, ddof=1)
    # 3. 标准差：直观反映离散范围（用于图表标注，🔶1-87）
    std_x = np.sqrt(var_x)
    std_y = np.sqrt(var_y)

    # -------------------------- 统计结果打印（对齐文档评估报告风格，与LQR测试格式统一） --------------------------
    x_star_np = x_star.cpu().numpy()  # 目标着陆位置（文档IV.D节定义，🔶1-80）
    print(f"\n=== 落地位置统计结果（文档IV.D节量化评估） ===")
    print(f"目标着陆位置（x_star）：(x={x_star_np[0]:.3f}, y={x_star_np[1]:.3f})")
    print(f"实际落地位置均值：(x={mean_x:.3f}, y={mean_y:.3f})")
    print(f"实际落地位置方差（样本方差）：var_x={var_x:.6f}, var_y={var_y:.6f}")
    print(f"实际落地位置标准差：std_x={std_x:.3f}, std_y={std_y:.3f}")
    print(f"落地位置相对于目标的偏移：Δx={mean_x - x_star_np[0]:.3f}, Δy={mean_y - x_star_np[1]:.3f}")

    # -------------------------- 轨迹汇总图绘制（严格对齐文档图8风格，与LQR测试视觉统一） --------------------------
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10.colors  # 多轨迹颜色区分（避免重叠遮挡，文档图8多回合展示逻辑，🔶1-87）

    # 1. 绘制所有episode的完整轨迹（文档图8核心内容，直观展示MPC的路径规划能力）
    for ep, trajectory in enumerate(all_trajectories):
        x_coords = [p[0] for p in trajectory]
        y_coords = [p[1] for p in trajectory]
        color = colors[ep % len(colors)]  # 循环分配颜色，适配10次测试回合
        plt.plot(x_coords, y_coords, color=color, alpha=0.7)

    # 2. 标注目标着陆位置（文档IV.D节定义的着陆区，红色正方形，与LQR测试视觉一致）
    plt.scatter(
        x_star_np[0], x_star_np[1], 
        color="red", marker="s", s=80, edgecolor="black", 
        label=f"Target Landing Pos (x={x_star_np[0]:.1f}, y={x_star_np[1]:.1f})"
    )

    # 3. 标注落地位置均值（蓝色圆形，反映集中趋势，文档量化评估可视化，🔶1-87）
    plt.scatter(
        mean_x, mean_y, 
        color="blue", marker="o", s=100, edgecolor="black", 
        label=f"Landing Mean (x={mean_x:.3f}, y={mean_y:.3f})"
    )

    # 4. 标注落地位置方差范围（蓝色半透明矩形，±1倍标准差，直观反映鲁棒性，🔶1-83）
    plt.gca().add_patch(
        plt.Rectangle(
            (mean_x - std_x, mean_y - std_y),  # 矩形左下角（均值-标准差）
            2 * std_x, 2 * std_y,  # 矩形宽（2*std_x）、高（2*std_y）
            color="blue", alpha=0.2, edgecolor="blue", linestyle="--",
            label=f"Landing Std Range (±1σ)"
        )
    )

    # 5. 标注着陆平台（黑色虚线，文档IV.D节“着陆区y=0”定义，🔶1-82）
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.8, label="Landing Pad (y=0)")

    # 6. 坐标轴设置（匹配文档状态空间：x∈[-1.5,1.5]，y∈[0,1.5]，确保与LQR测试对比时尺度统一，🔶1-80）
    plt.xlim(-1.5, 1.5)
    plt.ylim(0, 1.5)

    # 7. 标签与标题（文档图8规范，明确控制器类型，与LQR测试区分）
    plt.xlabel("X Position (Horizontal)", fontsize=12)
    plt.ylabel("Y Position (Altitude)", fontsize=12)
    if version == "v1":
        plt.title("Lunar Lander Trajectory Summary (DKRC + MPC) with Landing Stats", fontsize=14)
    elif version == "v2":
        plt.title("Lunar Lander Trajectory Summary (RDKRC + MPC) with Landing Stats", fontsize=14)
    elif version == "v3":
        plt.title("Lunar Lander Trajectory Summary (RRDKRC + MPC) with Landing Stats", fontsize=14)

    # 8. 图例（右侧外摆式布局，避免遮挡轨迹，与LQR测试格式一致）
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1), fontsize=10)
    plt.grid(True, alpha=0.5)

    # 9. 保存汇总图（确保完整显示图例，符合文档实验结果保存要求，便于后续对比分析，🔶1-87）
    plt.savefig(f"./fig/lunar_lander_trajectory_summary_{version}_mpc_with_stats.png", bbox_inches="tight", dpi=300)
    plt.close()

    # -------------------------- 测试总总结（文档IV.D节评估框架，与LQR测试指标统一） --------------------------
    avg_score = np.mean(episode_scores)
    std_score = np.std(episode_scores)
    print(f"\n=== 测试总总结（文档IV.D节评估框架） ===")
    print(f"平均得分：{avg_score:.1f}±{std_score:.1f} | 成功着陆：{success_count}/{num_episodes} 次")
    print(f"落地位置均值：(x={mean_x:.3f}, y={mean_y:.3f}) | 落地位置标准差：(x={std_x:.3f}, y={std_y:.3f})")

    return episode_scores

def train_mc_dkn(
    X_train: torch.Tensor,  # [N, T, x_dim]
    U_train: torch.Tensor,  # [N, T, u_dim]
    batch_size: int = 128,
    epochs_stage1: int = 100,
    epochs_stage2: int = 300,
    lr: float = 1e-3,
    neighbors: int = 10,
    K_steps: int = 15,
    alpha: float = 0.1,  # 嵌入流形约束权重
    beta: float = 0.4,   # 控制流形约束权重
    gamma: float = 0.2,   # 逆映射损失权重
    version:str = 'v1'
):
    env = gym.make("LunarLanderContinuous-v2")
    action_low = env.action_space.low
    action_high = env.action_space.high
    state_low = [-2, -2, -5, -5, -math.pi, -5]
    state_high = [2, 2, 5, 5, math.pi, 5]
    dataset = TensorDataset(X_train, U_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = DKN_MC(x_dim=args.x_dim, u_dim=args.control_dim,hidden_dim=128,manifold_dim=args.x_dim, state_low=state_low, state_high=state_high, 
                   action_low=action_low, action_high=action_high, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # 初始化组件
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    k_step_loss = nn.MSELoss()
    manifold_emb_loss = ManifoldEmbLoss(k=neighbors)
    manifold_ctrl_loss = ManifoldCtrlLoss()
    inv_loss = nn.MSELoss()
    # 5. 初始化损失记录列表（分阶段存储各项损失）
    stage1_k_losses: List[float] = []  # 阶段1：仅K-step损失
    # 阶段2：总损失 + 各子损失
    stage2_total_losses: List[float] = []
    stage2_k_losses: List[float] = []
    stage2_emb_losses: List[float] = []
    stage2_ctrl_losses: List[float] = []
    stage2_inv_losses: List[float] = []
    # -------------------------- 阶段1：基础预训练（无流形约束） --------------------------
    model.train()
    print("阶段1：基础预训练（无流形约束）...")
    for epoch in range(epochs_stage1):
        total_loss = 0.0
        actual_num_batches = 0

        for batch in dataloader:
            # 取批次数据
            batch_X, batch_U = batch
            # K步预测（k=15，文档V.B节）
            x0 = batch_X[:, 0, :]  # [batch, x_dim]
            u_seq = batch_U.permute(1, 0, 2)  # [15, batch, u_dim]
            x_pred_seq = model.predict_k_steps(x0, u_seq, k=K_steps)  # [16, batch, x_dim]
            x_pred_seq = x_pred_seq.permute(1, 0, 2)  # [batch, 16, x_dim]
            
            # 原K步损失（Eq.14）
            loss_k = 0.0
            for i in range(1, K_steps):
                weight = 0.95 ** (i-1)  # gamma=0.95，文档Eq.14
                loss_k += weight * k_step_loss(x_pred_seq[:, i, :], batch_X[:, i, :])
            
            # 优化
            optimizer.zero_grad()
            loss_k.backward()
            optimizer.step()
            total_loss += loss_k.item()
            actual_num_batches += 1
        # 计算当前epoch平均损失并记录
        avg_k_loss = total_loss / actual_num_batches
        stage1_k_losses.append(avg_k_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Stage1 Epoch {epoch+1:4d} | K-step Loss: {avg_k_loss:.6f}")
    plot_stage1_losses(stage1_k_losses, version)
    # -------------------------- 阶段2：流形约束训练 --------------------------
    print("\n阶段2：流形约束训练...")
    for epoch in range(epochs_stage2):
        total_total_loss = 0.0
        total_k_loss = 0.0
        total_emb_loss = 0.0
        total_ctrl_loss = 0.0
        total_inv_loss = 0.0
        actual_num_batches = 0
        
        for batch in dataloader:
            batch_X, batch_U = batch  # 直接从dataloader获取batch
            batch_X = batch_X.to(device)
            batch_U = batch_U.to(device)
            batch_size = batch_X.shape[0]
            
            # 1. K步预测损失（基础）
            x0 = batch_X[:, 0, :]
            u_seq = batch_U.permute(1, 0, 2)
            x_pred_seq = model.predict_k_steps(x0, u_seq, k=K_steps)
            x_pred_seq = x_pred_seq.permute(1, 0, 2)
            loss_k = 0.0
            for i in range(1, K_steps):
                weight = 0.95 ** (i-1)
                loss_k += weight * k_step_loss(x_pred_seq[:, i, :], batch_X[:, i, :])
            
            # 2. 嵌入流形约束损失（局部邻域保持）
            # 取批次内所有状态样本（flat为[N*T, x_dim]）
            X_batch_flat = batch_X.view(-1, model.x_dim)  # [batch*T, x_dim]
            z_batch_flat = model.embed(X_batch_flat)  # [batch*T, manifold_dim]
            loss_emb = manifold_emb_loss(z_batch_flat, X_batch_flat)
            
            # 3. 控制流形约束损失（线性演化一致性）
            # 取t=0到t=T-2的时序对（z_t, z_t1, g_phi_t）
            z_M_t = model.embed(batch_X[:, :-1, :].reshape(-1, model.x_dim))  # [batch*(T-1), manifold_dim]
            z_M_t1 = model.embed(batch_X[:, 1:, :].reshape(-1, model.x_dim))  # [batch*(T-1), manifold_dim]
            g_phi_t = model.forward_control(
                batch_X[:, :-1, :].reshape(-1, model.x_dim),
                batch_U[:, :-1, :].reshape(-1, model.u_dim)
            )  # [batch*(T-1), u_dim]
            loss_ctrl = manifold_ctrl_loss(model.A, model.B, z_M_t, z_M_t1, g_phi_t)
            
            # 4. 逆映射损失
            u_flat = batch_U.view(-1, model.u_dim)  # [batch*T, u_dim]
            g_phi_flat = model.forward_control(X_batch_flat, u_flat)  # [batch*T, u_dim]
            u_recov = model.forward_inv_control(X_batch_flat, g_phi_flat)  # [batch*T, u_dim]
            loss_inv = inv_loss(u_flat, u_recov)
            
            # 总损失
            loss_total = loss_k + alpha * loss_emb + beta * loss_ctrl + gamma * loss_inv
            
            # 优化
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            # 累计损失
            # 累计各项损失与batch数
            total_total_loss += loss_total.item()
            total_k_loss += loss_k.item()
            total_emb_loss += loss_emb.item()
            total_ctrl_loss += loss_ctrl.item()
            total_inv_loss += loss_inv.item()
            actual_num_batches += 1
        # 计算当前epoch平均损失并记录
        avg_total_loss = total_total_loss / actual_num_batches
        avg_k_loss = total_k_loss / actual_num_batches
        avg_emb_loss = total_emb_loss / actual_num_batches
        avg_ctrl_loss = total_ctrl_loss / actual_num_batches
        avg_inv_loss = total_inv_loss / actual_num_batches
        
        stage2_total_losses.append(avg_total_loss)
        stage2_k_losses.append(avg_k_loss)
        stage2_emb_losses.append(avg_emb_loss)
        stage2_ctrl_losses.append(avg_ctrl_loss)
        stage2_inv_losses.append(avg_inv_loss)
        # 打印进度（每50轮）
        if (epoch + 1) % 100 == 0:
            for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
        if (epoch + 1) % 50 == 0:
            print(f"Stage2 Epoch {epoch+1:4d} | Total Loss: {avg_total_loss:.6f} | "
                  f"K-step: {avg_k_loss:.6f} | Emb: {avg_emb_loss:.6f} | "
                  f"Ctrl: {avg_ctrl_loss:.6f} | Inv: {avg_inv_loss:.6f}")
    # 阶段2结束：绘制各项损失对比曲线
    plot_stage2_losses(
        total_losses=stage2_total_losses,
        k_losses=stage2_k_losses,
        emb_losses=stage2_emb_losses,
        ctrl_losses=stage2_ctrl_losses,
        inv_losses=stage2_inv_losses,
        version=version
    )
    return model

def calculate_parameter(psi, x_dim, z_dim, control_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A_lander = psi.A.weight
    B_lander = psi.B.weight
    I_n = torch.eye(x_dim, device=device)
    zero_mat = torch.zeros(x_dim, z_dim, device=device)
    C = torch.cat([I_n, zero_mat], dim=1)
    Q = torch.eye(x_dim, device=device)
    Q_ = C.T @ Q @ C
    Q_ = 0.5 * (Q_ + Q_.T)
    R_ = 0.1 * torch.eye(control_dim, device=device)

    Q_ = Q_.cpu().detach().numpy()
    R_ = R_.cpu().detach().numpy()
    return A_lander, B_lander, Q_, R_

def plot_stage1_losses(loss_list: List[float], version: str) -> None:
    """绘制阶段1的K-step损失曲线（仅1条曲线，聚焦预训练收敛情况）"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_list)+1), loss_list, color="#2E86AB", linewidth=2, label="K-step Loss")
    
    # 图表美化与标注
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Stage 1: K-step Loss Curve (Version: {version})", fontsize=14, pad=20)
    plt.yscale("log")  # 对数刻度：清晰展示损失下降趋势（尤其前期快速下降阶段）
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=10)
    
    # 保存图片（确保目录存在，避免报错）
    os.makedirs("./fig", exist_ok=True)  # 若./fig不存在则创建
    plt.savefig(f"./fig/stage1_kstep_loss_{version}.png", dpi=300, bbox_inches="tight")
    plt.close()  # 关闭画布，释放内存


def plot_stage2_losses(
    total_losses: List[float],
    k_losses: List[float],
    emb_losses: List[float],
    ctrl_losses: List[float],
    inv_losses: List[float],
    version: str
) -> None:
    """绘制阶段2的所有损失对比曲线（总损失+4个子损失，便于分析各约束效果）"""
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(total_losses)+1)
    
    # 绘制各损失曲线（颜色/线型区分，便于识别）
    plt.plot(epochs, total_losses, color="#A23B72", linewidth=3, label="Total Loss", zorder=5)  # 总损失置顶
    plt.plot(epochs, k_losses, color="#F18F01", linestyle="--", linewidth=2, label="K-step Loss")
    plt.plot(epochs, emb_losses, color="#C73E1D", linestyle="-.", linewidth=2, label="Embedding Loss")
    plt.plot(epochs, ctrl_losses, color="#2E86AB", linestyle=":", linewidth=2, label="Control Loss")
    plt.plot(epochs, inv_losses, color="#6A994E", linestyle="--", linewidth=2, label="Inverse Loss")
    
    # 图表美化与标注
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Stage 2: Loss Curves Comparison (Version: {version})", fontsize=14, pad=20)
    plt.yscale("log")  # 对数刻度：避免某类损失过大掩盖其他损失的变化
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=10, loc="upper right")  # 图例放右上角，避免遮挡曲线
    
    # 保存图片（确保目录存在）
    os.makedirs("./fig", exist_ok=True)
    plt.savefig(f"./fig/stage2_all_losses_{version}.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":  
    parse = argparse.ArgumentParser()
    parse.add_argument('--test_version', type=str, default='MCDKN', help='PsiMLP版本（v1或v2）')
    parse.add_argument('--controller_type', type=str, default='lqr', help='控制器类型（lqr或mpc）')
    parse.add_argument('--seed', type=int, default=50, help='随机种子')
    parse.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parse.add_argument('--epochs_stage1', type=int, default=100, help='一阶段训练轮次')
    parse.add_argument('--epochs_stage2', type=int, default=500, help='二阶段训练轮次')
    parse.add_argument('--data_epochs', type=int, default=50, help='数据轮次')
    parse.add_argument('--batch_size', type=int, default=256, help='批量大小')
    parse.add_argument('--num_episodes', type=int, default=100, help='测试回合数')
    parse.add_argument('--data_prepared', action='store_true', help='是否使用预生成数据')
    parse.add_argument('--z_dim', type=int, default=12, help='高维状态维度N')
    parse.add_argument('--x_dim', type=int, default=6, help='状态维度')
    parse.add_argument('--control_dim', type=int, default=2, help='控制维度')
    parse.add_argument('--neighbors', type=int, default=10, help='邻居数')
    parse.add_argument('--K_steps', type=int, default=15, help='时域长度')
    # 选择测试版本（"v1"为基础版，"v2"为改进版） seed history:2\33\444\22\\789\666
    # test_version = "v1"
    args = parse.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 完整DKRC流程（文档IV.D节实验步骤：数据生成→网络训练→控制测试）
    # 步骤1：生成数据（文档IV.D节：5次游戏→1876组数据，Ornstein-Uhlenbeck噪声）
    print("="*50 + " 步骤1/3：生成月球着陆器数据 " + "="*50)
    if args.data_prepared:
        # 如果数据已准备好，直接加载（避免重复生成）
        data = np.load(f"./data/lunar_lander_ksteps_seed{args.seed}_ep{args.data_epochs}_K{args.K_steps}.npz")
        x_prev = data['x_seq']
        u_prev = data['u_seq']
        x_next = data['x_next_seq']
        print(f"已加载预生成数据：{x_prev.shape[0]}组数据")
    else:
        x_prev, u_prev, x_next = generate_lunar_lander_data_ksteps(
            num_episodes=args.data_epochs,  # 文档指定5次，对应1876组数据
            noise_scale=0.1,  # 文档IV.D节指定噪声强度
            K_steps=args.K_steps,
            seed=args.seed,
            window_step=1
        )

    print("\n" + "="*50 + " 步骤2/3：训练PsiMLP网络 " + "="*50)
    x_prev = torch.tensor(x_prev, dtype=torch.float32, device=device)
    u_prev = torch.tensor(u_prev, dtype=torch.float32, device=device)
    # 步骤2：训练PsiMLP网络（文档II.28节+Algorithm 1）
    psi_lander = train_mc_dkn(
        X_train=x_prev,
        U_train=u_prev,
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        lr=args.lr,
        neighbors=args.neighbors
    )
    # 保存A/B/C矩阵（便于后续分析）
    # np.savez(f"./data/lunar_lander_ABC_{args.test_version}_seed{args.seed}.npz", A=A_lander.cpu().numpy(), B=B_lander.cpu().numpy(), C=C_lander.cpu().numpy())
    # 步骤3：LQR控制测试（文档III节+IV.D节，用训练后的A/B计算LQR增益）
    print("\n" + "="*50 + " 步骤3/3：LQR控制测试 " + "="*50)
    # 目标状态x*：文档IV.D节定义（x=0, y=0，其余为0）
    x_star_lander = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=next(psi_lander.parameters()).device)
    A_lander, B_lander, Q_, R_ = calculate_parameter(psi_lander, args.x_dim, args.z_dim, args.control_dim)
    K_lqr = solve_discrete_lqr(A_lander, B_lander)
    test_lander_lqr(psi_lander, K_lqr, x_star_lander, num_episodes=args.num_episodes, version=args.test_version, seed=args.seed)