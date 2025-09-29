import torch
import gym
import torch.optim as optim
import numpy as np
import argparse
import math
import matplotlib.pyplot as plt
from tqdm import trange
from typing import Tuple, List
from torch.utils.data import TensorDataset, DataLoader
from rdkrc.utils.data_utils import generate_lunar_lander_data
from rdkrc.models.psi_mlp import PsiMLP, PsiMLP_v2
from rdkrc.trainer.loss_functions import compute_total_loss
from rdkrc.utils.matrix_utils import compute_C_matrix, update_A_B
from rdkrc.controller.lqr_controller import solve_discrete_lqr, solve_discrete_lqr_v2
from rdkrc.controller.mpc_controller import DKRCMPCController


def test_lander_lqr(
    psi: PsiMLP,
    K_lqr: np.ndarray,
    x_star: torch.Tensor,
    num_episodes: int = 10,
    max_steps: int = 500,
    version: str = "v1",
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
                x_prev_tensor = torch.tensor(x_prev, device=device, dtype=torch.float32).unsqueeze(0)
                z_prev = psi.compute_z(x_prev_tensor, x_star)
                z_prev_np = z_prev.cpu().detach().numpy()

                # 2. 计算LQR控制输入（文档III节：v_t=-K_lqr z_t，u_t=v_t+u₀，控制律设计）
                v_t = -K_lqr @ z_prev_np.T  # 变换后控制输入（适配高维线性模型）
                u0 = psi.forward_u0(x_prev_tensor).cpu().detach().numpy().squeeze()  # 文档II.36节u₀补偿（控制固定点）
                u_t = v_t.squeeze() + u0
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
            if abs(landing_x) <= 0.5 and -0.1 <= landing_y <= 0.1:
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

    # 8. 图例（避免遮挡轨迹，文档图8右侧布局，包含新增的均值/方差标注）
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1), fontsize=10)
    plt.grid(True, alpha=0.5)

    # 9. 保存汇总图（确保完整显示图例，符合文档实验结果保存要求，🔶1-87）
    plt.savefig(f"lunar_lander_trajectory_summary_{version}_with_stats.png", bbox_inches="tight", dpi=300)
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

    # 8. 图例（右侧外摆式布局，避免遮挡轨迹，与LQR测试格式一致）
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1), fontsize=10)
    plt.grid(True, alpha=0.5)

    # 9. 保存汇总图（确保完整显示图例，符合文档实验结果保存要求，便于后续对比分析，🔶1-87）
    plt.savefig(f"lunar_lander_trajectory_summary_{version}_mpc_with_stats.png", bbox_inches="tight", dpi=300)
    plt.close()

    # -------------------------- 测试总总结（文档IV.D节评估框架，与LQR测试指标统一） --------------------------
    avg_score = np.mean(episode_scores)
    std_score = np.std(episode_scores)
    print(f"\n=== 测试总总结（文档IV.D节评估框架） ===")
    print(f"平均得分：{avg_score:.1f}±{std_score:.1f} | 成功着陆：{success_count}/{num_episodes} 次")
    print(f"落地位置均值：(x={mean_x:.3f}, y={mean_y:.3f}) | 落地位置标准差：(x={std_x:.3f}, y={std_y:.3f})")

    return episode_scores

def train_psi_lander(
    x_prev: np.ndarray,
    u_prev: np.ndarray,
    x_next: np.ndarray,
    z_dim: int = 36,
    epochs: int = 500,
    batch_size: int = 128,
    lr: float = 1e-4,
    version: str = "v1"
) -> Tuple[torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    训练月球着陆器的PsiMLP网络（文档Algorithm 1完整流程）
    核心修正：补充\(u_0\)调用、纠正A/B初始化、用全部数据计算最终A/B/C、适配DataLoader批量逻辑。
    
    Args:
        x_prev: 原始状态序列，shape=[total_samples,6]（文档IV.D节数据格式）
        u_prev: 控制输入序列，shape=[total_samples,2]（文档IV.D节控制维度）
        x_next: 下一状态序列，shape=[total_samples,6]
        z_dim: 高维线性空间维度N（文档II.28节未指定，默认256）
        epochs: 训练轮次（文档II.28节未指定，默认500）
        batch_size: 批量大小（文档II.27节批量训练逻辑，默认128）
        lr: 学习率（文档II.28节用ADAM优化器，默认1e-4）
        version: PsiMLP版本选择（"v1"为基础版，"v2"为改进版，默认"v1"）
    Returns:
        psi: 训练好的PsiMLP网络（含\(u_0\)）
        A_final: 收敛后的Koopman矩阵，shape=[256,256]（文档Equation 5）
        B_final: 收敛后的控制矩阵，shape=[256,2]（文档Equation 5）
        C_final: 状态重构矩阵，shape=[6,256]（文档Equation 9）
    """
    # 1. 设备与环境参数初始化（文档II.28节推荐GPU，获取状态上下界）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("LunarLanderContinuous-v2")
    state_low = [-5, -5, -20, -20, -10, -10]
    state_high = [5, 5, 20, 20, 10, 10]
    env.close()
    print(f"使用设备：{device}（文档II.28节推荐NVIDIA GPU）")

    # 2. 数据转换与批量加载（文档II.27节数据预处理逻辑）
    x_prev_tensor = torch.tensor(x_prev, device=device, dtype=torch.float32)
    u_prev_tensor = torch.tensor(u_prev, device=device, dtype=torch.float32)
    x_next_tensor = torch.tensor(x_next, device=device, dtype=torch.float32)
    # 用DataLoader实现批量采样（打乱+分批，避免手动切片误差）
    dataset = TensorDataset(x_prev_tensor, u_prev_tensor, x_next_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 3. 核心模块初始化（严格匹配文档定义）
    # 3.1 PsiMLP：输入6维，输出256维（N≫6），控制维度2，传入状态上下界
    if version == "v1":
        psi = PsiMLP(
            input_dim=6,
            output_dim=z_dim,
            control_dim=2,
            low=state_low,
            high=state_high,
            hidden_dims=[256, 256, 256, 256]  # 文档II.28节4层隐藏层
        ).to(device)
    elif version == "v2":
        psi = PsiMLP_v2(
            input_dim=6,
            output_dim=z_dim,
            control_dim=2,
            low=state_low,
            high=state_high,
            hidden_dims=[256, 256, 256, 256]  # 文档II.28节4层隐藏层
        ).to(device)
    # 3.2 优化器：ADAM（文档II.28节指定）
    optimizer = optim.Adam(psi.parameters(), lr=lr)
    # 3.3 目标状态x*：文档IV.D节定义为着陆区（x=10, y=4，其余状态为0）
    x_star = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)
    # 3.4 A/B初始化：随机正态分布（文档II.39节“随机初始化A/B”），避免对角矩阵偏置
    N = z_dim  # 高维空间维度
    m = 2      # 控制输入维度
    A = torch.randn(N, N, device=device)
    B = torch.randn(N, 2, device=device)
    # 初始化归一化（避免数值溢出，文档未明说但为训练稳定性必需）
    A = A / torch.norm(A, dim=0, keepdim=True)
    B = B / torch.norm(B, dim=0, keepdim=True)
    avg_loss_list: List[float] = []
    # 4. 训练循环（文档Algorithm 1步骤1-4）
    psi.train()
    for epoch in range(epochs):
        total_epoch_loss = 0.0
        L1_loss = 0.0
        L2_loss = 0.0
        for batch in dataloader:
            x_prev_batch, u_prev_batch, x_next_batch = batch  # [B,6], [B,2], [B,6]
            
            # 4.1 计算高维线性状态z（文档Algorithm 1步骤1：z = Ψ(x) - Ψ(x*)）
            z_prev_batch = psi.compute_z(x_prev_batch, x_star)  # [B,256]
            z_next_batch = psi.compute_z(x_next_batch, x_star)  # [B,256]
            
            # 4.2 获取控制固定点u0（文档II.36节“辅助网络学习u0”，匹配批量大小）
            u0_batch = psi.forward_u0(x_prev_batch)  # [B,2]
            
            # 4.3 更新A/B矩阵（文档Algorithm 1隐含步骤，调用matrix_utils）
            A, B = update_A_B(z_prev_batch, z_next_batch, u_prev_batch, A, B)
            
            # 4.4 计算总损失（文档Algorithm 1步骤4：L(θ) = L1 + L2，加入u_prev和u0）
            total_loss, L1, L2 = compute_total_loss(
                z_prev=z_prev_batch,
                z_next=z_next_batch,
                A=A,
                B=B,
                u_prev=u_prev_batch,
                u0=u0_batch,
                lambda_L1=1,
                lambda_L2=0.01  
            )
            
            # 4.5 反向传播与参数更新
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_epoch_loss += total_loss.item() * batch_size  # 累积 epoch 损失
            L1_loss += L1.item() * batch_size
            L2_loss += L2.item() * batch_size
        # 每过20个epoch降低一次学习率
        if (epoch + 1) % 20 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        # 打印epoch信息（平均损失，便于监控收敛）
        avg_epoch_loss = total_epoch_loss / len(dataset)
        L1 = L1_loss / len(dataset)
        L2 = L2_loss / len(dataset)
        avg_loss_list.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1:3d}/{epochs}] | 平均总损失：{avg_epoch_loss:.4f} | L1：{L1:.4f} | L2：{L2:.4f}", end='\r', flush=True)
    plot_loss_curve(avg_loss_list, version)
    # 5. 计算最终A/B/C矩阵（文档Algorithm 1步骤5，用全部数据确保收敛精度）
    psi.eval()
    with torch.no_grad():
        # 5.1 计算全部数据的z（用于A/B/C计算）
        z_prev_all = psi.compute_z(x_prev_tensor, x_star)  # [total,256]
        z_next_all = psi.compute_z(x_next_tensor, x_star)  # [total,256]
        # 5.2 最终A/B：用全部数据更新一次（避免批量偏差）
        A_final, B_final = update_A_B(z_prev_all, z_next_all, u_prev_tensor, A, B)
        # 5.3 最终C：文档Equation 9，输入z_prev（而非Ψ(x)），满足CΨ0=0约束
        C_final = compute_C_matrix(x_prev_tensor, z_prev_all)  # [6,256]

    print(f"\nPsiMLP训练完成 | A_final.shape: {A_final.shape} | B_final.shape: {B_final.shape} | C_final.shape: {C_final.shape}")
    return psi, A_final, B_final, C_final

def plot_loss_curve(loss_list: List[float], version: str) -> None:
    """
    绘制训练损失曲线（便于监控训练过程）
    
    Args:
        loss_list: 每个epoch的平均损失列表
        version: PsiMLP版本标识（用于保存文件命名）
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label='Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.yscale('log')  # 对数刻度便于观察收敛趋势
    plt.grid(True)
    plt.legend()
    plt.savefig(f'training_loss_curve_{version}.png')


def design_q_matrix(psi: PsiMLP, x_star: torch.Tensor, pos_weight: float = 100.0, other_weight: float = 1.0) -> np.ndarray:
    """
    为复杂网络设计Q矩阵：通过Psi网络找到x/y对应的Z分量，放大其权重
    """
    device = next(psi.parameters()).device
    N = psi.output_dim  # Z维度（如256）
    Q = np.eye(N) * other_weight  # 基础权重

    # 1. 找到x/y变化敏感的Z分量（通过梯度计算：dΨ/dx、dΨ/dy）
    x_sample = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32).unsqueeze(0)  # x偏移样本
    y_sample = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32).unsqueeze(0)  # y偏移样本
    xy_sample = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32).unsqueeze(0)  # x/y偏移样本
    x_star_tensor = x_star.unsqueeze(0)

    # 2. 计算Ψ对x/y的梯度（敏感Z分量梯度大）
    x_sample.requires_grad_(True)
    z_x = psi.compute_z(x_sample, x_star_tensor)
    z_x.sum().backward()
    x_sensitivity = x_sample.grad.squeeze().cpu().numpy()  # 对x的敏感Z分量

    y_sample.requires_grad_(True)
    z_y = psi.compute_z(y_sample, x_star_tensor)
    z_y.sum().backward()
    y_sensitivity = y_sample.grad.squeeze().cpu().numpy()  # 对y的敏感Z分量

    xy_sample.requires_grad_(True)
    z_xy = psi.compute_z(xy_sample, x_star_tensor)
    z_xy.sum().backward()
    xy_sensitivity = xy_sample.grad.squeeze().cpu().numpy()  # 对x/y的敏感Z分量


    # 3. 放大敏感Z分量的权重
    sensitive_indices = np.where((abs(x_sensitivity) > 1e-6) | (abs(y_sensitivity) > 1e-6)| (abs(xy_sensitivity) > 1e-6))[0]  # 阈值可调整
    Q[sensitive_indices, sensitive_indices] = pos_weight  # 位置相关Z分量权重=10
    print(f"Q矩阵设计完成：{len(sensitive_indices)}/{N}个Z分量为位置敏感维度，权重={pos_weight}")
    return Q

if __name__ == "__main__":  
    parse = argparse.ArgumentParser()
    parse.add_argument('--test_version', type=str, default='v1', help='PsiMLP版本（v1或v2）')
    parse.add_argument('--controller_type', type=str, default='lqr', help='控制器类型（lqr或mpc）')
    parse.add_argument('--seed', type=int, default=50, help='随机种子')
    parse.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parse.add_argument('--epochs', type=int, default=10, help='训练轮次')
    parse.add_argument('--batch_size', type=int, default=256, help='批量大小')
    parse.add_argument('--num_episodes', type=int, default=100, help='测试回合数')
    parse.add_argument('--data_prepared', action='store_true', help='是否使用预生成数据')
    parse.add_argument('--z_dim', type=int, default=36, help='高维状态维度N')
    # 选择测试版本（"v1"为基础版，"v2"为改进版） seed history:2\33\444\22\\789\666
    # test_version = "v1"
    args = parse.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 完整DKRC流程（文档IV.D节实验步骤：数据生成→网络训练→控制测试）
    # 步骤1：生成数据（文档IV.D节：5次游戏→1876组数据，Ornstein-Uhlenbeck噪声）
    print("="*50 + " 步骤1/3：生成月球着陆器数据 " + "="*50)
    if args.data_prepared:
        # 如果数据已准备好，直接加载（避免重复生成）
        data = np.load(f"./data/lunar_lander_data_seed{args.seed}_episodes10.npz")
        x_prev = data['x_prev']
        u_prev = data['u_prev']
        x_next = data['x_next']
        print(f"已加载预生成数据：{x_prev.shape[0]}组数据")
    else:
        x_prev, u_prev, x_next = generate_lunar_lander_data(
            num_episodes=10,  # 文档指定5次，对应1876组数据
            noise_scale=0.1,  # 文档IV.D节指定噪声强度
            seed=args.seed
        )

    print("\n" + "="*50 + " 步骤2/3：训练PsiMLP网络 " + "="*50)
      
    # 步骤2：训练PsiMLP网络（文档II.28节+Algorithm 1）
    psi_lander, A_lander, B_lander, C_lander = train_psi_lander(
        x_prev=x_prev,
        u_prev=u_prev,
        x_next=x_next,
        z_dim=args.z_dim if hasattr(args, 'z_dim') else 256,
        epochs=args.epochs,  # 足够轮次确保收敛
        batch_size=args.batch_size,
        lr=args.lr,
        version=args.test_version
    )
    # 保存A/B/C矩阵（便于后续分析）
    np.savez(f"lunar_lander_ABC_{args.test_version}_seed{args.seed}.npz", A=A_lander.cpu().numpy(), B=B_lander.cpu().numpy(), C=C_lander.cpu().numpy())
    # 步骤3：LQR控制测试（文档III节+IV.D节，用训练后的A/B计算LQR增益）
    print("\n" + "="*50 + " 步骤3/3：LQR控制测试 " + "="*50)
    # 目标状态x*：文档IV.D节定义（x=0, y=0，其余为0）
    x_star_lander = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=next(psi_lander.parameters()).device)
    # 求解LQR增益（文档III节离散黎卡提方程）
    if args.test_version == "v1":
        if args.controller_type == "lqr":
            K_lqr = solve_discrete_lqr(A_lander, B_lander)
        elif args.controller_type == "mpc":
            mpc_controller = DKRCMPCController(A=A_lander, B=B_lander, C=C_lander, psi_net=psi_lander, Q=np.diag([10]*6), R=0.1*np.eye(2), pred_horizon=10, x_star=x_star_lander, u0=psi_lander.u0.detach().cpu().numpy())
    elif args.test_version == "v2":
        if args.controller_type == "lqr":
            Q_complex = design_q_matrix(psi_lander, x_star_lander, pos_weight=1.0, other_weight=100.0)
            K_lqr = solve_discrete_lqr_v2(A_lander, B_lander, Q=Q_complex)
        elif args.controller_type == "mpc":
            mpc_controller = DKRCMPCController(A=A_lander, B=B_lander, C=C_lander, psi_net=psi_lander, Q=np.diag([10]*3 + [1]*3), R=0.1*np.eye(2), pred_horizon=10, x_star=x_star_lander, u0=psi_lander.u0.detach().cpu().numpy())
        # low_dim = 64
        # high_dim = 256 - low_dim
        # K_lqr = solve_discrete_lqr(A_lander, B_lander, Q=np.diag([10] * low_dim + [1] * high_dim), R=0.1*np.eye(2))
    # 测试控制效果（文档IV.D节10次测试）
    if args.controller_type == "lqr":
        test_lander_lqr(psi_lander, K_lqr, x_star_lander, num_episodes=args.num_episodes, version=args.test_version, seed=args.seed)
    elif args.controller_type == "mpc":
        test_lander_mpc(psi_lander, mpc_controller, x_star_lander, num_episodes=args.num_episodes, version=args.test_version, seed=args.seed)