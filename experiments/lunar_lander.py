import torch
import gym
import torch.optim as optim
import numpy as np
import tqdm
import math
import matplotlib.pyplot as plt
from typing import Tuple, List
from torch.utils.data import TensorDataset, DataLoader
from rdkrc.utils.data_utils import generate_lunar_lander_data
from rdkrc.models.psi_mlp import PsiMLP
from rdkrc.trainer.loss_functions import compute_total_loss
from rdkrc.utils.matrix_utils import compute_C_matrix, update_A_B
from rdkrc.controller.lqr_controller import solve_discrete_lqr


def test_lander_lqr(
    psi: PsiMLP,
    K_lqr: np.ndarray,
    x_star: torch.Tensor,
    num_episodes: int = 10,
    max_steps: int = 500
) -> List[float]:
    """
    月球着陆器LQR控制测试（文档IV.D节）
    核心修正：加入\(u_0\)计算\(u_t = v_t + u_0\)，纠正目标状态与控制输入裁剪逻辑。
    
    Args:
        psi: 训练好的PsiMLP网络（含\(u_0\)参数）
        K_lqr: LQR控制增益，shape=[2, 256]（文档III节输出）
        x_star: 目标状态（着陆区，文档IV.D节定义：x=0, y=0），shape=[6]
        num_episodes: 测试回合数（文档IV.D节指定10次）
        max_steps: 每回合最大步数（避免无限循环）
    Returns:
        episode_scores: 每回合得分列表（Gym内置得分，>200为成功着陆）
    """
    env = gym.make("LunarLanderContinuous-v2")
    device = next(psi.parameters()).device
    episode_scores: List[float] = []

    psi.eval()  # 推理模式（禁用梯度计算）
    with torch.no_grad():
        for ep in range(num_episodes):
            x_prev = env.reset()  # Gym新版本返回(obs, info)，适配接口
            x_prev = x_prev[0:6]  # 仅保留文档定义的6维状态（x,y,θ,ẋ,ẏ,θ̇）
            done = False
            total_score = 0.0
            step = 0

            while not done and step < max_steps:
                # 1. 状态预处理：计算高维线性状态z = Ψ(x) - Ψ(x*)（文档Equation 4）
                x_prev_tensor = torch.tensor(x_prev, device=device, dtype=torch.float32).unsqueeze(0)  # [1,6]
                z_prev = psi.compute_z(x_prev_tensor, x_star)  # [1,256]
                z_prev_np = z_prev.cpu().detach().numpy()

                # 2. 计算LQR控制输入（文档III节逻辑：v_t = -K_lqr z_t，u_t = v_t + u_0）
                v_t = -K_lqr @ z_prev_np.T  # [2,1]（变换后控制输入）
                u0 = psi.forward_u0(x_prev_tensor).cpu().detach().numpy().squeeze()  # [2]（从PsiMLP获取训练后的u0）
                u_t = v_t.squeeze() + u0  # [2]（原始控制输入，加入u0补偿）

                # 3. 控制输入裁剪（确保在环境动作空间内，文档IV.D节隐含约束）
                u_t = np.clip(u_t, env.action_space.low, env.action_space.high)

                # 4. 环境交互（获取下一状态与奖励）
                x_next, reward, done, _ = env.step(u_t)  # 适配Gym新版本接口
                total_score += reward
                x_prev = x_next[0:6]  # 更新状态，保留前6维
                step += 1

            episode_scores.append(total_score)
            print("最终状态：", x_prev)
            print(f"测试回合 {ep+1:2d}/{num_episodes} | 得分：{total_score:5.1f} | 步数：{step:3d}")

    env.close()
    # 结果统计（文档IV.D节评估标准：平均得分、成功着陆次数）
    avg_score = np.mean(episode_scores)
    std_score = np.std(episode_scores)
    success_count = sum(score > 200 for score in episode_scores)
    print(f"\n测试总结：平均得分 {avg_score:.1f}±{std_score:.1f} | 成功着陆 {success_count}/{num_episodes} 次")
    return episode_scores


def train_psi_lander(
    x_prev: np.ndarray,
    u_prev: np.ndarray,
    x_next: np.ndarray,
    epochs: int = 500,
    batch_size: int = 128,
    lr: float = 1e-4
) -> Tuple[PsiMLP, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    训练月球着陆器的PsiMLP网络（文档Algorithm 1完整流程）
    核心修正：补充\(u_0\)调用、纠正A/B初始化、用全部数据计算最终A/B/C、适配DataLoader批量逻辑。
    
    Args:
        x_prev: 原始状态序列，shape=[total_samples,6]（文档IV.D节数据格式）
        u_prev: 控制输入序列，shape=[total_samples,2]（文档IV.D节控制维度）
        x_next: 下一状态序列，shape=[total_samples,6]
        epochs: 训练轮次（文档II.28节未指定，默认500）
        batch_size: 批量大小（文档II.27节批量训练逻辑，默认128）
        lr: 学习率（文档II.28节用ADAM优化器，默认1e-4）
    Returns:
        psi: 训练好的PsiMLP网络（含\(u_0\)）
        A_final: 收敛后的Koopman矩阵，shape=[256,256]（文档Equation 5）
        B_final: 收敛后的控制矩阵，shape=[256,2]（文档Equation 5）
        C_final: 状态重构矩阵，shape=[6,256]（文档Equation 9）
    """
    # 1. 设备与环境参数初始化（文档II.28节推荐GPU，获取状态上下界）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("LunarLanderContinuous-v2")
    state_low = [-1.5, 0, -5, -5, -math.pi, -8]
    state_high = [1.5, 1.5, 5, 5, math.pi, 8]
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
    psi = PsiMLP(
        input_dim=6,
        output_dim=256,
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
    N = 256  # 高维空间维度
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
                lambda_L1=0.999,
                lambda_L2=0.001  # 文档Algorithm 1无权重，默认1.0
            )
            
            # 4.5 反向传播与参数更新
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_epoch_loss += total_loss.item() * batch_size  # 累积 epoch 损失

        # 打印epoch信息（平均损失，便于监控收敛）
        avg_epoch_loss = total_epoch_loss / len(dataset)
        avg_loss_list.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1:3d}/{epochs}] | 平均总损失：{avg_epoch_loss:.4f} | L1：{L1.item():.4f} | L2：{L2.item():.4f}")
    plot_loss_curve(avg_loss_list)
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

def plot_loss_curve(loss_list: List[float]) -> None:
    """
    绘制训练损失曲线（便于监控训练过程）
    
    Args:
        loss_list: 每个epoch的平均损失列表
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label='Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.yscale('log')  # 对数刻度便于观察收敛趋势
    plt.grid(True)
    plt.legend()
    plt.savefig('training_loss_curve.png')

if __name__ == "__main__":
    # 完整DKRC流程（文档IV.D节实验步骤：数据生成→网络训练→控制测试）
    # 步骤1：生成数据（文档IV.D节：5次游戏→1876组数据，Ornstein-Uhlenbeck噪声）
    print("="*50 + " 步骤1/3：生成月球着陆器数据（文档IV.D节） " + "="*50)
    x_prev, u_prev, x_next = generate_lunar_lander_data(
        num_episodes=10,  # 文档指定5次，对应1876组数据
        noise_scale=0.1  # 文档IV.D节指定噪声强度
    )

    # 步骤2：训练PsiMLP网络（文档II.28节+Algorithm 1）
    print("\n" + "="*50 + " 步骤2/3：训练PsiMLP网络（文档Algorithm 1） " + "="*50)
    psi_lander, A_lander, B_lander, C_lander = train_psi_lander(
        x_prev=x_prev,
        u_prev=u_prev,
        x_next=x_next,
        epochs=50,  # 足够轮次确保收敛
        batch_size=128,
        lr=1e-5
    )

    # 步骤3：LQR控制测试（文档III节+IV.D节，用训练后的A/B计算LQR增益）
    print("\n" + "="*50 + " 步骤3/3：LQR控制测试（文档III节） " + "="*50)
    # 目标状态x*：文档IV.D节定义（x=0, y=0，其余为0）
    x_star_lander = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=next(psi_lander.parameters()).device)
    # 求解LQR增益（文档III节离散黎卡提方程）
    K_lqr = solve_discrete_lqr(A_lander, B_lander)
    # 测试控制效果（文档IV.D节10次测试）
    test_lander_lqr(psi_lander, K_lqr, x_star_lander, num_episodes=10)