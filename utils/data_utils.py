import gym
import os
import torch
import numpy as np
from typing import Tuple
from scipy.spatial import KDTree

def compute_knn_neighbors(X: torch.Tensor, k: int = 10) -> torch.Tensor:
    """计算原状态空间的K近邻索引（X: [N, n]，N为样本数，n为状态维度）"""
    X_np = X.cpu().detach().numpy()
    kdtree = KDTree(X_np)
    _, neighbors_idx = kdtree.query(X_np, k=k+1)  # 第0个是自身，取1~k
    return torch.tensor(neighbors_idx[:, 1:], device=X.device, dtype=torch.long)  # [N, k]


def generate_lunar_lander_data(
    num_episodes: int = 10,
    noise_scale: float = 0.1,
    env_name: str = "LunarLanderContinuous-v2",
    seed: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成论文IV.D节月球着陆器训练数据：随机策略+Ornstein-Uhlenbeck噪声
    仅用于当前月球着陆器实验，不支持其他环境。
    
    Args:
        num_episodes: 生成数据的游戏次数（论文指定5次，对应1876组数据）
        noise_scale: Ornstein-Uhlenbeck噪声强度（论文IV.D节用0.1）
        env_name: 环境名（固定为月球着陆器环境，不修改）
        seed: 随机种子，确保结果可复现（默认2，论文未指定）
    
    Returns:
        x_prev: 原始状态序列，shape=[total_samples, 6]（6维：x,y,θ,ẋ,ẏ,θ_dot）
        u_prev: 控制输入序列，shape=[total_samples, 2]（2维：u₁主引擎, u₂侧引擎）
        x_next: 下一状态序列，shape=[total_samples, 6]
    """
    env = gym.make(env_name)
    env.seed(seed)
    x_prev_list: list[np.ndarray] = []
    u_prev_list: list[np.ndarray] = []
    x_next_list: list[np.ndarray] = []

    # 论文指定的Ornstein-Uhlenbeck噪声（用于数据探索）
    def _ou_noise(prev_noise: np.ndarray) -> np.ndarray:
        theta = 0.15  # 噪声衰减系数（论文默认值）
        return theta * (-prev_noise) + noise_scale * np.random.randn(*prev_noise.shape)

    for _ in range(num_episodes):
        x_prev = env.reset()
        prev_noise = np.zeros(env.action_space.shape[0])
        done = False
        while not done:
            # 生成带噪声的随机控制（论文：未训练RL策略）
            u_prev = env.action_space.sample() + _ou_noise(prev_noise)
            u_prev = np.clip(u_prev, env.action_space.low, env.action_space.high)
            prev_noise = _ou_noise(prev_noise)

            # 交互获取下一状态
            x_next, _, done, _ = env.step(u_prev)

            # 存储数据（严格匹配论文6维状态、2维控制）
            x_prev_list.append(x_prev[0:6])  # 取前6维状态（忽略第7、8维）
            u_prev_list.append(u_prev)
            x_next_list.append(x_next[0:6])  # 取前6维状态（忽略第7、8维）

            x_prev = x_next

    env.close()
    # 转换为numpy数组（类型锁定为float32，适配PyTorch）
    x_prev = np.array(x_prev_list, dtype=np.float32)
    u_prev = np.array(u_prev_list, dtype=np.float32)
    x_next = np.array(x_next_list, dtype=np.float32)
    # 保存至/data/lunar_lander_data_seed{seed}_episodes{num_episodes}.npz
    # 检查是否有data目录，没有则创建
    if not os.path.exists("./data"):
        os.makedirs("./data")
    np.savez_compressed(
        f"./data/lunar_lander_data_seed{seed}_episodes{num_episodes}.npz",
        x_prev=x_prev,
        u_prev=u_prev,
        x_next=x_next
    )
    print(f"数据生成完成：{x_prev.shape[0]}组数据（论文目标1876组）")
    return x_prev, u_prev, x_next



def generate_lunar_lander_data_ksteps(
    num_episodes: int = 10,
    noise_scale: float = 0.1,
    env_name: str = "LunarLanderContinuous-v2",
    seed: int = 2,
    K_steps: int=15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成论文IV.D节月球着陆器训练数据：随机策略+Ornstein-Uhlenbeck噪声
    仅用于当前月球着陆器实验，不支持其他环境。
    
    Args:
        num_episodes: 生成数据的游戏次数（论文指定5次，对应1876组数据）
        noise_scale: Ornstein-Uhlenbeck噪声强度（论文IV.D节用0.1）
        env_name: 环境名（固定为月球着陆器环境，不修改）
        seed: 随机种子，确保结果可复现（默认2，论文未指定）
    
    Returns:
        x_prev: 原始状态序列，shape=[total_samples, 6]（6维：x,y,θ,ẋ,ẏ,θ_dot）
        u_prev: 控制输入序列，shape=[total_samples, 2]（2维：u₁主引擎, u₂侧引擎）
        x_next: 下一状态序列，shape=[total_samples, 6]
    """
    env = gym.make(env_name)
    env.seed(seed)
    x_prev_list: list[np.ndarray] = []
    u_prev_list: list[np.ndarray] = []
    x_next_list: list[np.ndarray] = []

    # 论文指定的Ornstein-Uhlenbeck噪声（用于数据探索）
    def _ou_noise(prev_noise: np.ndarray) -> np.ndarray:
        theta = 0.15  # 噪声衰减系数（论文默认值）
        return theta * (-prev_noise) + noise_scale * np.random.randn(*prev_noise.shape)

    for _ in range(num_episodes):
        x_prev = env.reset()
        prev_noise = np.zeros(env.action_space.shape[0])
        done = False
        while not done:
            # 生成带噪声的随机控制（论文：未训练RL策略）
            u_prev = env.action_space.sample() + _ou_noise(prev_noise)
            u_prev = np.clip(u_prev, env.action_space.low, env.action_space.high)
            prev_noise = _ou_noise(prev_noise)

            # 交互获取下一状态
            x_next, _, done, _ = env.step(u_prev)

            # 存储数据（严格匹配论文6维状态、2维控制）
            x_prev_list.append(x_prev[0:6])  # 取前6维状态（忽略第7、8维）
            u_prev_list.append(u_prev)
            x_next_list.append(x_next[0:6])  # 取前6维状态（忽略第7、8维）

            x_prev = x_next

    env.close()
    # 转换为numpy数组（类型锁定为float32，适配PyTorch）
    x_prev = np.array(x_prev_list, dtype=np.float32)
    u_prev = np.array(u_prev_list, dtype=np.float32)
    x_next = np.array(x_next_list, dtype=np.float32)
    # 保存至/data/lunar_lander_data_seed{seed}_episodes{num_episodes}.npz
    # 检查是否有data目录，没有则创建
    if not os.path.exists("./data"):
        os.makedirs("./data")
    np.savez_compressed(
        f"./data/lunar_lander_data_seed{seed}_episodes{num_episodes}.npz",
        x_prev=x_prev,
        u_prev=u_prev,
        x_next=x_next
    )
    print(f"数据生成完成：{x_prev.shape[0]}组数据（论文目标1876组）")
    return x_prev, u_prev, x_next


def generate_lunar_lander_data_ksteps(
    num_episodes: int = 10,
    noise_scale: float = 0.1,
    env_name: str = "LunarLanderContinuous-v2",
    seed: int = 2,
    K_steps: int = 15,
    window_step: int = 15  # 滑动窗口步长：默认=K_steps（无重叠），设1为全重叠
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成论文IV.D节月球着陆器K步训练数据：随机策略+Ornstein-Uhlenbeck噪声
    新增K步连续序列提取，输出格式适配论文K-steps损失训练（Eq.14）
    
    Args:
        num_episodes: 生成数据的游戏次数（论文指定5次≈1876组单步数据，10次≈3700组）
        noise_scale: Ornstein-Uhlenbeck噪声强度（论文IV.D节用0.1，确保探索性）
        env_name: 环境名（固定为月球着陆器，不支持修改）
        seed: 随机种子（确保数据可复现，默认2）
        K_steps: 目标连续序列长度（论文IV.B节指定15步，用于K-steps损失）
        window_step: 滑动窗口步长（提取K步序列时的步长，默认=K_steps无重叠，设1为全重叠）
    
    Returns:
        x_seq: K步状态序列，shape=[total_k_samples, K_steps, 6] 
               （6维状态：x,y,θ,ẋ,ẏ,θ_dot，对应论文IV.D节定义）
        u_seq: K步控制序列，shape=[total_k_samples, K_steps, 2]
               （2维控制：u₁主引擎推力，u₂侧引擎推力）
        x_next_seq: K步下一状态序列，shape=[total_k_samples, K_steps, 6]
                   （x_next_seq[i,t] = F(x_seq[i,t], u_seq[i,t])，对应论文时序关系）
    """
    # 1. 初始化环境与随机种子（确保可复现）
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    
    # 存储每个episode的完整时序数据（单步）
    all_episode_x: list[np.ndarray] = []  # 每个元素：[episode_len, 6]（单episode的所有状态）
    all_episode_u: list[np.ndarray] = []  # 每个元素：[episode_len, 2]（单episode的所有控制）
    all_episode_x_next: list[np.ndarray] = []  # 每个元素：[episode_len, 6]（单episode的所有下一状态）

    # 2. 论文指定的Ornstein-Uhlenbeck噪声（用于随机探索，避免数据分布过窄）
    def _ou_noise(prev_noise: np.ndarray) -> np.ndarray:
        theta = 0.15  # 噪声衰减系数（论文默认值，平衡探索稳定性）
        sigma = noise_scale  # 噪声强度（用户可调整，默认0.1）
        return theta * (-prev_noise) + sigma * np.random.randn(*prev_noise.shape)

    # 3. 生成每个episode的单步时序数据
    print(f"开始生成{num_episodes}个episode的单步数据（K_steps={K_steps}）...")
    for ep in range(num_episodes):
        x_prev = env.reset()  # 初始状态（原环境输出8维，后续取前6维）
        prev_noise = np.zeros(env.action_space.shape[0])  # 噪声初始值（2维，对应两个引擎）
        done = False
        
        # 存储当前episode的单步数据
        ep_x = []
        ep_u = []
        ep_x_next = []
        
        while not done:
            # 3.1 生成带OU噪声的随机控制（论文IV.D节：未训练RL策略，纯随机探索）
            u_rand = env.action_space.sample()  # 基础随机控制（[-1,1]范围）
            u_noise = _ou_noise(prev_noise)     # OU噪声（平滑随机波动）
            u_prev = u_rand + u_noise
            u_prev = np.clip(u_prev, env.action_space.low, env.action_space.high)  # 物理约束裁剪
            prev_noise = u_noise  # 更新噪声状态（确保时序相关性）
            
            # 3.2 与环境交互，获取下一状态
            x_next, _, done, _ = env.step(u_prev)
            
            # 3.3 存储单步数据（严格取前6维状态，符合论文IV.D节定义）
            ep_x.append(x_prev[0:6])       # 当前状态：x,y,θ,ẋ,ẏ,θ_dot
            ep_u.append(u_prev)            # 当前控制：u₁, u₂
            ep_x_next.append(x_next[0:6])  # 下一状态：对应F(x_prev, u_prev)
            
            # 更新当前状态，进入下一步
            x_prev = x_next
        
        # 3.4 保存当前episode的完整单步序列（转换为numpy数组）
        ep_x = np.array(ep_x, dtype=np.float32)  # [episode_len, 6]
        ep_u = np.array(ep_u, dtype=np.float32)  # [episode_len, 2]
        ep_x_next = np.array(ep_x_next, dtype=np.float32)  # [episode_len, 6]
        
        all_episode_x.append(ep_x)
        all_episode_u.append(ep_u)
        all_episode_x_next.append(ep_x_next)
        
        print(f"Episode {ep+1:2d} | 单步数据长度：{len(ep_x)}（需≥{K_steps}才有效）")

    env.close()
    print(f"\n单步数据生成完成：共{len(all_episode_x)}个episode，总单步数据量：{sum(len(ep) for ep in all_episode_x)}")

    # 4. 核心新增：从单步数据中提取K步连续序列（滑动窗口法）
    x_seq_list = []  # 存储所有K步状态序列
    u_seq_list = []  # 存储所有K步控制序列
    x_next_seq_list = []  # 存储所有K步下一状态序列

    print(f"\n开始提取K步连续序列（窗口步长={window_step}）...")
    for ep_idx in range(num_episodes):
        # 获取当前episode的单步数据
        ep_x = all_episode_x[ep_idx]          # [L, 6]，L为当前episode长度
        ep_u = all_episode_u[ep_idx]          # [L, 2]
        ep_x_next = all_episode_x_next[ep_idx]# [L, 6]
        L = len(ep_x)
        
        # 跳过长度不足K_steps的episode（无法提取有效K步序列）
        if L < K_steps:
            print(f"警告：Episode {ep_idx+1}长度={L} < K_steps={K_steps}，跳过该episode")
            continue
        
        # 滑动窗口提取K步序列：起始索引范围[0, L-K_steps]，步长=window_step
        max_start_idx = L - K_steps
        for start_idx in range(0, max_start_idx + 1, window_step):
            # 提取K步状态序列：x_seq[t] = 第start_idx+t步的状态（t=0~K_steps-1）
            x_k = ep_x[start_idx : start_idx + K_steps]  # [K_steps, 6]
            # 提取对应K步控制序列：u_seq[t] = 第start_idx+t步的控制
            u_k = ep_u[start_idx : start_idx + K_steps]  # [K_steps, 2]
            # 提取对应K步下一状态序列：x_next_seq[t] = F(x_k[t], u_k[t])
            x_next_k = ep_x_next[start_idx : start_idx + K_steps]  # [K_steps, 6]
            
            # 保存到列表
            x_seq_list.append(x_k)
            u_seq_list.append(u_k)
            x_next_seq_list.append(x_next_k)

    # 5. 转换为最终numpy数组（适配论文K-steps训练输入格式）
    x_seq = np.array(x_seq_list, dtype=np.float32)  # [total_k_samples, K_steps, 6]
    u_seq = np.array(u_seq_list, dtype=np.float32)  # [total_k_samples, K_steps, 2]
    x_next_seq = np.array(x_next_seq_list, dtype=np.float32)  # [total_k_samples, K_steps, 6]

    # 6. 数据校验与日志
    total_k_samples = len(x_seq)
    print(f"\nK步序列提取完成：")
    print(f"- 有效K步样本数：{total_k_samples}（每个样本含{K_steps}步连续数据）")
    print(f"- x_seq shape: {x_seq.shape}（K步状态序列）")
    print(f"- u_seq shape: {u_seq.shape}（K步控制序列）")
    print(f"- x_next_seq shape: {x_next_seq.shape}（K步下一状态序列）")

    # 7. 保存K步数据到本地（npz压缩格式，方便后续训练加载）
    save_dir = "./data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f"{save_dir}/lunar_lander_ksteps_seed{seed}_ep{num_episodes}_K{K_steps}.npz"
    
    np.savez_compressed(
        save_path,
        x_seq=x_seq,          # K步状态序列
        u_seq=u_seq,          # K步控制序列
        x_next_seq=x_next_seq, # K步下一状态序列
        K_steps=K_steps,      # 存储K_steps参数，方便加载时校验
        seed=seed             # 存储随机种子，确保可复现
    )
    print(f"\n数据已保存至：{save_path}")

    # 8. 额外返回单步数据统计（可选，方便用户核对）
    total_single_samples = sum(len(ep) for ep in all_episode_x)
    print(f"=== 数据生成总结 ===")
    print(f"总单步数据量：{total_single_samples}组（原格式）")
    print(f"总K步样本量：{total_k_samples}组（训练格式，K_steps={K_steps}）")
    print(f"数据利用率：{total_k_samples * K_steps / total_single_samples * 100:.1f}%（窗口步长={window_step}）")

    return x_seq, u_seq, x_next_seq