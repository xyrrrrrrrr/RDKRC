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



def _ou_noise(prev_noise: np.ndarray, theta: float = 0.15, sigma: float = 0.1) -> np.ndarray:
    """Ornstein-Uhlenbeck噪声（KRBF.pdf 8.1节随机探索策略）"""
    return theta * (-prev_noise) + sigma * np.random.randn(*prev_noise.shape)


def generate_lunar_lander_data_ksteps(
    num_episodes: int = 50,
    noise_scale: float = 0.5,
    env_name: str = "LunarLanderContinuous-v2",
    seed: int = 2,
    K_steps: int = 15,
    window_step: int = 1,
    save_dir: str = "./data"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成月球着陆器K步数据+单步数据（适配KRBF单步EDMD输入）
    Returns:
        x_seq: K步状态序列 [total_k, K_steps, 6]（原格式）
        u_seq: K步控制序列 [total_k, K_steps, 2]（原格式）
        x_next_seq: K步下一状态序列 [total_k, K_steps, 6]（原格式）
        X_single: 单步状态 [n, K_total]（KRBF输入：n=6，K_total=总单步样本数）
        U_single: 单步控制 [m, K_total]（KRBF输入：m=2）
        Y_single: 单步下一状态 [n, K_total]（KRBF输入）
    """
    # 初始化环境
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    # 存储单episode时序数据
    all_ep_x = []  # 每个元素：[ep_len, 6]
    all_ep_u = []  # 每个元素：[ep_len, 2]
    all_ep_x_next = []  # 每个元素：[ep_len, 6]

    # 1. 生成单episode单步数据
    print(f"[Data Gen] 生成{num_episodes}个episode的单步数据...")
    for ep in range(num_episodes):
        x_prev = env.reset()[:6]  # 取前6维状态（x,y,θ,ẋ,ẏ,θ̇）
        prev_noise = np.zeros(2)  # 2维控制噪声
        ep_x, ep_u, ep_x_next = [], [], []
        done = False

        while not done:
            # 生成带OU噪声的随机控制（KRBF.pdf 8.1节：随机输入探索）
            u_rand = env.action_space.sample()
            u_noise = _ou_noise(prev_noise, sigma=noise_scale)
            u_prev = np.clip(u_rand + u_noise, env.action_space.low, env.action_space.high)
            
            # 环境交互
            x_next, _, done, _ = env.step(u_prev)
            x_next = x_next[:6]

            # 存储单步数据
            ep_x.append(x_prev)
            ep_u.append(u_prev)
            ep_x_next.append(x_next)

            x_prev = x_next
            prev_noise = u_noise

        # 保存当前episode数据
        all_ep_x.append(np.array(ep_x, dtype=np.float32))
        all_ep_u.append(np.array(ep_u, dtype=np.float32))
        all_ep_x_next.append(np.array(ep_x_next, dtype=np.float32))
        print(f"[Data Gen] Episode {ep+1:2d} | 长度：{len(ep_x)}步")

    env.close()

    # 2. 提取K步序列数据（原格式，兼容扩展）
    x_seq_list, u_seq_list, x_next_seq_list = [], [], []
    print(f"\n[Data Gen] 提取K步序列（K={K_steps}，窗口步长={window_step}）...")
    for ep_x, ep_u, ep_xn in zip(all_ep_x, all_ep_u, all_ep_x_next):
        L = len(ep_x)
        if L < K_steps:
            print(f"[Data Gen] 跳过短episode（长度{l} < {K_steps}）")
            continue
        # 滑动窗口提取K步序列
        for s in range(0, L - K_steps + 1, window_step):
            x_seq_list.append(ep_x[s:s+K_steps])
            u_seq_list.append(ep_u[s:s+K_steps])
            x_next_seq_list.append(ep_xn[s:s+K_steps])

    x_seq = np.array(x_seq_list, dtype=np.float32)  # [total_k, K_steps, 6]
    u_seq = np.array(u_seq_list, dtype=np.float32)  # [total_k, K_steps, 2]
    x_next_seq = np.array(x_next_seq_list, dtype=np.float32)  # [total_k, K_steps, 6]

    # 3. 展平为KRBF所需单步数据（n×K_total，m×K_total）
    X_single = np.concatenate(all_ep_x, axis=0).T  # [6, K_total]
    U_single = np.concatenate(all_ep_u, axis=0).T  # [2, K_total]
    Y_single = np.concatenate(all_ep_x_next, axis=0).T  # [6, K_total]

    # 4. 保存数据
    save_path = f"{save_dir}/lunar_lander_ksteps_seed{seed}_ep{num_episodes}.npz"
    np.savez_compressed(
        save_path,
        x_seq=x_seq, u_seq=u_seq, x_next_seq=x_next_seq,
        X_single=X_single, U_single=U_single, Y_single=Y_single,
        K_steps=K_steps, seed=seed
    )
    print(f"\n[Data Gen] 数据保存至：{save_path}")
    print(f"[Data Gen] 单步数据规模：X_single={X_single.shape}, U_single={U_single.shape}")

    return x_seq, u_seq, x_next_seq, X_single, U_single, Y_single

def load_lunar_lander_data(
    load_path: str = "./data/lunar_lander_ksteps_seed2_ep50.npz"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载预生成的月球着陆器数据（KRBF专用）"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"数据文件不存在：{load_path}")
    data = np.load(load_path)
    print(f"[Data Load] 加载数据：{load_path}")
    return (
        data["x_seq"], data["u_seq"], data["x_next_seq"],
        data["X_single"], data["U_single"], data["Y_single"]
    )