import gym
import numpy as np
from typing import Tuple


def generate_lunar_lander_data(
    num_episodes: int = 10,
    noise_scale: float = 0.1,
    env_name: str = "LunarLanderContinuous-v2"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成论文IV.D节月球着陆器训练数据：随机策略+Ornstein-Uhlenbeck噪声
    仅用于当前月球着陆器实验，不支持其他环境。
    
    Args:
        num_episodes: 生成数据的游戏次数（论文指定5次，对应1876组数据）
        noise_scale: Ornstein-Uhlenbeck噪声强度（论文IV.D节用0.1）
        env_name: 环境名（固定为月球着陆器环境，不修改）
    
    Returns:
        x_prev: 原始状态序列，shape=[total_samples, 6]（6维：x,y,θ,ẋ,ẏ,θ_dot）
        u_prev: 控制输入序列，shape=[total_samples, 2]（2维：u₁主引擎, u₂侧引擎）
        x_next: 下一状态序列，shape=[total_samples, 6]
    """
    env = gym.make(env_name)
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

    print(f"数据生成完成：{x_prev.shape[0]}组数据（论文目标1876组）")
    return x_prev, u_prev, x_next