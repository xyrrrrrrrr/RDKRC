import gym
import os
import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, List, Optional
from tqdm import trange


def _ou_noise(prev_noise: np.ndarray, theta: float = 0.15, sigma: float = 0.1) -> np.ndarray:
    """生成Ornstein-Uhlenbeck噪声"""
    return theta * (-prev_noise) + sigma * np.random.randn(*prev_noise.shape)


def compute_high_order_derivatives(
    state_seq: np.ndarray,
    dt: float,
    n_deriv: int,
    filter_window: int = 5
) -> List[np.ndarray]:
    """计算状态序列的高阶导数"""
    ep_len, n = state_seq.shape
    derivatives = []
    current_seq = state_seq.copy()

    # 确保滤波窗口为奇数
    filter_window = filter_window if filter_window % 2 == 1 else filter_window + 1

    for _ in range(n_deriv):
        # 中心差分计算当前阶导数
        deriv = np.zeros_like(current_seq)
        for i in range(1, ep_len - 1):
            deriv[i] = (current_seq[i+1] - current_seq[i-1]) / (2 * dt)
        # 边界样本处理
        deriv[0] = (current_seq[1] - current_seq[0]) / dt
        deriv[-1] = (current_seq[-1] - current_seq[-2]) / dt

        # 平滑滤波
        if filter_window > 1 and ep_len > filter_window:
            deriv = savgol_filter(deriv, window_length=filter_window, polyorder=2, axis=0)

        derivatives.append(deriv)
        current_seq = deriv

    return derivatives


def generate_training_data(
    env_name: str,
    n: int,
    m: int,
    n_deriv: int,
    dt: float,
    K_steps: int = 15,
    window_step: int = 1,
    num_episodes: int = 200,
    noise_scale: float = 0.5,
    filter_window: int = 1,
    save_path: str = "./data",
    seed: int = 2
) -> None:
    """生成训练数据（保持原有功能不变）"""
    # 初始化环境与随机种子
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    os.makedirs(save_path, exist_ok=True)

    # 存储数据的列表
    X_list, U_list, Y_list = [], [], []
    dX_list = [[] for _ in range(n_deriv)]
    X_seq_list, U_seq_list, Y_seq_list = [], [], []
    dX_seq_list = []

    print(f"[DataGen] 生成{env_name}训练数据：{num_episodes}回合，K_steps={K_steps}...")
    for ep in trange(num_episodes):
        # 采集当前episode的完整连续数据
        state_prev = env.reset()[:n]
        ep_state_full = [state_prev]
        ep_control_full = []
        prev_noise = np.zeros(m)
        done = False

        while not done:
            # 生成带OU噪声的随机控制
            u_rand = env.action_space.sample()
            u_noise = _ou_noise(prev_noise, sigma=noise_scale)
            u_curr = np.clip(u_rand + u_noise, env.action_space.low, env.action_space.high)
            # 环境交互获取下一状态
            state_next, _, done, _ = env.step(u_curr)
            state_next = state_next[:n]
            # 追加到完整序列
            ep_state_full.append(state_next)
            ep_control_full.append(u_curr)
            prev_noise = u_noise

        # 转换为numpy数组
        ep_state_full = np.array(ep_state_full, dtype=np.float32)
        ep_control_full = np.array(ep_control_full, dtype=np.float32)
        ep_len_full = len(ep_state_full)

        # 提取K步序列数据
        max_start_idx = ep_len_full - K_steps - 1
        if max_start_idx >= 0:
            for start_idx in range(0, max_start_idx + 1, window_step):
                X_seq = ep_state_full[start_idx : start_idx + K_steps]
                U_seq = ep_control_full[start_idx : start_idx + K_steps]
                Y_seq = ep_state_full[start_idx + 1 : start_idx + K_steps + 1]
                ep_deriv_full = compute_high_order_derivatives(ep_state_full, dt, n_deriv, filter_window)
                dX_seq = np.array([deriv[start_idx : start_idx + K_steps] for deriv in ep_deriv_full], dtype=np.float32)

                X_seq_list.append(X_seq)
                U_seq_list.append(U_seq)
                Y_seq_list.append(Y_seq)
                dX_seq_list.append(dX_seq)

        # 提取单步数据
        ep_len = len(ep_control_full)
        if ep_len < n_deriv + 1:
            continue
        start_idx_single = n_deriv // 2
        end_idx_single = ep_len_full - (n_deriv // 2)
        valid_len_single = end_idx_single - start_idx_single - 1

        X_ep = ep_state_full[start_idx_single : end_idx_single - 1]
        Y_ep = ep_state_full[start_idx_single + 1 : end_idx_single]
        U_ep = ep_control_full[start_idx_single : start_idx_single + valid_len_single]
        
        ep_deriv_single = compute_high_order_derivatives(ep_state_full, dt, n_deriv, filter_window)
        for i in range(n_deriv):
            dX_ep = ep_deriv_single[i][start_idx_single : end_idx_single - 1]
            dX_list[i].append(dX_ep)

        X_list.append(X_ep)
        U_list.append(U_ep)
        Y_list.append(Y_ep)

    env.close()

    # 转换为最终numpy数组
    X = np.concatenate(X_list, axis=0) if X_list else np.array([], dtype=np.float32)
    U = np.concatenate(U_list, axis=0) if U_list else np.array([], dtype=np.float32)
    Y = np.concatenate(Y_list, axis=0) if Y_list else np.array([], dtype=np.float32)
    dX = np.array([np.concatenate(dl, axis=0) for dl in dX_list], dtype=np.float32) if dX_list[0] else np.array([], dtype=np.float32)
    
    X_seq = np.array(X_seq_list, dtype=np.float32) if X_seq_list else np.array([], dtype=np.float32)
    U_seq = np.array(U_seq_list, dtype=np.float32) if U_seq_list else np.array([], dtype=np.float32)
    Y_seq = np.array(Y_seq_list, dtype=np.float32) if Y_seq_list else np.array([], dtype=np.float32)
    dX_seq = np.array(dX_seq_list, dtype=np.float32) if dX_seq_list else np.array([], dtype=np.float32)

    # 保存元数据
    dt_arr = np.array(dt, dtype=np.float32)
    seed_arr = np.array(seed, dtype=np.int32)
    n_deriv_arr = np.array(n_deriv, dtype=np.int32)
    K_steps_arr = np.array(K_steps, dtype=np.int32)
    window_step_arr = np.array(window_step, dtype=np.int32)

    save_file = os.path.join(save_path, f"train_data_{env_name}_n{n}_m{m}_deriv{n_deriv}_K{K_steps}_seed{seed}.npz")
    np.savez_compressed(
        save_file,
        X=X, U=U, Y=Y, dX=dX,
        X_seq=X_seq, U_seq=U_seq, Y_seq=Y_seq, dX_seq=dX_seq,
        dt=dt_arr, seed=seed_arr, n_deriv=n_deriv_arr, K_steps=K_steps_arr, window_step=window_step_arr
    )
    print(f"[DataGen] 训练数据保存至：{save_file}")
    print(f"[DataGen] 规模：单步{X.shape[0]}个样本 | K步{X_seq.shape[0]}个序列（每个{K_steps}步）")


def generate_test_data(
    env_name: str,
    n: int,
    m: int,
    K_steps: int = 15,
    num_episodes: int = 100,
    max_steps: int = 500,
    target_state: Optional[np.ndarray] = None,
    save_path: str = "./data",
    seed: int = 2,
    extended_length: bool = False  # 新增参数：是否生成2*K_steps长度数据
) -> None:
    """生成测试数据，新增支持2*K_steps长度的轨迹数据"""
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    os.makedirs(save_path, exist_ok=True)

    # 存储测试数据的列表
    trajectories_list = []
    # 普通K步和扩展2*K步序列数据
    test_X_seq_list, test_U_seq_list, test_Y_seq_list = [], [], []
    extended_X_seq_list, extended_U_seq_list, extended_Y_seq_list = [], [], []  # 新增：2*K步序列
    
    # 确定序列长度
    seq_length = 2 * K_steps if extended_length else K_steps
    print(f"[DataGen] 生成{env_name}测试数据：{num_episodes}回合，序列长度={seq_length}...")
    
    for ep in trange(num_episodes):
        # 采集完整轨迹
        state_prev = env.reset()[:n]
        ep_trajectory = [state_prev]
        ep_control = []
        done = False
        step = 0

        while not done and step < max_steps:
            u_rand = env.action_space.sample()
            state_next, _, done, _ = env.step(u_rand)
            state_next = state_next[:n]
            ep_trajectory.append(state_next)
            ep_control.append(u_rand)
            state_prev = state_next
            step += 1

        # 转换为数组并追加到列表
        ep_trajectory = np.array(ep_trajectory, dtype=np.float32)
        ep_control = np.array(ep_control, dtype=np.float32)
        trajectories_list.append(ep_trajectory)

        # 提取当前episode的K步序列和2*K步序列
        ep_len_full = len(ep_trajectory)
        
        # 提取标准K步序列
        max_start_idx = ep_len_full - K_steps - 1
        if max_start_idx >= 0:
            start_idx = max_start_idx // 2
            test_X_seq = ep_trajectory[start_idx : start_idx + K_steps]
            test_U_seq = ep_control[start_idx : start_idx + K_steps]
            test_Y_seq = ep_trajectory[start_idx + 1 : start_idx + K_steps + 1]
            test_X_seq_list.append(test_X_seq)
            test_U_seq_list.append(test_U_seq)
            test_Y_seq_list.append(test_Y_seq)
        
        # 新增：提取2*K步序列
        if extended_length:
            max_extended_start_idx = ep_len_full - 2 * K_steps - 1
            if max_extended_start_idx >= 0:
                start_idx_ext = max_extended_start_idx // 2
                extended_X_seq = ep_trajectory[start_idx_ext : start_idx_ext + 2*K_steps]
                extended_U_seq = ep_control[start_idx_ext : start_idx_ext + 2*K_steps]
                extended_Y_seq = ep_trajectory[start_idx_ext + 1 : start_idx_ext + 2*K_steps + 1]
                extended_X_seq_list.append(extended_X_seq)
                extended_U_seq_list.append(extended_U_seq)
                extended_Y_seq_list.append(extended_Y_seq)

    env.close()

    # 转换为numpy数组
    test_X_seq = np.array(test_X_seq_list, dtype=np.float32) if test_X_seq_list else np.array([], dtype=np.float32)
    test_U_seq = np.array(test_U_seq_list, dtype=np.float32) if test_U_seq_list else np.array([], dtype=np.float32)
    test_Y_seq = np.array(test_Y_seq_list, dtype=np.float32) if test_Y_seq_list else np.array([], dtype=np.float32)
    
    # 新增：处理2*K步序列数据
    extended_X_seq = np.array(extended_X_seq_list, dtype=np.float32) if extended_X_seq_list else np.array([], dtype=np.float32)
    extended_U_seq = np.array(extended_U_seq_list, dtype=np.float32) if extended_U_seq_list else np.array([], dtype=np.float32)
    extended_Y_seq = np.array(extended_Y_seq_list, dtype=np.float32) if extended_Y_seq_list else np.array([], dtype=np.float32)
    
    # 元数据
    seed_arr = np.array(seed, dtype=np.int32)
    max_steps_arr = np.array(max_steps, dtype=np.int32)
    K_steps_arr = np.array(K_steps, dtype=np.int32)
    target_state_arr = target_state if target_state is not None else np.array([], dtype=np.float32)
    seq_length_arr = np.array(seq_length, dtype=np.int32)

    # 保存文件（区分普通和扩展数据）
    file_suffix = f"_ep{num_episodes}_K{K_steps}_seed{seed}"
    if extended_length:
        file_suffix += "_extended"
        
    save_file = os.path.join(save_path, f"test_data_{env_name}{file_suffix}.npz")
    np.savez_compressed(
        save_file,
        test_X_seq=test_X_seq, test_U_seq=test_U_seq, test_Y_seq=test_Y_seq,
        extended_X_seq=extended_X_seq, extended_U_seq=extended_U_seq, extended_Y_seq=extended_Y_seq,  # 新增
        seed=seed_arr, max_steps=max_steps_arr, K_steps=K_steps_arr, 
        target_state=target_state_arr, seq_length=seq_length_arr
    )
    print(f"[DataGen] 测试数据保存至：{save_file}")


if __name__ == "__main__":
    """示例：生成月球着陆器训练/测试数据，包括2*K_steps长度的扩展测试数据"""
    # 配置参数
    env_name = "LunarLanderContinuous-v2"
    n = 6          
    m = 2          
    n_deriv = 2    
    dt = 0.01      
    K_steps = 15   
    num_train_ep = 200  
    num_test_ep = 100   
    seed = 2       
    save_path = "./data"
    target_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # 1. 生成训练数据
    # generate_training_data(
    #     env_name=env_name,
    #     n=n, m=m, n_deriv=n_deriv,
    #     dt=dt, K_steps=K_steps,
    #     num_episodes=num_train_ep,
    #     seed=seed, save_path=save_path
    # )

    # 2. 生成标准测试数据
    # generate_test_data(
    #     env_name=env_name,
    #     n=n, m=m, K_steps=K_steps,
    #     num_episodes=num_test_ep,
    #     target_state=target_state,
    #     seed=seed, save_path=save_path,
    #     extended_length=False  # 标准长度
    # )
    
    # 3. 新增：生成2*K_steps长度的扩展测试数据
    generate_test_data(
        env_name=env_name,
        n=n, m=m, K_steps=K_steps,
        num_episodes=num_test_ep,
        target_state=target_state,
        seed=seed, save_path=save_path,
        extended_length=True  # 扩展长度为2*K_steps
    )

    # 4. 加载扩展测试数据示例
    extended_test_load_path = os.path.join(save_path, f"test_data_{env_name}_ep{num_test_ep}_K{K_steps}_seed{seed}_extended.npz")
    if os.path.exists(extended_test_load_path):
        extended_test_data = np.load(extended_test_load_path)
        print(f"\n[加载示例] 扩展测试数据 - 2*K步X_seq形状：{extended_test_data['extended_X_seq'].shape}")
        print(f"[加载示例] 扩展测试数据 - 序列长度：{extended_test_data['seq_length'].item()}")
        extended_test_data.close()
