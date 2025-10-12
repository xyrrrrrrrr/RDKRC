import gym
import os
import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, List, Optional
from tqdm import trange

import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt


# ReplayBuffer from https://github.com/seungeunrho/minimalRL
class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        s_batch = torch.tensor(s_lst, dtype=torch.float)
        a_batch = torch.tensor(a_lst, dtype=torch.float)
        r_batch = torch.tensor(r_lst, dtype=torch.float)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float)
        done_batch = torch.tensor(done_mask_lst, dtype=torch.float)

        # r_batch = (r_batch - r_batch.mean()) / (r_batch.std() + 1e-7)

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def size(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, q_lr):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, action_dim)

        self.lr = q_lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        q = F.leaky_relu(self.fc_1(x))
        q = F.leaky_relu(self.fc_2(q))
        q = self.fc_out(q)
        return q


class DQNAgent:
    def __init__(self):
        self.state_dim     = 3
        self.action_dim    = 9  # 9개 행동 : -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0
        self.lr            = 0.01
        self.gamma         = 0.98
        self.tau           = 0.01
        self.epsilon       = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min   = 0.001
        self.buffer_size   = 100000
        self.batch_size    = 200
        self.memory        = ReplayBuffer(self.buffer_size)

        self.Q        = QNetwork(self.state_dim, self.action_dim, self.lr)
        self.Q_target = QNetwork(self.state_dim, self.action_dim, self.lr)
        self.Q_target.load_state_dict(self.Q.state_dict())

    def choose_action(self, state):
        random_number = np.random.rand()
        maxQ_action_count = 0
        if self.epsilon < random_number:
            with torch.no_grad():
                action = float(torch.argmax(self.Q(state)).numpy())
                # action = float(action.numpy())
                real_action = (action - 4) / 4
                maxQ_action_count = 1
        else:
            action = np.random.choice([n for n in range(9)])
            real_action = (action - 4) / 2  # -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0

        return action, real_action, maxQ_action_count

    def calc_target(self, mini_batch):
        s, a, r, s_prime, done = mini_batch
        with torch.no_grad():
            q_target = self.Q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + self.gamma * done * q_target
        return target

    def train_agent(self):
        mini_batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = mini_batch
        a_batch = a_batch.type(torch.int64)

        td_target = self.calc_target(mini_batch)

        #### Q train ####
        Q_a = self.Q(s_batch).gather(1, a_batch)
        q_loss = F.smooth_l1_loss(Q_a, td_target)
        self.Q.optimizer.zero_grad()
        q_loss.mean().backward()
        self.Q.optimizer.step()
        #### Q train ####

        #### Q soft-update ####
        for param_target, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)



def compute_high_order_derivatives(
    state_seq: np.ndarray,
    dt: float,
    n_deriv: int,
    filter_window: int = 5
) -> List[np.ndarray]:
    """计算状态序列的高阶导数（复用原逻辑）"""
    ep_len, n = state_seq.shape
    derivatives = []
    current_seq = state_seq.copy()
    filter_window = filter_window if filter_window % 2 == 1 else filter_window + 1

    for _ in range(n_deriv):
        deriv = np.zeros_like(current_seq)
        for i in range(1, ep_len - 1):
            deriv[i] = (current_seq[i+1] - current_seq[i-1]) / (2 * dt)
        deriv[0] = (current_seq[1] - current_seq[0]) / dt
        deriv[-1] = (current_seq[-1] - current_seq[-2]) / dt
        if filter_window > 1 and ep_len > filter_window:
            deriv = savgol_filter(deriv, window_length=filter_window, polyorder=2, axis=0)
        derivatives.append(deriv)
        current_seq = deriv
    return derivatives


def generate_pendulum_training_data(
    DQN: DQNAgent,
    n_deriv: int = 2,
    dt: float = 0.05,  # Pendulum默认步长
    K_steps: int = 10,
    num_episodes: int = 300,
    noise_scale: float = 1.0,  # 动作噪声缩放（Pendulum动作范围小）
    save_path: str = "./data/pendulum",
    seed: int = 2
) -> None:
    """生成Pendulum训练数据（适配3维状态+1维动作）"""
    env_name = "Pendulum-v0"
    n = 3  # 状态维度：[cosθ, sinθ, θ_dot]
    m = 1  # 动作维度：力矩
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    os.makedirs(save_path, exist_ok=True)

    X_list, U_list, Y_list = [], [], []
    dX_list = [[] for _ in range(n_deriv)]
    X_seq_list, U_seq_list, Y_seq_list = [], [], []
    dX_seq_list = []

    print(f"[DataGen] 生成{env_name}训练数据：{num_episodes}回合，K_steps={K_steps}...")
    for ep in trange(num_episodes):
        state_prev = env.reset()[:n]
        ep_state_full = [state_prev]
        ep_control_full = []
        done = False

        while not done:
            # 带噪声的随机控制（限制力矩范围[-2,2]）
            state_prev = torch.tensor(state_prev, dtype=torch.float32).unsqueeze(0)
            u_curr = DQN.choose_action(state_prev)[1]
            state_next, _, done, _ = env.step([u_curr])
            state_next = state_next[:n]
            ep_state_full.append(state_next)
            ep_control_full.append([u_curr])

        ep_state_full = np.array(ep_state_full, dtype=np.float32)
        ep_control_full = np.array(ep_control_full, dtype=np.float32)
        ep_len_full = len(ep_state_full)

        # 提取K步序列
        max_start_idx = ep_len_full - K_steps - 1
        if max_start_idx >= 0:
            for start_idx in range(0, max_start_idx + 1, 1):  # 步长固定为1
                X_seq = ep_state_full[start_idx : start_idx + K_steps]
                U_seq = ep_control_full[start_idx : start_idx + K_steps]
                Y_seq = ep_state_full[start_idx + 1 : start_idx + K_steps + 1]
                ep_deriv_full = compute_high_order_derivatives(ep_state_full, dt, n_deriv)
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
        
        ep_deriv_single = compute_high_order_derivatives(ep_state_full, dt, n_deriv)
        for i in range(n_deriv):
            dX_ep = ep_deriv_single[i][start_idx_single : end_idx_single - 1]
            dX_list[i].append(dX_ep)

        X_list.append(X_ep)
        U_list.append(U_ep)
        Y_list.append(Y_ep)

    env.close()

    # 保存数据
    X = np.concatenate(X_list, axis=0) if X_list else np.array([], dtype=np.float32)
    U = np.concatenate(U_list, axis=0) if U_list else np.array([], dtype=np.float32)
    Y = np.concatenate(Y_list, axis=0) if Y_list else np.array([], dtype=np.float32)
    dX = np.array([np.concatenate(dl, axis=0) for dl in dX_list], dtype=np.float32) if dX_list[0] else np.array([], dtype=np.float32)
    
    X_seq = np.array(X_seq_list, dtype=np.float32) if X_seq_list else np.array([], dtype=np.float32)
    U_seq = np.array(U_seq_list, dtype=np.float32) if U_seq_list else np.array([], dtype=np.float32)
    Y_seq = np.array(Y_seq_list, dtype=np.float32) if Y_seq_list else np.array([], dtype=np.float32)
    dX_seq = np.array(dX_seq_list, dtype=np.float32) if dX_seq_list else np.array([], dtype=np.float32)

    save_file = os.path.join(save_path, f"train_data_{env_name}_n{n}_m{m}_deriv{n_deriv}_K{K_steps}_seed{seed}.npz")
    np.savez_compressed(
        save_file,
        X=X, U=U, Y=Y, dX=dX,
        X_seq=X_seq, U_seq=U_seq, Y_seq=Y_seq, dX_seq=dX_seq,
        dt=dt, seed=seed, n_deriv=n_deriv, K_steps=K_steps
    )
    print(f"[DataGen] 数据保存至：{save_file} | 单步样本：{X.shape[0]} | K步序列：{X_seq.shape[0]}")


def generate_pendulum_test_data(
    DQN: DQNAgent,
    K_steps: int = 10,
    num_episodes: int = 100,
    max_steps: int = 200,  # Pendulum典型最大步数
    save_path: str = "./data/pendulum",
    seed: int = 2,
    extended_length: bool = False
) -> None:
    """生成Pendulum测试数据"""
    env_name = "Pendulum-v0"
    n = 3
    m = 1

    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    os.makedirs(save_path, exist_ok=True)

    trajectories_list = []
    test_X_seq_list, test_U_seq_list, test_Y_seq_list = [], [], []
    extended_X_seq_list, extended_U_seq_list, extended_Y_seq_list = [], [], []
    
    seq_length = 2 * K_steps if extended_length else K_steps
    print(f"[DataGen] 生成{env_name}测试数据：{num_episodes}回合，序列长度={seq_length}...")
    
    for ep in trange(num_episodes):
        state_prev = env.reset()[:n]
        ep_trajectory = [state_prev]
        ep_control = []
        done = False
        step = 0

        while not done and step < max_steps:
            state_prev = torch.tensor(state_prev, dtype=torch.float32).unsqueeze(0)
            u_curr = DQN.choose_action(state_prev)[1]
            state_next, _, done, _ = env.step([u_curr])
            state_next = state_next[:n]
            ep_trajectory.append(state_next)
            ep_control.append([u_curr])
            state_prev = state_next
            step += 1

        ep_trajectory = np.array(ep_trajectory, dtype=np.float32)
        ep_control = np.array(ep_control, dtype=np.float32)
        trajectories_list.append(ep_trajectory)

        # 提取K步序列
        ep_len_full = len(ep_trajectory)
        max_start_idx = ep_len_full - K_steps - 1
        if max_start_idx >= 0:
            start_idx = max_start_idx // 2
            test_X_seq = ep_trajectory[start_idx : start_idx + K_steps]
            test_U_seq = ep_control[start_idx : start_idx + K_steps]
            test_Y_seq = ep_trajectory[start_idx + 1 : start_idx + K_steps + 1]
            test_X_seq_list.append(test_X_seq)
            test_U_seq_list.append(test_U_seq)
            test_Y_seq_list.append(test_Y_seq)
        
        # 提取2*K步序列
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

    # 保存测试数据
    test_X_seq = np.array(test_X_seq_list, dtype=np.float32) if test_X_seq_list else np.array([], dtype=np.float32)
    test_U_seq = np.array(test_U_seq_list, dtype=np.float32) if test_U_seq_list else np.array([], dtype=np.float32)
    test_Y_seq = np.array(test_Y_seq_list, dtype=np.float32) if test_Y_seq_list else np.array([], dtype=np.float32)
    extended_X_seq = np.array(extended_X_seq_list, dtype=np.float32) if extended_X_seq_list else np.array([], dtype=np.float32)
    extended_U_seq = np.array(extended_U_seq_list, dtype=np.float32) if extended_U_seq_list else np.array([], dtype=np.float32)
    extended_Y_seq = np.array(extended_Y_seq_list, dtype=np.float32) if extended_Y_seq_list else np.array([], dtype=np.float32)

    save_file = os.path.join(save_path, f"test_data_{env_name}_n{n}_m{m}_K{K_steps}_seed{seed}_extended.npz")
    np.savez_compressed(
        save_file,
        test_X_seq=test_X_seq, test_U_seq=test_U_seq, test_Y_seq=test_Y_seq,
        extended_X_seq=extended_X_seq, extended_U_seq=extended_U_seq, extended_Y_seq=extended_Y_seq,
        max_steps=max_steps, seed=seed
    )
    print(f"[DataGen] 测试数据保存至：{save_file}")


if __name__ == '__main__':

    ###### logging ######
    log_name = '0404'


    ###### logging ######

    agent = DQNAgent()
    if os.path.exists("./data/pendulum/DQN_Q_network.pt"):
        agent.Q.load_state_dict(torch.load("./data/pendulum/DQN_Q_network.pt"))
        agent.Q_target.load_state_dict(agent.Q.state_dict())
        agent.Q.eval()
        agent.Q_target.eval()
    else: 
        print("모델로드실패")
        env = gym.make('Pendulum-v0')

        EPISODE = 500
        print_once = True
        score_list = []  # [-2000]

        for EP in range(EPISODE):
            state = env.reset()
            score, done = 0.0, False
            maxQ_action_count = 0

            while not done:
                action, real_action, count = agent.choose_action(torch.FloatTensor(state))

                state_prime, reward, done, _ = env.step([real_action])

                agent.memory.put((state, action, reward, state_prime, done))

                score += reward
                maxQ_action_count += count

                state = state_prime

                if agent.memory.size() > 1000:  # 1000개의 [s,a,r,s']이 쌓이면 학습 시작
                    if print_once: print("학습시작!")
                    print_once = False
                    agent.train_agent()

                print("EP:{}, Avg_Score:{:.1f}, MaxQ_Action_Count:{}, Epsilon:{:.5f}".format(EP, score, maxQ_action_count, agent.epsilon))
                score_list.append(score)

                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay

            # score = [float(s) for s in data]
            torch.save(agent.Q.state_dict(), "./data/pendulum/DQN_Q_network.pt")
            plt.plot(score_list)
            plt.show()
        agent.Q.eval()
        agent.Q_target.eval()

    generate_pendulum_training_data(
        DQN=agent,
        n_deriv=2,
        K_steps=15,
        num_episodes=500,
        save_path="./data/pendulum"
    )
    # 生成测试数据示例
    generate_pendulum_test_data(
        DQN=agent,
        K_steps=15,
        num_episodes=100,
        extended_length=True,
        save_path="./data/pendulum"
    )