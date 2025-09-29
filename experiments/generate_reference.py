import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def generate_reference_trajectory(total_steps: int, freq: float = 0.02) -> List[np.ndarray]:
    """
    生成参考轨迹（水平正弦摆动+垂直匀速下降）
    符合文档IV.D节对动态轨迹的要求：包含慢变（y下降）和快变（x摆动）成分
    """
    ref_traj = []
    for t in range(total_steps):
        # x方向：正弦摆动（快变成分）
        x = 0.5 * np.sin(freq * t)  # 振幅0.8，频率0.02
        # y方向：从1.2m匀速下降到0（慢变成分）
        y = max(1.4 - 0.003 * t, 0.0)
        # 姿态θ、速度ẋ、ẏ、θ̇均为0（理想轨迹）
        ref_state = np.array([x, y, 0.0, 0.0, 0.0, 0.0])
        ref_traj.append(ref_state)
    return ref_traj


# 测试：生成并可视化参考轨迹
if __name__ == "__main__":
    total_steps = 500
    ref_traj = generate_reference_trajectory(total_steps)
    
    # 提取x-y轨迹
    x_coords = [state[0] for state in ref_traj]
    y_coords = [state[1] for state in ref_traj]
    
    # 绘制轨迹
    plt.figure(figsize=(8, 6))
    plt.plot(x_coords, y_coords, "b-", linewidth=2, label="Reference Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Reference Trajectory (Sinusoidal Horizontal + Uniform Vertical Descent)")
    plt.grid(True)
    plt.legend()
    plt.savefig("reference_trajectory.png", dpi=300)
    plt.show()
    