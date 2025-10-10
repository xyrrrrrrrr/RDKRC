import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple

# 设置全局字体，确保中文显示正常

def plot_error_summary(
    file_info: List[Tuple[str, str, str, str]],  # (文件路径, 模型名称, 线条样式, 标记样式)
    k_value: int = 15,
    save_path: str = "./fig/lunarlander/prediction_error_summary.png",
    xlim: Tuple[int, int] = (0, 30),
    ylim: Tuple[float, float] = (-5, 0)
) -> None:
    """
    绘制多模型预测误差汇总图
    
    参数:
        file_info: 包含文件路径、模型名称、线条样式、标记样式的元组列表
        k_value: K值，用于标题显示
        save_path: 图片保存路径
        xlim: x轴范围
        ylim: y轴范围
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 颜色列表（使用更鲜明的对比色）
    colors = [
        "#2E7D32",  # 深绿
        "#C62828",  # 深红
        "#1565C0",  # 深蓝
        "#FF8F00",  # 橙色
        "#6A1B9A"   # 紫色
    ]
    
    # 遍历每个模型的结果文件
    for i, (file_path, label, linestyle, marker) in enumerate(file_info):
        if not os.path.exists(file_path):
            print(f"警告：文件 {file_path} 不存在，已跳过")
            continue
            
        try:
            # 读取数据
            data = np.load(file_path)
            log10_errors = data["log10_errors"]
            time_steps = np.arange(len(log10_errors))  # 时间步: 0 ~ 2*K
            
            # 绘制曲线（每5步添加一个标记点，避免过于密集）
            ax.plot(
                time_steps,
                log10_errors,
                color=colors[i % len(colors)],
                linestyle=linestyle,
                linewidth=2.5,
                marker=marker,
                markersize=6,
                markevery=5,  # 每5步显示一个标记
                label=label
            )
            print(f"已加载: {label} (数据长度: {len(log10_errors)})")
            
        except KeyError as e:
            print(f"错误：文件 {file_path} 缺少字段 {e}")
        except Exception as e:
            print(f"处理 {file_path} 时出错: {str(e)}")
    
    # 图表美化
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Log Error")
    ax.set_title(f"Lunar Lander Prediction Error Comparison")
    
    # 设置坐标轴范围
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # 添加网格（虚线，低透明度）
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 3. 绘制网格：主网格实线（粗），次网格虚线（细）
    ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.4)  # 主网格
    ax.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.2) # 次网格
    # ----------------------------------------------------------------------

    # 添加图例（边框、背景半透明）
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),  # 图例放在图外右侧
        borderaxespad=0,
        frameon=True,
        facecolor='white',
        edgecolor='gray',
        framealpha=0.9
    )
    
    # 调整布局，避免元素被截断
    plt.tight_layout()
    
    # 保存图片（高分辨率）
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"汇总图已保存至: {save_path}")
    
    # 显示图片
    plt.show()

if __name__ == "__main__":
    # 配置文件信息：(文件路径, 模型名称, 线条样式, 标记样式)
    file_config = [
        ("./results/dkn_mc2_pred_results_K15_exp4.npz", "DKN + Manifold Constraints", "-", "o"),
        ("./results/dkn_pred_results_K15_exp4.npz", "Baseline DKN", "--", "s"),
        ("./results/kderiv_pred_results_K15.npz", "KDeriv (Kernel Koopman)", "-.", "^"),
        ("./results/krbf_pred_results_K15.npz", "KRBF", ":", "D")
    ]
    
    # 绘制汇总图
    plot_error_summary(
        file_info=file_config,
        k_value=15,
        save_path="./fig/lunarlander/prediction_error_summary_final.png",
        xlim=(0, 30),
        ylim=(-5, 0)
    )
