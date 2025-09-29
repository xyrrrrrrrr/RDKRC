import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")  # 忽略数值计算中的警告


def load_abc_matrix(version: str, seed: int = 50, data_dir: str = ".") -> Dict[str, np.ndarray]:
    """
    读取指定版本和种子的 A、B、C 矩阵（适配用户的 npz 文件名格式）
    
    Args:
        version: 模型版本（如 "v1"、"v2"）
        seed: 随机种子（与保存时一致，默认2）
        data_dir: 文件所在目录（默认当前目录）
    
    Returns:
        abc_dict: 包含 A、B、C 矩阵的字典
    """
    # 匹配用户的文件命名格式：lunar_lander_ABC_{version}_seed{seed}.npz
    file_path = f"{data_dir}/lunar_lander_ABC_{version}_seed{seed}.npz"
    try:
        data = np.load(file_path)
        # 验证矩阵维度（确保符合 DKRC 定义：A[N×N], B[N×m], C[n×N]）
        A = data["A"]
        B = data["B"]
        C = data["C"]
        assert A.ndim == 2 and A.shape[0] == A.shape[1], f"A矩阵需为方阵，当前形状{A.shape}"
        assert B.ndim == 2 and B.shape[0] == A.shape[0], f"B矩阵行数需与A一致，当前B.shape{B.shape}"
        assert C.ndim == 2 and C.shape[1] == A.shape[0], f"C矩阵列数需与A一致，当前C.shape{C.shape}"
        print(f"成功读取 {version} 版本（seed={seed}）：A[{A.shape[0]}×{A.shape[1]}], B[{B.shape[0]}×{B.shape[1]}], C[{C.shape[0]}×{C.shape[1]}]")
        return {"A": A, "B": B, "C": C}
    except FileNotFoundError:
        raise FileNotFoundError(f"未找到文件：{file_path}，请检查版本和种子是否正确")
    except KeyError as e:
        raise KeyError(f"文件中缺少矩阵：{e}，请确认保存时的键为'A'、'B'、'C'")


def compute_abc_metrics(abc_dict: Dict[str, np.ndarray], psi: Optional["PsiMLP"] = None, x_star: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    计算 A、B、C 矩阵的关键指标（基于文档核心要求）
    
    Args:
        abc_dict: 包含 A、B、C 的字典
        psi: 可选，PsiMLP 网络（用于计算 C 矩阵的状态重构误差）
        x_star: 可选，目标原状态（用于重构误差计算，shape=[n]）
    
    Returns:
        metrics: 量化指标字典
    """
    A = abc_dict["A"]
    B = abc_dict["B"]
    C = abc_dict["C"]
    N = A.shape[0]  # 高维空间维度
    m = B.shape[1]  # 控制维度（月球着陆器 m=2）
    n = C.shape[0]  # 原状态维度（月球着陆器 n=6）
    
    metrics = {}

    # -------------------------- 1. A 矩阵指标（稳定性+数值特性） --------------------------
    # 1.1 特征值模长（离散系统稳定性：max_eig_norm < 1 为稳定，🔶1-37）
    eig_vals = np.linalg.eigvals(A)
    eig_norms = np.abs(eig_vals)
    metrics["A_max_eig_norm"] = np.max(eig_norms)  # 最大特征值模长（越小越稳定）
    metrics["A_mean_eig_norm"] = np.mean(eig_norms)  # 平均特征值模长
    metrics["A_eig_norm_std"] = np.std(eig_norms)    # 特征值模长标准差（越小越均匀）
    
    # 1.2 矩阵范数（数值稳定性：避免过大导致控制震荡）
    metrics["A_fro_norm"] = np.linalg.norm(A, ord="fro")  # Frobenius 范数
    metrics["A_inf_norm"] = np.linalg.norm(A, ord=np.inf)  # 无穷范数

    # -------------------------- 2. B 矩阵指标（能控性+控制强度） --------------------------
    # 2.1 能控性矩阵秩（满秩说明系统完全能控，🔶1-44）
    def compute_controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """计算能控性矩阵：[B, A*B, A²*B, ..., A^(N-1)*B]（文档 III 节 LQR 能控性要求）"""
        N = A.shape[0]
        ctrl_mat = B
        current = B
        for _ in range(1, N):
            current = A @ current
            ctrl_mat = np.hstack([ctrl_mat, current])
        return ctrl_mat
    
    ctrl_mat = compute_controllability_matrix(A, B)
    metrics["B_ctrl_rank"] = np.linalg.matrix_rank(ctrl_mat)  # 能控性矩阵秩（满秩=N为优）
    metrics["B_ctrl_rank_ratio"] = metrics["B_ctrl_rank"] / N  # 秩占比（越接近1越好）
    
    # 2.2 列范数（控制输入对各高维状态的影响强度）
    B_col_norms = np.linalg.norm(B, axis=0)  # 每列范数（对应每个控制输入的影响）
    metrics["B_mean_col_norm"] = np.mean(B_col_norms)
    metrics["B_max_col_norm"] = np.max(B_col_norms)

    # -------------------------- 3. C 矩阵指标（状态重构精度） --------------------------
    # 3.1 矩阵范数（数值规模）
    metrics["C_fro_norm"] = np.linalg.norm(C, ord="fro")
    C_row_norms = np.linalg.norm(C, axis=1)  # 每行范数（对应原状态各维度的重构权重）
    metrics["C_mean_row_norm"] = np.mean(C_row_norms)
    
    # 3.2 状态重构误差（需 Psi 网络，可选，🔶1-42 Equation 9）
    if psi is not None and x_star is not None:
        # 生成随机原状态样本（覆盖月球着陆器状态空间，🔶1-80）
        np.random.seed(2)
        x_samples = np.random.uniform(
            low=[-1.5, 0.0, -np.pi, -5.0, -5.0, -8.0],
            high=[1.5, 1.5, np.pi, 5.0, 5.0, 8.0],
            size=(100, n)  # 100个样本，统计平均误差
        )
        recon_errors = []
        psi.eval()
        with torch.no_grad():
            device = next(psi.parameters()).device
            x_star_tensor = torch.tensor(x_star, device=device, dtype=torch.float32).unsqueeze(0)
            for x in x_samples:
                x_tensor = torch.tensor(x, device=device, dtype=torch.float32).unsqueeze(0)
                z = psi.compute_z(x_tensor, x_star_tensor)  # z = Ψ(x) - Ψ(x*)
                z_np = z.cpu().numpy().squeeze()
                # 重构原状态：x_recon = C @ (z + Ψ(x*))（因 z = Ψ(x)-Ψ(x*)，故 Ψ(x) = z + Ψ(x*)）
                psi_x_star = psi(x_star_tensor).cpu().numpy().squeeze()
                x_recon = C @ (z_np + psi_x_star)
                # 计算重构误差（L2范数）
                recon_errors.append(np.linalg.norm(x - x_recon, ord=2))
        metrics["C_mean_recon_error"] = np.mean(recon_errors)
        metrics["C_recon_error_std"] = np.std(recon_errors)
    
    return metrics


def plot_abc_comparison(v1_abc: Dict[str, np.ndarray], v2_abc: Dict[str, np.ndarray], v1_metrics: Dict[str, float], v2_metrics: Dict[str, float]):
    """
    可视化对比 v1 和 v2 的 A、B、C 矩阵关键指标（贴合文档关注重点）
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("DKRC A/B/C Matrix Comparison (v1 vs v2)", fontsize=16, fontweight="bold")
    colors = ["#1f77b4", "#ff7f0e"]  # v1蓝，v2橙

    # -------------------------- 子图1：A矩阵特征值分布（稳定性核心指标） --------------------------
    ax1 = axes[0, 0]
    # 计算特征值
    v1_eig = np.linalg.eigvals(v1_abc["A"])
    v2_eig = np.linalg.eigvals(v2_abc["A"])
    # 绘制特征值散点（实部vs虚部）
    ax1.scatter(np.real(v1_eig), np.imag(v1_eig), color=colors[0], alpha=0.6, label=f"v1 (max_norm={v1_metrics['A_max_eig_norm']:.3f})")
    ax1.scatter(np.real(v2_eig), np.imag(v2_eig), color=colors[1], alpha=0.6, label=f"v2 (max_norm={v2_metrics['A_max_eig_norm']:.3f})")
    # 绘制单位圆（稳定性边界：离散系统特征值需在圆内）
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.5, label="Unit Circle (Stability Boundary)")
    ax1.set_xlabel("Real Part of Eigenvalue", fontsize=12)
    ax1.set_ylabel("Imaginary Part of Eigenvalue", fontsize=12)
    ax1.set_title("A Matrix Eigenvalue Distribution (Stability)", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # -------------------------- 子图2：能控性与稳定性核心指标对比 --------------------------
    ax2 = axes[0, 1]
    # 选择关键指标（稳定性+能控性，文档 III 节重点）
    metrics_names = [
        "A Max Eigen Norm\n(Stability, <1 is better)",
        "A Mean Eigen Norm\n(Uniformity)",
        "B Controllability Rank\n(Ratio, 1 is full rank)",
        "B Mean Column Norm\n(Control Strength)"
    ]
    v1_vals = [
        v1_metrics["A_max_eig_norm"],
        v1_metrics["A_mean_eig_norm"],
        v1_metrics["B_ctrl_rank_ratio"],
        v1_metrics["B_mean_col_norm"]
    ]
    v2_vals = [
        v2_metrics["A_max_eig_norm"],
        v2_metrics["A_mean_eig_norm"],
        v2_metrics["B_ctrl_rank_ratio"],
        v2_metrics["B_mean_col_norm"]
    ]
    # 绘制柱状图
    x = np.arange(len(metrics_names))
    width = 0.35
    ax2.bar(x - width/2, v1_vals, width, color=colors[0], label="v1")
    ax2.bar(x + width/2, v2_vals, width, color=colors[1], label="v2")
    # 添加数值标签
    for i, (v1, v2) in enumerate(zip(v1_vals, v2_vals)):
        ax2.text(i - width/2, v1 + 0.01, f"{v1:.3f}", ha="center", fontsize=10)
        ax2.text(i + width/2, v2 + 0.01, f"{v2:.3f}", ha="center", fontsize=10)
    ax2.set_xlabel("Key Metrics", fontsize=12)
    ax2.set_ylabel("Metric Value", fontsize=12)
    ax2.set_title("Core Metrics Comparison (Stability + Controllability)", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names, rotation=0, fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # -------------------------- 子图3：C矩阵状态重构误差对比（若有数据） --------------------------
    ax3 = axes[1, 0]
    if "C_mean_recon_error" in v1_metrics and "C_mean_recon_error" in v2_metrics:
        # 重构误差箱线图
        recon_data = [
            np.random.normal(v1_metrics["C_mean_recon_error"], v1_metrics["C_recon_error_std"], 100),
            np.random.normal(v2_metrics["C_mean_recon_error"], v2_metrics["C_recon_error_std"], 100)
        ]
        bp = ax3.boxplot(recon_data, labels=["v1", "v2"], patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        # 添加均值线
        ax3.axhline(y=v1_metrics["C_mean_recon_error"], color=colors[0], linestyle="--", alpha=0.8, label=f"v1 Mean: {v1_metrics['C_mean_recon_error']:.3f}")
        ax3.axhline(y=v2_metrics["C_mean_recon_error"], color=colors[1], linestyle="--", alpha=0.8, label=f"v2 Mean: {v2_metrics['C_mean_recon_error']:.3f}")
        ax3.set_xlabel("Model Version", fontsize=12)
        ax3.set_ylabel("State Reconstruction Error (L2 Norm)", fontsize=12)
        ax3.set_title("C Matrix State Reconstruction Error", fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")
    else:
        ax3.text(0.5, 0.5, "Need PsiMLP to Compute Reconstruction Error", ha="center", va="center", transform=ax3.transAxes, fontsize=12)
        ax3.set_xlabel("Model Version", fontsize=12)
        ax3.set_ylabel("Reconstruction Error", fontsize=12)
        ax3.set_title("C Matrix State Reconstruction Error (No Data)", fontsize=14)

    # -------------------------- 子图4：矩阵范数对比（数值稳定性） --------------------------
    ax4 = axes[1, 1]
    # 矩阵范数指标（Frobenius 范数，数值稳定性）
    norm_names = ["A Norm", "B Norm", "C Norm"]
    v1_norms = [
        v1_metrics["A_fro_norm"],
        np.linalg.norm(v1_abc["B"], ord="fro"),
        v1_metrics["C_fro_norm"]
    ]
    v2_norms = [
        v2_metrics["A_fro_norm"],
        np.linalg.norm(v2_abc["B"], ord="fro"),
        v2_metrics["C_fro_norm"]
    ]
    # 绘制堆叠柱状图
    x = np.arange(len(norm_names))
    ax4.bar(x - width/2, v1_norms, width, color=colors[0], label="v1")
    ax4.bar(x + width/2, v2_norms, width, color=colors[1], label="v2")
    # 添加数值标签
    for i, (v1, v2) in enumerate(zip(v1_norms, v2_norms)):
        ax4.text(i - width/2, v1 + 0.5, f"{v1:.1f}", ha="center", fontsize=10)
        ax4.text(i + width/2, v2 + 0.5, f"{v2:.1f}", ha="center", fontsize=10)
    ax4.set_xlabel("Matrix Type", fontsize=12)
    ax4.set_ylabel("Frobenius Norm (Numerical Stability)", fontsize=12)
    ax4.set_title("Matrix Norm Comparison (Numerical Scale)", fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(norm_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    # 保存图片（符合文档实验结果保存要求，🔶1-87）
    plt.tight_layout()
    plt.savefig("lunar_lander_abc_comparison_v1_vs_v2.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("\nA/B/C 对比图表已保存为：lunar_lander_abc_comparison_v1_vs_v2.png")


def main(seed: int = 2, data_dir: str = ".", psi_v1: Optional["PsiMLP"] = None, psi_v2: Optional["PsiMLP"] = None):
    """
    主函数：读取 v1/v2 的 A/B/C，计算指标，对比并可视化
    """
    # 1. 读取 v1 和 v2 的 A/B/C 矩阵
    v1_abc = load_abc_matrix(version="v1", seed=seed, data_dir=data_dir)
    v2_abc = load_abc_matrix(version="v2", seed=seed, data_dir=data_dir)

    # 2. 计算关键指标（若有 Psi 网络，可传入计算重构误差）
    x_star = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 目标状态（文档 IV.D 节）
    v1_metrics = compute_abc_metrics(v1_abc, psi=psi_v1, x_star=x_star)
    v2_metrics = compute_abc_metrics(v2_abc, psi=psi_v2, x_star=x_star)

    # 3. 打印量化对比结果（贴合文档评估框架）
    print("\n" + "="*80)
    print("DKRC A/B/C Matrix Quantitative Comparison (v1 vs v2)")
    print("="*80)
    # 按矩阵分类打印
    print("\n【1. A Matrix (Stability)】")
    print(f"{'Metric':<30} {'v1':<12} {'v2':<12} {'Better Version':<10}")
    print("-"*64)
    metrics_a = [
        ("Max Eigen Norm (<1 is stable)", "A_max_eig_norm"),
        ("Mean Eigen Norm (uniformity)", "A_mean_eig_norm"),
        ("Eigen Norm Std (consistency)", "A_eig_norm_std"),
        ("Frobenius Norm (numerical scale)", "A_fro_norm")
    ]
    for name, key in metrics_a:
        v1_val = v1_metrics[key]
        v2_val = v2_metrics[key]
        better = "v1" if v1_val < v2_val else "v2"
        print(f"{name:<30} {v1_val:<12.4f} {v2_val:<12.4f} {better:<10}")

    print("\n【2. B Matrix (Controllability)】")
    print(f"{'Metric':<30} {'v1':<12} {'v2':<12} {'Better Version':<10}")
    print("-"*64)
    metrics_b = [
        ("Controllability Rank (full=N)", "B_ctrl_rank"),
        ("Controllability Rank Ratio (1 is best)", "B_ctrl_rank_ratio"),
        ("Mean Column Norm (control strength)", "B_mean_col_norm"),
        ("Max Column Norm (max control impact)", "B_max_col_norm")
    ]
    for name, key in metrics_b:
        v1_val = v1_metrics[key]
        v2_val = v2_metrics[key]
        better = "v1" if (key == "B_ctrl_rank_ratio" and v1_val > v2_val) else ("v2" if v1_val < v2_val else "v1")
        print(f"{name:<30} {v1_val:<12.4f} {v2_val:<12.4f} {better:<10}")

    if "C_mean_recon_error" in v1_metrics and "C_mean_recon_error" in v2_metrics:
        print("\n【3. C Matrix (Reconstruction)】")
        print(f"{'Metric':<30} {'v1':<12} {'v2':<12} {'Better Version':<10}")
        print("-"*64)
        metrics_c = [
            ("Mean Reconstruction Error (accuracy)", "C_mean_recon_error"),
            ("Recon Error Std (consistency)", "C_recon_error_std"),
            ("Mean Row Norm (weight uniformity)", "C_mean_row_norm"),
            ("Frobenius Norm (numerical scale)", "C_fro_norm")
        ]
        for name, key in metrics_c:
            v1_val = v1_metrics[key]
            v2_val = v2_metrics[key]
            better = "v1" if v1_val < v2_val else "v2"
            print(f"{name:<30} {v1_val:<12.4f} {v2_val:<12.4f} {better:<10}")

    # 4. 可视化对比
    plot_abc_comparison(v1_abc, v2_abc, v1_metrics, v2_metrics)

    # 5. 总结关键结论（基于文档控制逻辑）
    print("\n" + "="*80)
    print("Key Conclusion (Based on DKRC Control Logic)")
    print("="*80)
    # 稳定性结论
    if v1_metrics["A_max_eig_norm"] < 1 and v2_metrics["A_max_eig_norm"] >= 1:
        print("❌ v2 A matrix is UNSTABLE (max eigen norm ≥1) → LQR control may oscillate")
    elif v1_metrics["A_max_eig_norm"] >= 1 and v2_metrics["A_max_eig_norm"] < 1:
        print("✅ v2 A matrix is MORE STABLE (max eigen norm <1) → Better control stability")
    else:
        print(f"⚠️ Both A matrices are {'stable' if v1_metrics['A_max_eig_norm'] <1 else 'unstable'} (v1: {v1_metrics['A_max_eig_norm']:.3f}, v2: {v2_metrics['A_max_eig_norm']:.3f})")
    # 能控性结论
    if v1_metrics["B_ctrl_rank_ratio"] == 1 and v2_metrics["B_ctrl_rank_ratio"] < 1:
        print("❌ v2 B matrix has INSUFFICIENT CONTROLLABILITY → LQR cannot design effective gain")
    elif v1_metrics["B_ctrl_rank_ratio"] < 1 and v2_metrics["B_ctrl_rank_ratio"] == 1:
        print("✅ v2 B matrix is FULLY CONTROLLABLE → Better LQR control performance")
    else:
        print(f"⚠️ Controllability: v1 ratio={v1_metrics['B_ctrl_rank_ratio']:.3f}, v2 ratio={v2_metrics['B_ctrl_rank_ratio']:.3f} (1.0 is full rank)")
    # 重构精度结论（若有数据）
    if "C_mean_recon_error" in v1_metrics:
        if v2_metrics["C_mean_recon_error"] < v1_metrics["C_mean_recon_error"]:
            print("✅ v2 C matrix has BETTER RECONSTRUCTION ACCURACY → More accurate state observation")
        else:
            print("❌ v2 C matrix has WORSE RECONSTRUCTION ACCURACY → Less accurate state observation")


# 调用示例（需根据实际环境调整）
if __name__ == "__main__":
    # 若需要计算 C 矩阵的重构误差，需传入训练好的 PsiMLP 网络（可选）
    # from rdkrc.models.psi_mlp import PsiMLP
    # psi_v1 = PsiMLP(...)  # 加载 v1 的 Psi 网络
    # psi_v2 = PsiMLP(...)  # 加载 v2 的 Psi 网络
    # main(seed=2, data_dir="./abc_files", psi_v1=psi_v1, psi_v2=psi_v2)
    parser = argparse.ArgumentParser(description="Compare DKRC A/B/C Matrices between v1 and v2")
    parser.add_argument("--seed", type=int, default=2, help="Random seed used in training (default: 2)")
    args = parser.parse_args()
    # 若无 Psi 网络，仅对比 A/B/C 的基础指标
    main(seed=args.seed, data_dir=".")  # data_dir 为 A/B/C 文件所在目录