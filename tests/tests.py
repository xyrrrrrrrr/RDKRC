import sys
import os
# 获取工程根目录路径（即包含dkrc包的目录）
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将根目录加入Python搜索路径
sys.path.append(root_dir)
import torch
from rdkrc import PsiMLP, compute_L1_loss, update_A_B, compute_C_matrix


def test_psi_mlp_forward():
    """测试PsiMLP前向传播维度正确性"""
    # 配置（倒立摆:input_dim=3,基函数维度N=128）
    input_dim = 3
    output_dim = 128
    batch_size = 64
    
    # 初始化模型
    model = PsiMLP(input_dim=input_dim, output_dim=output_dim)
    # 生成测试数据
    x = torch.randn(batch_size, input_dim)  # 随机状态批量
    
    # 前向传播
    psi_x = model.forward(x)
    
    # 验证维度
    assert psi_x.shape == (batch_size, output_dim), \
        f"PsiMLP输出维度错误:预期({batch_size}, {output_dim}),实际{psi_x.shape}"
    print("test_psi_mlp_forward:  passed")


def test_psi_mlp_compute_z():
    """测试z = Ψ(x) - Ψ(x*)的维度正确性"""
    input_dim = 3
    output_dim = 128
    batch_size = 64
    
    model = PsiMLP(input_dim=input_dim, output_dim=output_dim)
    x = torch.randn(batch_size, input_dim)
    x_star = torch.randn(input_dim)  # 目标状态（单一样本）
    
    # 计算z
    z = model.compute_z(x, x_star)
    
    # 验证维度
    assert z.shape == (batch_size, output_dim), \
        f"compute_z输出维度错误:预期({batch_size}, {output_dim}),实际{z.shape}"
    print("test_psi_mlp_compute_z:  passed")


def test_loss_and_matrix_utils():
    """测试损失函数和矩阵工具函数的维度正确性"""
    # 配置
    N = 128  # 基函数维度
    m = 1    # 控制维度（倒立摆:扭矩输入）
    n = 3    # 状态维度
    batch_size = 64
    
    # 生成测试数据
    z_prev = torch.randn(batch_size, N)
    z_next = torch.randn(batch_size, N)
    u_prev = torch.randn(batch_size, m)
    x_prev = torch.randn(batch_size, n)
    
    # 1. 测试L1损失
    L1 = compute_L1_loss(z_prev, z_next)
    assert L1.dim() == 0, f"L1损失应为标量,实际维度{L1.dim()}"
    
    # 2. 测试A/B更新
    A = torch.randn(N, N)
    B = torch.randn(N, m)
    A, B = update_A_B(z_prev, z_next, u_prev, A, B)
    assert A.shape == (N, N), f"A矩阵维度错误:预期({N}, {N}),实际{A.shape}"
    assert B.shape == (N, m), f"B矩阵维度错误:预期({N}, {m}),实际{B.shape}"
    
    # 3. 测试C矩阵求解
    C = compute_C_matrix(x_prev, z_prev)
    assert C.shape == (n, N), f"C矩阵维度错误:预期({n}, {N}),实际{C.shape}"
    
    print("test_loss_and_matrix_utils:  passed")


if __name__ == "__main__":
    # 运行所有测试
    test_psi_mlp_forward()
    test_psi_mlp_compute_z()
    test_loss_and_matrix_utils()
    print("所有单元测试通过！")