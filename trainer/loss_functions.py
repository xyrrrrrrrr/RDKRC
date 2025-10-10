import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple, Optional
from rdkrc.utils.matrix_utils import compute_controllability_matrix
from rdkrc.utils.data_utils import compute_knn_neighbors

def compute_L1_loss(
    z_prev: torch.Tensor,
    z_next: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    u_prev: torch.Tensor,
    u0: torch.Tensor
) -> torch.Tensor:
    """
    计算L1损失（文档Algorithm 1步骤2 + Equation 7）：
    确保高维线性系统的预测误差最小化，核心公式为：
    L1(θ) = (1/L) · Σ||z_{t+1} - A z_t - B(u_t - u0)||_F 
    （L为批量大小，对应文档“t=0到L-1求和”，L = 批量样本数）
    
    文档依据：
    - Algorithm 1步骤2：L1(θ) = (1/(L-1))·Σ||z(x_{t+1}) - K·z(x_t)||（K为Koopman算子初步近似）
    - Section II Equation 7：L = Σ||z_{t+1} - A z_t - B(u - u0)||_F（最终线性模型误差）
    此处融合两者：用当前迭代的A/B替代K，加入u0项，确保线性模型精度。

    Args:
        z_prev (torch.Tensor): t时刻高维状态z(x_t)，形状[batch_size, N]（N=基函数维度，如128）
        z_next (torch.Tensor): t+1时刻高维状态z(x_{t+1})，形状[batch_size, N]
        A (torch.Tensor): 当前迭代的Koopman矩阵，形状[N, N]（来自`matrix_utils.update_A_B`）
        B (torch.Tensor): 当前迭代的控制矩阵，形状[N, m]（m=控制维度，如倒立摆m=1）
        u_prev (torch.Tensor): t时刻控制输入，形状[batch_size, m]（与`matrix_utils.update_A_B`输入一致）
        u0 (torch.Tensor): 控制固定点（来自PsiMLP.forward_u0），形状[batch_size, m]

    Returns:
        torch.Tensor: 批量平均后的L1损失（标量）
    """
    # 1. 设备一致性确保（避免CPU/GPU混合计算错误）
    device = z_prev.device
    A, B, u_prev, u0 = A.to(device), B.to(device), u_prev.to(device), u0.to(device)
    
    # 2. 计算变换后控制输入v_t = u_t - u0（文档Equation 4）
    v_prev = u_prev - u0  # 形状[batch_size, m]
    
    # 3. 高维线性模型预测z_next（文档Equation 5：z_{t+1} = A z_t + B v_t）
    # 批量矩阵乘法：z_prev [B,N] × A.T [N,N] → [B,N]；v_prev [B,m] × B.T [m,N] → [B,N]
    z_next_pred = torch.matmul(z_prev, A.T) + torch.matmul(v_prev, B.T)  # 形状[batch_size, N]
    
    # 4. 计算每个样本的F范数误差（文档Equation 7的||·||_F）
    # dim=1：对每个样本的N维特征计算F范数（等价于L2范数）
    sample_errors = torch.norm(z_next - z_next_pred, p='fro', dim=1)  # 形状[batch_size]
    
    # 5. 批量平均（文档Algorithm 1步骤2的1/(L-1)，此处L=batch_size，因批量为t=0到L-1的L个样本）
    total_L1 = sample_errors.mean()  # 等价于sum(sample_errors) / batch_size
    
    return total_L1


def compute_L2_loss(
    A: torch.Tensor,
    B: torch.Tensor,
    lambda_rank: float = 0.8,
    lambda_A: float = 0.1,
    lambda_B: float = 0.1
) -> torch.Tensor:
    """
    计算L2损失（文档Algorithm 1步骤3）：
    确保系统能控性与矩阵参数正则化，核心公式为：
    L2(θ) = (N - rank(Cont(A,B))) + ||A||₁ + ||B||₁
    其中Cont(A,B)为能控性矩阵（由`matrix_utils.compute_controllability_matrix`计算）。

    文档依据：
    - Algorithm 1步骤3：L2(θ) = (N - rank(controllability(A,B))) + ||A||₁ + ||B||₁
    - 注：lambda_rank/A/B为超参数，文档未指定权重，默认设为1.0以对齐原文结构，可按需调整。

    Args:
        A (torch.Tensor): Koopman矩阵，形状[N, N]
        B (torch.Tensor): 控制矩阵，形状[N, m]
        lambda_rank (float): 能控性秩惩罚权重（默认1.0，对齐原文）
        lambda_A (float): A矩阵1范数惩罚权重（默认1.0，对齐原文）
        lambda_B (float): B矩阵1范数惩罚权重（默认1.0，对齐原文）

    Returns:
        torch.Tensor: L2损失（标量）
    """
    # 1. 设备一致性确保
    device = A.device
    B = B.to(device)
    
    # 2. 计算能控性矩阵（调用`matrix_utils`的辅助函数，严格对齐文档定义）
    controllability_mat = compute_controllability_matrix(A, B)  # 形状[N, N×m]
    
    # 3. 计算能控性矩阵的秩（奇异值>1e-5视为有效秩，避免数值误差）
    _, singular_vals, _ = torch.svd(controllability_mat)
    rank = (singular_vals > 1e-5).sum().item()  # 有效秩
    N = A.shape[0]
    rank_penalty = lambda_rank * (N - rank) / N  # 归一化秩惩罚，避免随N变化过大
    
    # 4. 计算A/B的1范数（文档Algorithm 1步骤3的正则化项）
    A_l1 = lambda_A * torch.norm(A, p=1)  # A矩阵1范数
    B_l1 = lambda_B * torch.norm(B, p=1)  # B矩阵1范数
    
    # 5. 总L2损失（文档定义的三项之和）
    total_L2 = rank_penalty + A_l1 + B_l1
    
    return total_L2


def compute_total_loss(
    z_prev: torch.Tensor,
    z_next: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    u_prev: torch.Tensor,
    u0: torch.Tensor,
    lambda_L1: float = 1.0,
    lambda_L2: float = 1.0,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算DKRC总损失（文档Algorithm 1步骤4）：
    总损失 = λ_L1·L1 + λ_L2·L2（λ_L1/λ_L2为损失平衡超参数，默认1.0）

    文档依据：
    - Algorithm 1步骤4：L(θ) = L1(θ) + L2(θ)（此处保留λ_L1/λ_L2以支持灵活调参，默认对齐原文）

    Args:
        z_prev (torch.Tensor): t时刻高维状态，形状[batch_size, N]
        z_next (torch.Tensor): t+1时刻高维状态，形状[batch_size, N]
        A (torch.Tensor): Koopman矩阵，形状[N, N]
        B (torch.Tensor): 控制矩阵，形状[N, m]
        u_prev (torch.Tensor): t时刻控制输入，形状[batch_size, m]
        u0 (torch.Tensor): 控制固定点，形状[batch_size, m]
        lambda_L1 (float): L1损失权重（默认1.0，对齐原文）
        lambda_L2 (float): L2损失权重（默认1.0，对齐原文）
        version (str): 损失版本（"v1"或"v2"）
        **kwargs: 传递给`compute_L2_loss`的额外参数（如lambda_rank）

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            total_loss: 总损失（标量）
            L1: L1损失（标量）
            L2: L2损失（标量）
    """
    # 1. 计算L1损失（传入A/B/u_prev/u0，对齐线性模型误差）
    L1 = compute_L1_loss(z_prev, z_next, A, B, u_prev, u0)
    
    # 2. 计算L2损失（调用`compute_L2_loss`，支持额外超参数）
    L2 = compute_L2_loss(A, B, **kwargs)
 
    # 3. 计算总损失（文档Algorithm 1步骤4）
    total_loss = lambda_L1 * L1 + lambda_L2 * L2
        
    return total_loss, L1, L2

# 修改后的ManifoldEmbLoss类示例（需替换你原有代码）
class ManifoldEmbLoss(nn.Module):
    def __init__(self, k=10):
        super().__init__()
        self.k = k  # K近邻数量
        self.neighbor_indices = None  # 不再预存全局索引，改为batch内临时存储
        self.GraphMatchingLoss = GraphMatchingLoss()

    def compute_knn(self, X):
        """针对单个batch的X，计算每个样本的K近邻索引（仅在当前batch内）"""
        # 计算X的 pairwise 距离（欧氏距离）
        n = X.shape[0]
        dist_matrix = torch.cdist(X, X, p=2)  # shape=[n, n]
        # 取每个样本的前k+1个近邻（排除自身，所以k+1），再去掉第0个（自身）
        _, indices = torch.topk(dist_matrix, k=self.k+1, largest=False, dim=1)
        self.neighbor_indices = indices[:, 1:]  # shape=[n, k]，每个样本的k个邻居索引
        return self.neighbor_indices

    def forward(self, z, X):
        """
        z: 当前batch的嵌入张量，shape=[batch*T, manifold_dim]
        X: 当前batch的原状态张量，shape=[batch*T, x_dim]
        """
        # 第一步：针对当前batch的X，动态计算K近邻索引
        self.compute_knn(X)
        # 第二步：根据邻居索引，提取z和X的邻居样本
        n = z.shape[0]
        # 确保索引在合法范围内（双重保险）
        self.neighbor_indices = torch.clamp(self.neighbor_indices, 0, n-1)
        
        # 提取每个样本的邻居（shape=[n, k, dim]）
        z_neighbors = z[self.neighbor_indices]  # [n, k, manifold_dim]
        x_neighbors = X[self.neighbor_indices]  # [n, k, x_dim]
        
        # 计算原状态与邻居的距离、嵌入后与邻居的距离
        x_dist = torch.cdist(X.unsqueeze(1), x_neighbors, p=2).squeeze(1) 
        z_dist = torch.cdist(z.unsqueeze(1), z_neighbors, p=2).squeeze(1) 

        x_dist_max = torch.max(x_dist, dim=1, keepdim=True)[0]
        x_dist_max = torch.clamp(x_dist_max, min=1e-8)  # 防止过小导致梯度爆炸
        x_dist = x_dist / x_dist_max  # 归一化，避免尺度
        z_dist_max = torch.max(z_dist, dim=1, keepdim=True)[0]
        z_dist_max = torch.clamp(z_dist_max, min=1e-8)  # 防止过小导致梯度爆炸
        z_dist = z_dist / z_dist_max  # 归一化，避免尺度

        # # 计算dij, dzij
        # dij = torch.cdist(x_neighbors, x_neighbors, p=2)  # [n, k, k]
        # d_zij = torch.cdist(z_neighbors, z_neighbors, p=2)  # [n, k, k]
        
        # # 计算流形损失（原逻辑不变）
        loss1 = torch.mean(torch.abs(z_dist - x_dist))
        # loss2 = self.GraphMatchingLoss(dij, d_zij)
        # return loss1 + loss2

        return loss1

class GraphMatchingLoss(nn.Module):
    """
    图匹配损失（PyTorch版）：对齐文档🔶2-59节图匹配损失公式
    功能：计算原空间距离矩阵dij与潜空间距离矩阵d_zij的"距离差一致性"损失，
          通过全局最大距离差归一化，避免尺度差异影响训练（文档隐含要求，🔶2-60节）
    """
    def __init__(self):
        super().__init__()
        # 继承ManifoldEmbLoss的简洁初始化风格，无额外超参（核心参数由forward传入）

    def forward(
        self, 
        dij: torch.Tensor, 
        d_zij: torch.Tensor, 
    ) -> torch.Tensor:
        """
        Args:
            dij: 原空间测地线距离矩阵（文档🔶2-33节d_D(xi,xj)），shape=[B, B]（B为样本数）
            d_zij: 潜空间距离矩阵（文档🔶2-33节d_M(φ(xi),φ(xj))），shape=[B, B]
            dij_diff_max: 原空间距离差的全局最大值（文档🔶2-59节归一化因子），标量；
                          若为None，自动计算（适配无预计算场景）
        
        Returns:
            gm_loss: 图匹配损失（标量），符合文档🔶2-59节公式定义
        """

        diff_dij_temp = dij.unsqueeze(1) - dij.unsqueeze(0)  # [B, B, B]
        dij_diff_max = torch.max(torch.abs(diff_dij_temp))  # 标量
        
        # 2. 数值稳定性处理（参考ManifoldEmbLoss的1e-8策略，避免除以零）
        dij_diff_max = torch.clamp(dij_diff_max, min=1e-8)  # 防止dij_diff_max过小导致梯度爆炸
        
        # 3. 计算距离差（对齐JAX原逻辑：dij[:,newaxis]-dij[newaxis]）
        # 文档依据：🔶2-55节图匹配损失需计算"每个样本对(i,j)相对于所有k的距离差"
        diff_dij = dij.unsqueeze(1) - dij.unsqueeze(0)  # [B, B, B]：diff_dij[i,j,k] = dij[i,k] - dij[j,k]
        diff_d_z_ij = d_zij.unsqueeze(1) - d_zij.unsqueeze(0)  # [B, B, B]：潜空间对应距离差
        
        # 4. 计算图匹配损失（文档🔶2-59节公式：归一化平方损失的均值）
        gm_loss = torch.mean(((diff_dij - diff_d_z_ij) / dij_diff_max) ** 2)
        
        return gm_loss

class ManifoldCtrlLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, A: nn.Linear, B: nn.Linear, z_t: torch.Tensor, z_t1: torch.Tensor, g_phi: torch.Tensor, u:torch.Tensor) -> torch.Tensor:
        """
        计算线性演化一致性损失
        A, B: Koopman算子（nn.Linear层，无偏置）
        z_t: t时刻嵌入向量 [batch, n+d]
        z_t1: t+1时刻嵌入向量 [batch, n+d]
        g_phi: 控制网络输出 [batch, m]（m为控制维度）
        """
        # 计算理论控制嵌入：g_phi_theo = B^+ (z_t1 - A z_t)
        A_z_t = A(z_t)  # [batch, n+d]
        z_diff = z_t1 - A_z_t  # [batch, n+d]
        
        # 计算B的伪逆（B.weight: [n+d, m]）
        B_weight = B.weight  # [out_dim= n+d, in_dim= m]
        B_pinv = torch.linalg.pinv(B_weight)  # [m, n+d]
        
        # 理论控制嵌入：[batch, m] = [batch, n+d] @ [n+d, m]
        g_phi_theo = z_diff @ B_pinv.T
        
        # 一致性损失
        loss1 = self.mse_loss(g_phi * u, g_phi_theo)

        return loss1
    
class ManifoldCtrlInvLoss(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.mse_loss = nn.MSELoss()

    def compute_knn(self, X):
        """针对单个batch的X，计算每个样本的K近邻索引（仅在当前batch内）"""
        # 计算X的 pairwise 距离（欧氏距离）
        n = X.shape[0]
        dist_matrix = torch.cdist(X, X, p=2)  # shape=[n, n]
        # 取每个样本的前k+1个近邻（排除自身，所以k+1），再去掉第0个（自身）
        _, indices = torch.topk(dist_matrix, k=self.k+1, largest=False, dim=1)
        self.neighbor_indices = indices[:, 1:]  # shape=[n, k]，每个样本的k个邻居索引
        return self.neighbor_indices

    def forward(self, U_recover: torch.Tensor, U_real: torch.Tensor) -> torch.Tensor:
        """
        计算线性演化一致性损失
        A, B: Koopman算子（nn.Linear层，无偏置）
        z_t: t时刻嵌入向量 [batch, n+d]
        z_t1: t+1时刻嵌入向量 [batch, n+d]
        g_phi: 控制网络输出 [batch, m]（m为控制维度）
        """
        # 差距损失
        loss1 = self.mse_loss(U_recover, U_real)

        # return loss1
         # 第一步：针对当前batch的X，动态计算K近邻索引
        self.compute_knn(U_real)
        # 第二步：根据邻居索引，提取z和X的邻居样本
        n = U_real.shape[0]
        # 确保索引在合法范围内（双重保险）
        self.neighbor_indices = torch.clamp(self.neighbor_indices, 0, n-1)
        
        # 提取每个样本的邻居（shape=[n, k, dim]）
        U_real_neighbors = U_real[self.neighbor_indices]  # [n, k, manifold_dim]
        U_recover_neighbors = U_recover[self.neighbor_indices]  # [n, k, x_dim]
        
        # 计算原状态与邻居的距离、嵌入后与邻居的距离
        U_real_dist = torch.cdist(U_real.unsqueeze(1), U_real_neighbors, p=2).squeeze(1) 
        U_recover_dist = torch.cdist(U_recover.unsqueeze(1), U_recover_neighbors, p=2).squeeze(1) 

        U_real_dist_max = torch.max(U_real_dist, dim=1, keepdim=True)[0] + 1e-8
        U_real_dist = U_real_dist / U_real_dist_max  # 归一化，避免尺度
        U_recover_dist_max = torch.max(U_recover_dist, dim=1, keepdim=True)[0] + 1e-8
        U_recover_dist = U_recover_dist / U_recover_dist_max  # 归一化，避免尺度

        # 计算流形损失（原逻辑不变）
        loss2 = torch.mean(torch.abs(U_real_dist - U_recover_dist))


        return loss1 + loss2
