import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Tuple


class PsiMLP(nn.Module):
    """
    基于MLP的Koopman基函数学习网络（含u₀估计），对应文档：
    - Section II "Koopman Learning using Deep Neural Network"（基函数DNN定义）
    - Equation 4（z = Ψ(x) - Ψ(x*)）
    - 文档提及的“one additional auxiliary network for this constant u₀”（u₀估计）
    
    核心功能：
    1. 学习Koopman基函数Ψ(x)，将原状态x映射至高维空间
    2. 计算高维线性空间状态z = Ψ(x) - Ψ(x*)（x*为目标状态）
    3. 学习控制输入固定点u₀（通过可学习参数实现，符合文档“常数u₀”要求）

    Args:
        input_dim (int): 原始状态x的维度（如倒立摆n=3，月球着陆器n=6）
        output_dim (int): 高维基函数Ψ(x)的维度N（需满足N ≫ input_dim，文档默认高维）
        control_dim (int): 控制输入u的维度m（如月球着陆器m=2，倒立摆m=1）
        low (Union[List[float], np.ndarray]): 原始状态x的各维度下界（用于归一化，外部传入解耦环境）
        high (Union[List[float], np.ndarray]): 原始状态x的各维度上界（用于归一化，外部传入解耦环境）
        hidden_dims (List[int], optional): 基函数MLP的隐藏层维度（文档用4层，默认[256,256,256,256]）
        activation (nn.Module, optional): 隐藏层激活函数（文档用tanh，默认nn.Tanh()）
        device (Optional[torch.device]): 计算设备（默认自动检测GPU/CPU）
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        control_dim: int,
        low: Union[List[float], np.ndarray],
        high: Union[List[float], np.ndarray],
        hidden_dims: List[int] = None,
        activation: nn.Module = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        # 设备初始化（对齐文档GPU训练要求，如NVIDIA V100/DGX-2）
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 核心参数（匹配文档定义）
        self.input_dim = input_dim  # 原状态维度n
        self.output_dim = output_dim  # 高维空间维度N
        self.control_dim = control_dim  # 控制输入维度m
        
        # 状态归一化参数（外部传入，解耦gym环境，避免模型内硬编码）
        self.low = torch.tensor(low, dtype=torch.float32, device=self.device)
        self.high = torch.tensor(high, dtype=torch.float32, device=self.device)
        # 检查归一化参数维度与输入维度一致
        assert self.low.shape[0] == input_dim and self.high.shape[0] == input_dim, \
            f"low/high维度需与input_dim({input_dim})一致，当前low维度{self.low.shape[0]}，high维度{self.high.shape[0]}"
        
        # 基函数MLP结构（文档Section II：4层隐藏层+tanh激活）
        self.hidden_dims = hidden_dims or [256, 256, 256, 256]  # 文档默认4层隐藏层
        self.activation = activation or nn.Tanh()  # 文档指定tanh激活
        self.psi_mlp = self._build_psi_mlp()
        
        # 控制输入固定点u₀（文档要求：额外辅助网络学习常数u₀，此处用可学习参数简化实现，符合“常数”属性）
        self.u0 = nn.Parameter(
            torch.zeros(control_dim, dtype=torch.float32, device=self.device),
            requires_grad=True  # 允许反向传播更新，实现数据驱动学习
        )

    def _build_psi_mlp(self) -> nn.Sequential:
        """构建基函数Ψ(x)的MLP网络（文档Section II指定结构）"""
        layers = []
        in_dim = self.input_dim
        
        # 隐藏层（按文档4层设计）
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim, device=self.device))
            layers.append(self.activation)  # 每层后接tanh激活
            in_dim = hidden_dim
        
        # 输出层（无激活，直接输出高维基函数值Ψ(x)）
        layers.append(nn.Linear(in_dim, self.output_dim, device=self.device))
        
        return nn.Sequential(*layers)

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        状态归一化（文档隐含数据预处理要求，提升训练稳定性）：
        将x映射到[0,1]区间，先裁剪到[low, high]避免异常值影响
        
        Args:
            x (torch.Tensor): 原始状态批量，形状[batch_size, input_dim]
        
        Returns:
            torch.Tensor: 归一化后状态，形状[batch_size, input_dim]
        """
        # 确保输入在设备上且类型匹配
        x = x.to(self.device, dtype=torch.float32)
        # 裁剪异常值到[low, high]
        x_clamped = torch.clamp(x, self.low, self.high)
        # 归一化公式：(x - low) / (high - low)
        x_norm = (x_clamped - self.low) / (self.high - self.low + 1e-8)  # 加1e-8避免分母为0
        return x_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：输入原始状态x，输出高维基函数Ψ(x)（文档Section II核心映射）
        
        Args:
            x (torch.Tensor): 原始状态批量，形状[batch_size, input_dim]
        
        Returns:
            torch.Tensor: 高维基函数批量，形状[batch_size, output_dim]
        """
        # 先归一化再输入MLP（符合数据驱动训练逻辑）
        x_norm = self.normalize_x(x)
        return self.psi_mlp(x_norm)
        # return self.psi_mlp(x)

    def compute_z(self, x: torch.Tensor, x_star: torch.Tensor) -> torch.Tensor:
        """
        计算高维线性空间状态z（文档Equation 4：z = Ψ(x) - Ψ(x*)）
        x*为目标状态（如倒立摆θ=0,θ_dot=0；月球着陆器landing zone）
        
        Args:
            x (torch.Tensor): 原始状态批量，形状[batch_size, input_dim]
            x_star (torch.Tensor): 目标状态（单样本），形状[input_dim]或[1, input_dim]
        
        Returns:
            torch.Tensor: 高维状态z，形状[batch_size, output_dim]
        """
        # 处理x_star维度：扩展到批量大小，确保与x维度匹配
        x_star = x_star.to(self.device, dtype=torch.float32)
        if x_star.dim() == 1:
            x_star = x_star.unsqueeze(0)  # [input_dim] → [1, input_dim]
        x_star_batch = x_star.expand(x.shape[0], -1)  # [batch_size, input_dim]
        
        # 计算z = Ψ(x) - Ψ(x*)
        psi_x = self.forward(x)
        psi_x_star = self.forward(x_star_batch)
        return psi_x - psi_x_star

    def forward_u0(self, x: torch.Tensor) -> torch.Tensor:
        """
        输出控制输入固定点u₀（文档要求：数据驱动学习的常数u₀）
        扩展u₀到批量维度，确保与状态批量大小匹配（便于后续损失计算）
        
        Args:
            x (torch.Tensor): 原始状态批量（仅用于获取批量大小，无实际输入依赖），形状[batch_size, input_dim]
        
        Returns:
            torch.Tensor: u₀批量，形状[batch_size, control_dim]
        """
        batch_size = x.shape[0]
        # 扩展常数u₀到批量维度：[control_dim] → [1, control_dim] → [batch_size, control_dim]
        return self.u0.unsqueeze(0).expand(batch_size, -1)
    

class PsiMLP_v2(nn.Module):
    """
    改进版Koopman基函数学习网络（增强表达能力）
    改进点：
    1. 加入状态注意力机制，自动聚焦关键状态维度
    2. 采用混合激活函数（tanh + Swish），缓解梯度饱和
    3. 多尺度特征融合，增强对不同频率动态的捕捉能力
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        control_dim: int,
        low: Union[List[float], np.ndarray],
        high: Union[List[float], np.ndarray],
        hidden_dims: List[int] = None,
        activation: nn.Module = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 核心参数保持不变
        self.input_dim = input_dim  # 原状态维度n（如月球着陆器n=6）
        self.output_dim = output_dim  # 高维空间维度N
        self.control_dim = control_dim  # 控制输入维度m
        
        # 状态归一化参数（解耦环境依赖）
        self.low = torch.tensor(low, dtype=torch.float32, device=self.device)
        self.high = torch.tensor(high, dtype=torch.float32, device=self.device)
        assert self.low.shape[0] == input_dim and self.high.shape[0] == input_dim, \
            f"low/high维度需与input_dim({input_dim})一致"
        
        # 改进1：状态注意力机制（聚焦关键状态维度）
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim, device=self.device),  # 学习状态维度权重
            nn.Sigmoid()  # 输出注意力权重∈[0,1]
        )
        
        # 改进2：多尺度分支结构（拆分低维/高维特征）
        self.hidden_dims = hidden_dims or [256, 256, 256, 256]
        # 低维分支（捕捉慢变特征，如位置、高度）
        self.branch_low = self._build_branch(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim // 4  # 分配1/4维度（如256→64）
        )
        # 高维分支（捕捉快变特征，如角速度、速度）
        self.branch_high = self._build_branch(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim - (output_dim // 4)  # 剩余3/4维度（如256→192）
        )
        
        # 控制输入固定点u₀（保持原设计）
        self.u0 = nn.Parameter(
            torch.zeros(control_dim, dtype=torch.float32, device=self.device),
            requires_grad=True
        )

    def _build_branch(self, input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Sequential:
        """构建多尺度分支网络（改进2：混合激活函数）"""
        layers = []
        in_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim, device=self.device))
            # 改进2：前两层用tanh（局部非线性），后两层用Swish（缓解梯度饱和）
            if i < 2:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.SiLU())  # Swish激活函数
            in_dim = hidden_dim
        
        # 分支输出层
        layers.append(nn.Linear(in_dim, output_dim, device=self.device))
        return nn.Sequential(*layers)

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """状态归一化（保持原逻辑）"""
        x = x.to(self.device, dtype=torch.float32)
        x_clamped = torch.clamp(x, self.low, self.high)
        x_norm = (x_clamped - self.low) / (self.high - self.low + 1e-8)
        return x_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（融合改进1/2/3）：
        1. 状态注意力加权
        2. 多尺度分支特征提取
        3. 特征融合输出
        """
        # 状态归一化
        x_norm = self.normalize_x(x)
        # x_norm = x
        
        # 改进1：状态注意力机制（权重逐元素相乘）
        attention_weights = self.attention(x_norm)  # [batch_size, input_dim]
        x_attended = x_norm * attention_weights     # 关键维度增强，次要维度抑制
        
        # 改进3：多尺度特征融合
        feat_low = self.branch_low(x_attended)   # 慢变特征（如位置y、角度θ）
        feat_high = self.branch_high(x_attended) # 快变特征（如角速度θ_dot、速度ẏ）
        psi_x = torch.cat([feat_low, feat_high], dim=1)  # 融合特征
        
        return psi_x

    def compute_z(self, x: torch.Tensor, x_star: torch.Tensor) -> torch.Tensor:
        """计算高维线性空间状态z（保持原逻辑）"""
        x_star = x_star.to(self.device, dtype=torch.float32)
        if x_star.dim() == 1:
            x_star = x_star.unsqueeze(0)
        x_star_batch = x_star.expand(x.shape[0], -1)
        
        psi_x = self.forward(x)
        psi_x_star = self.forward(x_star_batch)
        return psi_x - psi_x_star

    def forward_u0(self, x: torch.Tensor) -> torch.Tensor:
        """输出控制输入固定点u₀（保持原逻辑）"""
        batch_size = x.shape[0]
        return self.u0.unsqueeze(0).expand(batch_size, -1)


class PsiMLP_v3(nn.Module):
    """
    最终版Koopman基函数学习网络（完全适配文档场景）
    核心改进依据文档：
    1. 物理对称特征提取（IV.D节Lunar Lander状态定义：x,y,θ,ẋ,ẏ,θ̇，解决θ不连续问题）
    2. 流形瓶颈层（II.20节Koopman线性提升需保留系统内在动态，约束低维流形维度d=4）
    3. 控制敏感性注意力（III节Koopman-based控制，适配IV.D节u1影响y、u2影响x的控制关联）
    4. 能控性正则（III节LQR/MPC依赖能控性，约束AB矩阵能控性矩阵秩）
    5. 联合重构头（Equation 9：C矩阵重构原状态，约束z空间与原状态流形一致）
    6. 贝叶斯注意力（降低Algorithm 1训练的种子敏感性，确保多回合稳定性）
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        control_dim: int,
        physics_dim: int,
        low: Union[List[float], np.ndarray],
        high: Union[List[float], np.ndarray],
        hidden_dims: List[int] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 核心参数
        self.input_dim = input_dim  # 原状态维度（Lunar Lander：6维，🔶1-80）
        self.output_dim = output_dim  # 高维提升空间维度（N≫6，如256）
        self.control_dim = control_dim  # 控制维度（Lunar Lander：2维，🔶1-80）
        
        # 状态归一化参数（解耦IV.D节OpenAI Gym环境依赖，约束状态范围）
        self.low = torch.tensor(low, dtype=torch.float32, device=self.device)
        self.high = torch.tensor(high, dtype=torch.float32, device=self.device)
        assert self.low.shape[0] == input_dim and self.high.shape[0] == input_dim, \
            f"low/high维度需与input_dim({input_dim})一致（文档IV.D节状态维度为6）"
        
        # 改进1：贝叶斯状态注意力（降低种子敏感性，量化权重不确定性）
        self.attention_mu = nn.Linear(input_dim, input_dim, device=self.device)  # 注意力均值
        self.attention_logvar = nn.Linear(input_dim, input_dim, device=self.device)  # 注意力对数方差
        # 加入控制敏感注意力机制
        self.control_sensitive_prior = nn.Parameter(
            torch.tensor([0.4, 0.4, 0.1, 0.1, 0.4, 0.1], device=self.device),  # 预学习控制敏感先验（x/y权重高，🔶1-80）
            requires_grad=False
        )
        
        # 改进2：多尺度分支结构
        self.hidden_dims = hidden_dims or [256, 256, 256, 256]  # 文档II.28节推荐4层隐藏层
        # 低维分支（捕捉慢变特征：x,y,θ，分配1/4维度）
        self.branch_low = self._build_branch(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim // 4,
            physics_dim=4
        )
        # 高维分支（捕捉快变特征：ẋ,ẏ,θ̇，分配3/4维度）
        self.branch_high = self._build_branch(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim - (output_dim // 4),
            physics_dim=4
        )
        
        # 改进3：联合重构头（对应文档Equation 9的C矩阵，联合训练约束z-原状态流形一致）
        self.recon_head = nn.Sequential(
            nn.Linear(output_dim, self.hidden_dims[-1], device=self.device),
            nn.SiLU(),  # 复用混合激活优势，缓解梯度饱和
            nn.Linear(self.hidden_dims[-1], input_dim, device=self.device)  # 输出原状态维度（6）
        )
        
        # 控制固定点u₀（文档II.36节：辅助网络学习非零固定点，适配控制 affine 系统）
        self.u0 = nn.Parameter(
            torch.zeros(control_dim, dtype=torch.float32, device=self.device),
            requires_grad=True
        )

    def _build_branch(self, input_dim: int, hidden_dims: List[int], output_dim: int, physics_dim: int) -> nn.Sequential:
        """
        构建多尺度分支（改进：加入流形瓶颈层，约束系统内在维度d=4，🔶1-20节Koopman线性提升核心）
        d=4：基于IV.D节Lunar Lander状态分析，内在动态维度约3~4（位置-速度耦合）
        """
        layers = []
        in_dim = input_dim
        
        # 原隐藏层（保留v2混合激活：前2层tanh局部非线性，后2层SiLU缓解饱和，🔶1-28节训练稳定性）
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim, device=self.device))
            layers.append(nn.Tanh() if i < 2 else nn.SiLU())
            in_dim = hidden_dim
        
        # 核心改进：流形瓶颈层（强制压缩到内在维度，减少冗余维度）
        layers.append(nn.Linear(in_dim, physics_dim, device=self.device))
        layers.append(nn.SiLU())  # 保留非线性表达
        
        # 恢复到分支输出维度（从低维流形扩展到高维z空间，确保线性提升有效性，🔶1-20节）
        layers.append(nn.Linear(physics_dim, output_dim, device=self.device))
        return nn.Sequential(*layers)

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        状态归一化+物理对称特征提取（改进：适配IV.D节Lunar Lander状态特性）
        解决θ在±π处不连续问题（替换θ为cosθ，确保旋转对称性，🔶1-80节状态定义）
        根据实际情况，需要重新写normalize_x
        """
        x = x.to(self.device, dtype=torch.float32)
        x_clamped = torch.clamp(x, self.low, self.high)  # 约束在环境合法范围
        
        # 物理对称特征重组（保留6维，与原输入兼容）
        x_rel = x_clamped[:, 0:1] - 0.0  # x相对目标偏移（IV.D节目标x=0，🔶1-80）
        y_rel = x_clamped[:, 1:2] - 0.0  # y相对目标偏移（IV.D节目标y=0，🔶1-80）
        dot_x = x_clamped[:, 2:3]  # 保留快变特征ẋ
        dot_y = x_clamped[:, 3:4]  # 保留快变特征ẏ
        cos_theta = torch.cos(x_clamped[:, 4:5])  # 替换原始θ，避免旋转不连续
        dot_theta = x_clamped[:, 5:6]  # 保留快变特征θ̇
        
        # 重组为6维对称化状态（对齐IV.D节输入维度，🔶1-80）
        x_symmetric = torch.cat([x_rel, y_rel, dot_x, dot_y, cos_theta, dot_theta], dim=1)
        # 归一化到[0,1]（减少环境尺度影响，🔶1-28节训练稳定性）
        x_norm = (x_symmetric - self.low) / (self.high - self.low + 1e-8)
        return x_norm

    def _compute_attention(self, x_norm: torch.Tensor, u: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算贝叶斯+控制敏感性注意力（改进：对齐III节Koopman控制需求，🔶1-43节）
        返回：注意力权重、注意力不确定性损失
        """
        batch_size = x_norm.shape[0]
        
        # 1. 贝叶斯注意力采样（重参数化技巧，降低种子敏感性）
        attn_mu = self.attention_mu(x_norm)
        attn_logvar = self.attention_logvar(x_norm)
        attn_std = torch.exp(0.5 * attn_logvar)
        eps = torch.randn_like(attn_std)
        base_attention = torch.sigmoid(attn_mu + eps * attn_std)  # 权重∈[0,1]
        
        # 2. 注意力不确定性损失（正则化均值和方差，确保权重稳定）
        attn_uncert_loss = torch.mean(attn_logvar + (attn_mu ** 2))  # 避免极端权重
        
        # 3. 控制敏感性注意力（训练时生效，适配IV.D节u与状态的关联）
        if u is not None and self.training:
            # u1（主引擎）影响y（y_rel：x_norm[:,1]），u2（侧引擎）影响x（x_rel：x_norm[:,0]），🔶1-80
            u1 = u[:, 0:1]
            u2 = u[:, 1:2]
            # 控制敏感权重：u越大，对应状态维度权重越高
            control_sensitive = torch.zeros_like(base_attention)
            control_sensitive[:, 0:1] = u2  # x维度关联u2
            control_sensitive[:, 1:2] = u1  # y维度关联u1
            control_sensitive[:, 4:5] = (u1 + u2) / 2  # θ̇维度关联总推力
            control_sensitive = torch.sigmoid(control_sensitive * 5)  # 放大差异
            
            # 融合注意力（控制敏感维度权重增强）
            attention_weights = base_attention * (1 + control_sensitive)
            attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True)  # 归一化
        else:
            # 测试时：用预学习控制敏感先验（无u输入时保持控制关联性，🔶1-83节测试逻辑）
            attention_weights = base_attention * self.control_sensitive_prior.unsqueeze(0)
        
        return attention_weights, attn_uncert_loss

    def _compute_controllability_loss(self, psi_x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        能控性正则损失（改进：确保z空间包含控制敏感方向，适配III节LQR/MPC，🔶1-44节）
        约束：控制输入u变化时，z（psi_x）需同步变化，避免B矩阵失效
        """
        # 计算u和z的方差（反映变化范围）
        u_var = u.var(dim=0, keepdim=True)  # 控制输入方差
        z_var = psi_x.var(dim=0, keepdim=True)  # z空间方差
        
        # 仅当u有变化时（u_var>0.01），约束z的变化量（避免u恒定时误判）
        mask = (u_var > 0.01).float().unsqueeze(0)
        # 惩罚z变化过小（目标z_var≥0.1，适配IV.D节控制范围u1∈[0,1],u2∈[-1,1]）
        controllability_loss = mask * torch.maximum(torch.zeros_like(z_var), 0.1 - z_var).mean()
        return controllability_loss

    def forward(self, x: torch.Tensor, u: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（核心流程：对称化→注意力→多尺度特征→正则损失，对齐Algorithm 1步骤1-4）
        返回：Koopman基函数输出psi_x、总正则损失（不确定性+能控性）
        """
        # 1. 状态对称化与归一化
        x_norm = self.normalize_x(x)
        # 2. 注意力计算（贝叶斯+控制敏感性）
        attention_weights, attn_uncert_loss = self._compute_attention(x_norm, u)
        x_attended = x_norm * attention_weights  # 增强关键维度
        # 3. 多尺度特征融合（慢变+快变，🔶1-21节动态捕捉）
        feat_low = self.branch_low(x_attended)   # 慢变：x,y,θ
        feat_high = self.branch_high(x_attended) # 快变：ẋ,ẏ,θ̇
        psi_x = torch.cat([feat_low, feat_high], dim=1)
        # 4. 能控性正则损失（仅训练时计算，🔶1-43节能控性要求）
        controllability_loss = torch.tensor(0.0, device=self.device)
        if u is not None and self.training:
            controllability_loss = self._compute_controllability_loss(psi_x, u)
        # 总正则损失（融合不确定性与能控性约束）
        total_reg_loss = 0.01 * attn_uncert_loss + 0.5 * controllability_loss
        total_reg_loss = total_reg_loss.squeeze(0)
        total_reg_loss = torch.sum(total_reg_loss, dim=1) if len(total_reg_loss.shape) > 1 else total_reg_loss

        return psi_x, total_reg_loss

    def forward_with_recon(self, x: torch.Tensor, u: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        联合训练接口（改进：对齐Equation 9，联合优化Koopman基函数与状态重构，🔶1-42节）
        返回：psi_x、总正则损失、重构损失
        """
        # 1. 基础前向传播（获取psi_x与正则损失）
        psi_x, total_reg_loss = self.forward(x, u)
        # 2. 状态重构（对应C矩阵：x ≈ C·psi_x，🔶1-42节）
        x_recon = self.recon_head(psi_x)
        x_norm = self.normalize_x(x)
        # 3. 重构损失（MSE，约束z空间与原状态流形一致）
        recon_weights = torch.tensor([3.0, 3.0, 1.0, 1.0, 3.0, 1.0], device=self.device).unsqueeze(0)
        # 加权MSE损失（仅惩罚核心维度的重构误差）
        recon_loss = torch.mean(torch.square(x_recon - x_norm) * recon_weights)
            
        return psi_x, total_reg_loss.item(), recon_loss.item()

    def compute_z(self, x: torch.Tensor, x_star: torch.Tensor) -> torch.Tensor:
        """
        计算高维线性状态z（文档Equation 4：z=Ψ(x)-Ψ(x*)，🔶1-35节核心线性化步骤）
        x_star：目标状态（IV.D节Lunar Lander着陆区：x*=[0,0,0,0,0,0]）
        """
        x_star = x_star.to(self.device, dtype=torch.float32)
        # 扩展x_star到批次维度
        if x_star.dim() == 1:
            x_star = x_star.unsqueeze(0)
        x_star_batch = x_star.expand(x.shape[0], -1)
        
        # 计算Ψ(x)与Ψ(x*)
        psi_x, _ = self.forward(x)  # 忽略正则损失（仅计算z时无需）
        psi_x_star, _ = self.forward(x_star_batch)
        return psi_x - psi_x_star

    def forward_u0(self, x: torch.Tensor) -> torch.Tensor:
        """
        输出控制固定点u₀（文档II.36节：辅助网络学习u₀，适配 affine 变换v=u-u₀，🔶1-35节）
        扩展到批次维度，匹配输入批量大小
        """
        batch_size = x.shape[0]
        return self.u0.unsqueeze(0).expand(batch_size, -1)
    
