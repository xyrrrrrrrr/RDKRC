import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union


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
        # x_norm = self.normalize_x(x)
        # return self.psi_mlp(x_norm)
        return self.psi_mlp(x)

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