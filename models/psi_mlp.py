import torch
import torch.nn as nn
from typing import List, Optional


class PsiMLP(nn.Module):
    """
    基于MLP的Koopman基函数学习网络,用于将非线性状态x映射到高维线性空间的基函数Ψ(x)
    原文对应:Section II "Koopman Learning using Deep Neural Network" 中DNN基函数定义
    
    Args:
        input_dim (int): 原始状态x的维度(如倒立摆n=3:cosθ, sinθ, θ_dot)
        output_dim (int): 高维基函数Ψ(x)的维度(N,需满足N ≫ input_dim)
        hidden_dims (List[int]): 隐藏层维度列表(原文用4层,默认[256, 256, 256, 256])
        activation (nn.Module): 隐藏层激活函数(原文用tanh,默认nn.Tanh())
        device (Optional[torch.device]): 计算设备(默认自动检测GPU/CPU)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        activation: nn.Module = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 默认参数对齐原文(4层隐藏层 + tanh激活)
        self.hidden_dims = hidden_dims or [256, 256, 256, 256]
        self.activation = activation or nn.Tanh()
        
        # 构建MLP网络结构(输入层 → 隐藏层 × K → 输出层)
        layers = []
        in_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim, device=self.device))
            layers.append(self.activation)
            in_dim = hidden_dim
        # 输出层(无激活,直接输出基函数值)
        layers.append(nn.Linear(in_dim, self.output_dim, device=self.device))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播:输入原始状态x,输出高维基函数Ψ(x)
        
        Args:
            x (torch.Tensor): 原始状态批量,形状为[batch_size, input_dim]
        
        Returns:
            torch.Tensor: 高维基函数批量,形状为[batch_size, output_dim]
        """
        # 确保输入张量在指定设备上
        x = x.to(self.device)
        return self.mlp(x)

    def compute_z(self, x: torch.Tensor, x_star: torch.Tensor) -> torch.Tensor:
        """
        计算线性化空间状态z = Ψ(x) - Ψ(x*)(x*为目标位置,对应原文Equation 4)
        
        Args:
            x (torch.Tensor): 原始状态批量,形状[batch_size, input_dim]
            x_star (torch.Tensor): 目标状态(单一样本),形状[1, input_dim]或[input_dim]
        
        Returns:
            torch.Tensor: 线性化空间状态z,形状[batch_size, output_dim]
        """
        # 扩展x_star到批量维度,确保与x同批次大小
        if x_star.dim() == 1:
            x_star = x_star.unsqueeze(0)  # [input_dim] → [1, input_dim]
        x_star = x_star.expand(x.shape[0], -1).to(self.device)  # [batch_size, input_dim]
        
        # 计算z = Ψ(x) - Ψ(x*)
        psi_x = self.forward(x)
        psi_x_star = self.forward(x_star)
        return psi_x - psi_x_star