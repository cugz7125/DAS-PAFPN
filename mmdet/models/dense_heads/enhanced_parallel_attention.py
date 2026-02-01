import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet.registry import MODELS

class Enhanced_Parallel_Attention(BaseModule):
    """Enhanced Parallel Attention 模块，来自MixDehazeNet论文
    
    论文地址：https://arxiv.org/pdf/2305.17654
    
    Args:
        dim (int): 输入特征的通道数
    """
    def __init__(self, dim):
        super().__init__()
        
        # 使用GroupNorm替代BatchNorm2d，可以更好地处理1x1特征
        self.norm = nn.GroupNorm(8, dim)  # 使用8个组，适用于不同尺寸的特征
        
        # 简单像素注意力
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),  # 1x1卷积
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim)  # 深度可分离卷积
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化到1x1
            nn.Conv2d(dim, dim, 1),  # 1x1卷积
            nn.Sigmoid()  # Sigmoid激活函数
        )

        # 通道注意力
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化到1x1
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),  # 1x1卷积
            nn.GELU(),  # GELU激活函数
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),  # 1x1卷积
            nn.Sigmoid()  # Sigmoid激活函数
        )

        # 像素注意力
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),  # 1x1卷积，降维
            nn.GELU(),  # GELU激活函数
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),  # 1x1卷积，输出单通道
            nn.Sigmoid()  # Sigmoid激活函数
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),  # 1x1卷积，升维
            nn.GELU(),  # GELU激活函数
            nn.Conv2d(dim * 4, dim, 1)  # 1x1卷积，降维
        )

    def init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """前向传播
        
        Args:
            x (Tensor): 输入特征图

        Returns:
            Tensor: 增强后的特征图
        """
        identity = x  # 保存输入以便残差连接
        
        # 应用归一化
        x_norm = self.norm(x)
        
        # 简单像素注意力
        simple_attn = self.Wv(x_norm) * self.Wg(x_norm)
        
        # 通道注意力
        channel_attn = self.ca(x_norm) * x_norm
        
        # 像素注意力
        pixel_attn = self.pa(x_norm) * x_norm
        
        # 拼接并通过MLP
        concat = torch.cat([simple_attn, channel_attn, pixel_attn], dim=1)
        out = self.mlp(concat)
        
        # 残差连接
        return identity + out 