# Copyright (c) OpenMMLab. All rights reserved.
import math
import types
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
# 1. 从官方 neck 中导入 CSPNeXtPAFPN
from mmdet.models.necks import CSPNeXtPAFPN 
# 2. 导入 CSPLayer
from mmdet.models.layers import CSPLayer
from typing import Tuple

def _import_cspnext_pafpn_rcssc():
    from mmdet.models.necks.cspnext_pafpn_rcssc import CSPNeXtPAFPN_RCSSC
    return CSPNeXtPAFPN_RCSSC


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avgPoolW = nn.AdaptiveAvgPool2d((1, None))
        self.maxPoolW = nn.AdaptiveMaxPool2d((1, None))
        self.conv_1x1 = nn.Conv2d(in_channels=2 * channel, out_channels=2 * channel, kernel_size=1, padding=0, stride=1,
                                  bias=False)
        self.bn = nn.BatchNorm2d(2 * channel, eps=1e-5, momentum=0.01, affine=True)
        self.Relu = nn.LeakyReLU()
        self.F_h = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.F_w = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, H, W = x.size()
        res = x
        x_cat = torch.cat([self.avgPoolW(x), self.maxPoolW(x)], 1)
        x = self.Relu(self.bn(self.conv_1x1(x_cat)))
        x_1, x_2 = x.split(C, 1)

        x_1 = self.F_h(x_1)
        x_2 = self.F_w(x_2)
        s_h = self.sigmoid(x_1)
        s_w = self.sigmoid(x_2)

        out = res * s_h.expand_as(res) * s_w.expand_as(res)
        return out


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = Basic(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bn=False, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class RCSSC(nn.Module):
    def __init__(self, n_feat, reduction=16):
        super(RCSSC, self).__init__()
        pooling_r = 4
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.LeakyReLU(),
        )

        self.SC = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(n_feat)
        )
        self.SA = spatial_attn_layer()
        self.CA = CALayer(n_feat, reduction)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat, kernel_size=1),
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True)
        )
        self.ReLU = nn.LeakyReLU()
        self.tail = nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        x = self.head(x)

        sa_branch = self.SA(x)
        ca_branch = self.CA(x)
        x1 = torch.cat([sa_branch, ca_branch], dim=1)
        x1 = self.conv1x1(x1)

        A = self.SC(x)
        x2 = torch.sigmoid(torch.add(x, F.interpolate(A, x.size()[2:])))

        out = torch.mul(x1, x2)
        out = self.tail(out)
        out = out + res
        out = self.ReLU(out)
        return out


# 添加HPA模块实现
class HPA(nn.Module):
    def __init__(self, channels, factor=32):
        """混合池化注意力模块
        Args:
            channels: 输入通道数
            factor: 分组数，默认32组
        """
        super(HPA, self).__init__()
        # 基础参数校验
        self.groups = factor  # 将通道分成多少组处理（类似分组卷积）
        assert channels // self.groups > 0  
        # ----------------- 注意力机制组件 -----------------
        # 双池化分支：同时利用平均池化和最大池化捕捉不同特征
        self.agp = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化（输出1x1）
        self.map = nn.AdaptiveMaxPool2d((1, 1))  # 全局最大池化（输出1x1）

        # 空间维度池化（分别提取高度/宽度方向特征）
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 高度方向平均池化（保持宽度维度）
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 宽度方向平均池化（保持高度维度）
        self.max_h = nn.AdaptiveMaxPool2d((None, 1))  # 高度方向最大池化
        self.max_w = nn.AdaptiveMaxPool2d((1, None))  # 宽度方向最大池化

        # ----------------- 特征变换层 -----------------
        # 处理特殊情况：如果特征图尺寸为1x1，使用LayerNorm代替GroupNorm
        self.use_group_norm = True
        self.gn = nn.GroupNorm(
            num_groups=channels // self.groups,  
            num_channels=channels // self.groups  # 每组输入通道数
        )
        self.ln = nn.LayerNorm([channels // self.groups, 1, 1])
        
        self.conv1x1 = nn.Conv2d(
            in_channels=channels // self.groups,
            out_channels=channels // self.groups,
            kernel_size=1, 
            stride=1, padding=0
        )
        self.conv3x3 = nn.Conv2d(
            in_channels=channels // self.groups,
            out_channels=channels // self.groups,
            kernel_size=3,  
            padding=1 
        )
        self.softmax = nn.Softmax(dim=-1)  # 用于注意力权重归一化

    def forward(self, x):
        # 输入x形状: [batch_size, channels, height, width]
        b, c, h, w = x.size()

        # 检查是否是1x1特征，决定使用哪种归一化方法
        self.use_group_norm = not (h == 1 and w == 1)

        # ============= 特征分组处理 =============
        # 将通道维度拆分为groups组：[b,c,h,w] -> [b*groups, c/groups, h,w]
        group_x = x.reshape(b * self.groups, -1, h, w)  # -1自动计算为c/groups

        # ============= 平均池化分支 =============
        # 沿高度和宽度方向分别池化
        x_h = self.pool_h(group_x)  # 形状: [b*g, c/g, h, 1]
        x_w = self.pool_w(group_x)  # 形状: [b*g, c/g, 1, w]
        x_w = x_w.permute(0, 1, 3, 2)  # 维度置换：[b*g, c/g, w, 1]
        # 拼接后通过1x1卷积融合空间信息
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # dim=2在高度维度拼接
        # 拆分回原始维度（利用切片操作）
        x_h, x_w = torch.split(hw, [h, w], dim=2)  # x_h形状恢复为[b*g, c/g, h,1]
        
        # 根据特征图尺寸选择合适的归一化方法
        if self.use_group_norm:
            x1 = self.gn(  # 分组归一化增强训练稳定性
                group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
            )
        else:
            # 对于1x1特征，使用LayerNorm
            temp = group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
            x1 = self.ln(temp)

        # ============= 最大池化分支 =============
        # 处理逻辑与平均池化分支类似，但使用最大池化
        y_h = self.max_h(group_x)  # 形状: [b*g, c/g, h, 1]
        y_w = self.max_w(group_x).permute(0, 1, 3, 2)  # 维度置换
        yhw = self.conv1x1(torch.cat([y_h, y_w], dim=2))
        y_h, y_w = torch.split(yhw, [h, w], dim=2)
        

        if self.use_group_norm:
            y1 = self.gn(
                group_x * y_h.sigmoid() * y_w.permute(0, 1, 3, 2).sigmoid()
            )
        else:
            # 对于1x1特征，使用LayerNorm
            temp = group_x * y_h.sigmoid() * y_w.permute(0, 1, 3, 2).sigmoid()
            y1 = self.ln(temp)

        # ============= 注意力权重融合 =============
        # 处理平均池化分支
        x11 = x1.reshape(b * self.groups, -1, h * w)  # 展平空间维度：[b*g, c/g, h*w]
        x12 = self.agp(x1)  # 全局平均池化：[b*g, c/g, 1, 1]
        x12 = x12.reshape(b * self.groups, -1, 1)  # [b*g, c/g, 1]
        x12 = x12.permute(0, 2, 1)  # 调整为矩阵乘法维度：[b*g, 1, c/g]
        x12 = self.softmax(x12)  # 归一化得到注意力权重

        # 处理最大池化分支
        y11 = y1.reshape(b * self.groups, -1, h * w)  # [b*g, c/g, h*w]
        y12 = self.map(y1)  # 全局最大池化
        y12 = y12.reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        y12 = self.softmax(y12)

        # 双分支权重融合（矩阵乘法实现跨通道交互）
        weights = (
                torch.matmul(x12, y11) +  # [b*g, 1, h*w]
                torch.matmul(y12, x11)  # [b*g, 1, h*w]
        ).reshape(b * self.groups, 1, h, w)  # 恢复空间维度

        # ============= 最终输出 =============

        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


@MODELS.register_module()
class CSPNeXtPAFPN_RCSSC_HPA(_import_cspnext_pafpn_rcssc()):
    """CSPNeXtPAFPN with RCSSC and HPA modules.
    
    This neck first enhances the features with RCSSC modules (inherited from
    parent class CSPNeXtPAFPN_RCSSC), then applies HPA attention modules at 
    the end of the forward path to further refine the features.
    
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        rcssc_locations (list): Inherited from parent class.
        use_hpa (bool): Whether to use HPA modules. Default: True
        hpa_factor (int): Group factor for HPA module. Default: 32.
    """
    def __init__(
        self,
        *args,
        use_hpa: bool = True,
        hpa_factor: int = 32,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        
        # 添加HPA模块
        self.use_hpa = use_hpa
        if self.use_hpa:
            self.hpa_modules = nn.ModuleList()
            # 为每个输出特征创建一个HPA模块
            for _ in range(len(self.in_channels)):
                self.hpa_modules.append(
                    HPA(channels=self.out_channels, factor=hpa_factor)
                )

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor, ...]:
        rcssc_outs = super().forward(inputs)
        
        if not self.use_hpa:
            return rcssc_outs
        
        # 应用HPA模块进行最终精炼
        final_outs = []
        for idx, feat in enumerate(rcssc_outs):
            enhanced_feat = self.hpa_modules[idx](feat)
            final_outs.append(enhanced_feat)
            
        return tuple(final_outs) 