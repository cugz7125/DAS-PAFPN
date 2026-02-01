# Copyright (c) OpenMMLab. All rights reserved.
import math
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


class CSPNeXtBlock(BaseModule):
    """CSPNeXt Block with 3 convolutions.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (float): Expand ratio of the hidden channels. Defaults to 0.5.
        kernel_size (int): The kernel size of the second convolution layer.
            Defaults to 5.
        num_blocks (int): Number of blocks. Defaults to 1.
        add_identity (bool): Whether to add identity to the out.
            Defaults to True.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU').
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        kernel_size: int = 5,
        num_blocks: int = 1,
        add_identity: bool = True,
        use_depthwise: bool = False,
        conv_cfg: OptMultiConfig = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU'),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


# ... (RCSSC 等模块定义不变) ...

# @MODELS.register_module()
# class CSPNeXtPAFPN_RCSSC(BaseModule):
#     def __init__(
#         self,
#         in_channels: Sequence[int],
#         out_channels: int,
#         num_csp_blocks: int = 3,
#         use_depthwise: bool = False,
#         expand_ratio: float = 0.5,
#         upsample_cfg: ConfigType = dict(scale_factor=2, mode='nearest'),
#         conv_cfg: OptMultiConfig = None,
#         norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
#         act_cfg: ConfigType = dict(type='SiLU'),
#         # 新增参数来控制替换位置，更灵活
#         rcssc_locations: list = ['top_down'], # 可选: 'top_down', 'bottom_up'
#         init_cfg: OptMultiConfig = None
#     ) -> None:
#         super().__init__(init_cfg)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

#         # --- build top-down blocks ---
#         self.upsample = nn.Upsample(**upsample_cfg)
#         self.reduce_layers = nn.ModuleList()
#         self.top_down_blocks = nn.ModuleList()
#         for idx in range(len(in_channels) - 1, 0, -1):
#             self.reduce_layers.append(
#                 ConvModule(
#                     in_channels[idx], in_channels[idx - 1], 1,
#                     conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
            
#             # --- 【关键修复】 ---
#             # 决定当前 block 是否使用 RCSSC
#             if 'top_down' in rcssc_locations:
#                 # RCSSC 的输入通道数应该是拼接后的通道数
#                 rcssc_in_channels = in_channels[idx - 1] * 2
#                 block = RCSSC(n_feat=rcssc_in_channels)
#             else:
#                 # 原始的 CSPLayer
#                 block = CSPLayer(
#                     in_channels[idx - 1] * 2, in_channels[idx - 1],
#                     num_blocks=num_csp_blocks, add_identity=False,
#                     use_depthwise=use_depthwise, use_cspnext_block=True,
#                     expand_ratio=expand_ratio, conv_cfg=conv_cfg,
#                     norm_cfg=norm_cfg, act_cfg=act_cfg)
#             self.top_down_blocks.append(block)

#         # --- build bottom-up blocks (保持你之前正确的实现) ---
#         self.downsamples = nn.ModuleList()
#         self.bottom_up_blocks = nn.ModuleList()
#         for idx in range(len(in_channels) - 1):
#             self.downsamples.append(
#                 conv(
#                     in_channels[idx], in_channels[idx], 3, stride=2, padding=1,
#                     conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
            
#             if 'bottom_up' in rcssc_locations:
#                 # RCSSC 的输入通道数也应该是拼接后的
#                 rcssc_in_channels = in_channels[idx] * 2
#                 block = RCSSC(n_feat=rcssc_in_channels)
#             else:
#                 block = CSPLayer(
#                     in_channels[idx] * 2, in_channels[idx + 1],
#                     num_blocks=num_csp_blocks, add_identity=False,
#                     use_depthwise=use_depthwise, use_cspnext_block=True,
#                     expand_ratio=expand_ratio, conv_cfg=conv_cfg,
#                     norm_cfg=norm_cfg, act_cfg=act_cfg)
#             self.bottom_up_blocks.append(block)

#         # --- build out convs (不变) ---
#         self.out_convs = nn.ModuleList()
#         for i in range(len(in_channels)):
#             self.out_convs.append(
#                 conv(
#                     in_channels[i], out_channels, 3, padding=1,
#                     conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))

#     def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
#         assert len(inputs) == len(self.in_channels)

#         # --- top-down path ---
#         inner_outs = [inputs[-1]]
#         for idx in range(len(self.in_channels) - 1, 0, -1):
#             feat_heigh = inner_outs[0]
#             feat_low = inputs[idx - 1]
#             feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](feat_heigh)
#             inner_outs[0] = feat_heigh

#             upsample_feat = self.upsample(feat_heigh)
            
#             # --- 【关键修复】 恢复使用 torch.cat ---
#             fused_feat = torch.cat([upsample_feat, feat_low], 1)
            
#             inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](fused_feat)
#             inner_outs.insert(0, inner_out)

#         # --- bottom-up path (保持不变，因为本来就是正确的) ---
#         outs = [inner_outs[0]]
#         for idx in range(len(self.in_channels) - 1):
#             feat_low = outs[-1]
#             feat_height = inner_outs[idx + 1]
#             downsample_feat = self.downsamples[idx](feat_low)
            
#             fused_feat = torch.cat([downsample_feat, feat_height], 1)
            
#             out = self.bottom_up_blocks[idx](fused_feat)
#             outs.append(out)

#         # --- out convs ---
#         for idx, conv in enumerate(self.out_convs):
#             outs[idx] = conv(outs[idx])

#         return tuple(outs)


@MODELS.register_module()
class CSPNeXtPAFPN_RCSSC(CSPNeXtPAFPN):
    """
    继承自官方的 CSPNeXtPAFPN，并用 RCSSC 替换了部分 CSPLayer。
    """
    def __init__(
        self,
        *args, # 接收所有来自父类的参数
        rcssc_locations: list = ['top_down'], # 你的自定义参数
        **kwargs
    ):
        # 必须先调用父类的 __init__ 方法，让它完成所有的模块创建！
        super().__init__(*args, **kwargs)
        
        # --- 在父类初始化后，再进行修改 ---
        # 现在 self.top_down_blocks 和 self.bottom_up_blocks 已经被创建好了
        
        # 首先，创建一个特殊的包装类，使RCSSC的输入输出通道保持一致
        class RCSSCWrapper(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.rcssc = RCSSC(n_feat=in_channels)
                # 如果输入输出通道不同，添加一个1x1卷积进行调整
                self.need_channel_adjust = in_channels != out_channels
                if self.need_channel_adjust:
                    self.channel_adjust = ConvModule(
                        in_channels, 
                        out_channels, 
                        1,
                        conv_cfg=kwargs.get('conv_cfg', None),
                        norm_cfg=kwargs.get('norm_cfg', dict(type='BN')),
                        act_cfg=kwargs.get('act_cfg', dict(type='SiLU'))
                    )
                
            def forward(self, x):
                out = self.rcssc(x)
                if self.need_channel_adjust:
                    out = self.channel_adjust(out)
                return out
        
        if 'top_down' in rcssc_locations:
            # 遍历并替换 top_down_blocks 中的 CSPLayer
            for i in range(len(self.top_down_blocks)):
                # 获取当前位置的原始块以确定正确的通道数
                original_block = self.top_down_blocks[i]
                # 计算出正确的输入输出通道数
                idx = len(self.in_channels) - 1 - i
                if idx > 0:  # 确保索引有效
                    # 获取输入输出通道数
                    in_channels_for_block = self.in_channels[idx-1] * 2  # 拼接后的通道数
                    out_channels_for_block = self.in_channels[idx-1]     # CSPLayer的输出通道数
                    
                    # 使用包装类来确保通道数匹配
                    self.top_down_blocks[i] = RCSSCWrapper(
                        in_channels_for_block,
                        out_channels_for_block
                    )

        if 'bottom_up' in rcssc_locations:
            # 遍历并替换 bottom_up_blocks 中的 CSPLayer
            for i in range(len(self.bottom_up_blocks)):
                if i < len(self.in_channels) - 1:
                    # 获取输入输出通道数
                    in_channels_for_block = self.in_channels[i] * 2   # 拼接后的通道数  
                    out_channels_for_block = self.in_channels[i + 1]  # CSPLayer的输出通道数
                    
                    # 使用包装类来确保通道数匹配
                    self.bottom_up_blocks[i] = RCSSCWrapper(
                        in_channels_for_block,
                        out_channels_for_block
                    )

    # 父类的forward方法可以直接复用，因为我们只是替换了模块，没有改变数据流