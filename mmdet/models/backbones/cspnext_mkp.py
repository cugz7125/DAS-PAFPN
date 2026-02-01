import math
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from ..utils.mkp_conv import MKPConv
from ..utils.csp_layers_mkp import CSPLayerWithTweaks, CSPNeXtBlock


class CSPNeXtBlockWithMKP(BaseModule):
    """CSPNeXt Block with MKP module integrated.

    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        expansion (float): Ratio to adjust the number of channels of the hidden
            layer. Default: 0.5
        add_identity (bool): Whether to add identity in blocks.
            Default: True
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False
        kernel_size (int): The kernel size of the second convolution layer.
            Default: 5.
        use_mkp (bool): Whether to use MKP module. Default: True
        norm_cfg (ConfigType): Config dict for normalization layer.
            Default: dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (ConfigType): Config dict for activation layer.
            Default: dict(type='SiLU').
        init_cfg (OptMultiConfig, optional): Initialization config dict.
            Default: None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        add_identity: bool = True,
        use_depthwise: bool = False,
        kernel_size: int = 5,
        use_mkp: bool = False,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU'),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        hidden_channels = int(out_channels * expand_ratio)
        self.use_mkp = use_mkp
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
        if use_depthwise:
            self.conv2 = DepthwiseSeparableConvModule(
                hidden_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.conv2 = ConvModule(
                hidden_channels,
                out_channels,
                3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            
        # 如果启用MKP，创建MKP模块用于增强特征
        if self.use_mkp:
            self.mkp_module = MKPConv(out_channels, norm_cfg=norm_cfg)
            
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        # 如果启用MKP，使用MKP模块增强特征
        if self.use_mkp:
            out = self.mkp_module(out)
            
        if self.add_identity:
            return out + identity
        else:
            return out


@MODELS.register_module()
class CSPNeXtWithMKP(BaseModule):
    """CSPNeXt backbone with MKP integration.
    
    Args:
        arch (str): Architecture of CSPNeXt, from {P5, P6}.
            Defaults to P5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        arch_ovewrite (list): Overwrite default arch settings.
            Defaults to None.
        use_mkp (bool): Whether to use MKP module. Default: True
        mkp_stages (list): Stages to apply MKP. Default: [2, 3, 4]
        spp_kernel_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Default: (5, 9, 13).
        channel_attention (bool): Whether to add channel attention in each
            stage. Defaults to True.
        conv_cfg (ConfigType): Config dict for convolution layer.
            Defaults to None.
        norm_cfg (ConfigType): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (ConfigType): Config dict for activation layer.
            Defaults to dict(type='SiLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (Union[dict, list[dict]], optional): Initialization config
            dict. Defaults to None.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer in CSP layer. Defaults to 0.5
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(
        self,
        arch: str = 'P5',
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        use_depthwise: bool = False,
        arch_ovewrite: dict = None,
        use_mkp: bool = True,
        mkp_stages: list = [2, 3, 4],
        spp_kernel_sizes: Sequence[int] = (5, 9, 13),
        channel_attention: bool = True,
        conv_cfg: ConfigType = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU'),
        norm_eval: bool = False,
        init_cfg: OptMultiConfig = None,
        expand_ratio: float = 0.5,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        self.use_mkp = use_mkp
        self.mkp_stages = mkp_stages
        self.expand_ratio = expand_ratio

        # stem
        self.stem = ConvModule(
            3,
            int(arch_setting[0][0] * widen_factor),
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # 如果启用MKP，并且第一阶段在mkp_stages中，则创建MKP模块
        if self.use_mkp and 0 in self.mkp_stages:
            self.stem_mkp = MKPConv(int(arch_setting[0][0] * widen_factor), norm_cfg=norm_cfg)

        # build the first stage
        self.stages = nn.ModuleList()
        in_channels = int(arch_setting[0][0] * widen_factor)
        for i, setting in enumerate(arch_setting):
            # 正确解包5个值
            out_channels, num_blocks, add_identity, use_spp = setting[:4]
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            # build the first block
            stage.append(
                ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            # build the consecutive blocks
            csp_layer = CSPLayerWithTweaks(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                use_cspnext_block=True,
                use_mkp=self.use_mkp and (i + 1) in self.mkp_stages,
                expand_ratio=self.expand_ratio,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

            stage.append(csp_layer)
            if use_spp:
                spp_modules = []
                spp_modules.append(
                    ConvModule(
                        out_channels,
                        out_channels,
                        1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
                
                for kernel_size in spp_kernel_sizes:
                    spp_modules.append(
                        nn.MaxPool2d(
                            kernel_size=kernel_size,
                            stride=1,
                            padding=kernel_size // 2))
                
                spp_modules.append(
                    ConvModule(
                        out_channels * (len(spp_kernel_sizes) + 1),
                        out_channels,
                        1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
                
                spp = nn.Sequential(*spp_modules)
                stage.append(spp)
            
            # 如果启用MKP，并且当前阶段在mkp_stages中，则创建MKP模块
            if self.use_mkp and (i + 1) in self.mkp_stages:
                stage.append(MKPConv(out_channels, norm_cfg=norm_cfg))
                
            if channel_attention:
                stage.append(ChannelAttention(out_channels))
            self.stages.append(nn.Sequential(*stage))
            in_channels = out_channels

        self._freeze_stages()

    def _freeze_stages(self) -> None:
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            m = self.stages[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> tuple:
        """Forward function."""
        x = self.stem(x)
        
        # 如果启用MKP，并且第一阶段在mkp_stages中，则应用MKP模块
        if self.use_mkp and hasattr(self, 'stem_mkp'):
            x = self.stem_mkp(x)
            
        outputs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outputs.append(x)

        return tuple(outputs)

    def train(self, mode: bool = True) -> None:
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


class ChannelAttention(nn.Module):
    """Channel attention module for CSPNeXt."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        return x * self.act(self.conv(self.pool(x))) 