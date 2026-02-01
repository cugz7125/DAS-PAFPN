import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.models.dense_heads.rtmdet_ins_head import MaskFeatModule
from mmdet.models.dense_heads.enhanced_parallel_attention import Enhanced_Parallel_Attention
from typing import Tuple

from torch import Tensor

@MODELS.register_module()
class MaskFeatModuleEAP(MaskFeatModule):
    """带有Enhanced Parallel Attention的掩码特征模块
    
    在MaskFeatModule的基础上增加了EAP注意力机制，可以在原型生成前提升特征质量
    
    Args:
        in_channels (int): 输入特征的通道数
        feat_channels (int): 特征通道数
        stacked_convs (int): 堆叠卷积层数
        num_levels (int): 特征层数
        num_prototypes (int): 原型数量
        act_cfg (dict): 激活函数配置
        norm_cfg (dict): 归一化层配置
        use_eap (bool): 是否使用EAP模块
    """
    def __init__(
        self,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        num_levels: int = 3,
        num_prototypes: int = 8,
        act_cfg: dict = dict(type='ReLU', inplace=True),
        norm_cfg: dict = dict(type='BN'),
        use_eap: bool = True
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            feat_channels=feat_channels,
            stacked_convs=stacked_convs,
            num_levels=num_levels,
            num_prototypes=num_prototypes,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )
        
        self.use_eap = use_eap
        if self.use_eap:
            # 在原型生成前添加EAP模块
            self.eap = Enhanced_Parallel_Attention(feat_channels)
    
    def forward(self, features: Tuple[Tensor, ...]) -> Tensor:
        """前向传播
        
        Args:
            features (tuple[Tensor]): 各层特征图
            
        Returns:
            Tensor: 掩码特征图
        """
        # 多层特征融合（保持原有实现）
        fusion_feats = [features[0]]
        size = features[0].shape[-2:]
        for i in range(1, self.num_levels):
            f = F.interpolate(features[i], size=size, mode='bilinear')
            fusion_feats.append(f)
        fusion_feats = torch.cat(fusion_feats, dim=1)
        fusion_feats = self.fusion_conv(fusion_feats)
        
        # 预测掩码特征
        mask_features = self.stacked_convs(fusion_feats)
        
        # 在投影前应用EAP
        if self.use_eap:
            mask_features = self.eap(mask_features)
            
        mask_features = self.projection(mask_features)
        
        return mask_features 