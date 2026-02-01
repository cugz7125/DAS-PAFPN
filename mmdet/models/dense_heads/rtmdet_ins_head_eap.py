import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmdet.registry import MODELS
from mmdet.models.dense_heads.rtmdet_ins_head import RTMDetInsHead
from mmdet.models.dense_heads.enhanced_parallel_attention import Enhanced_Parallel_Attention
from typing import Tuple
from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet.models.utils import sigmoid_geometric_mean

from torch import Tensor

@MODELS.register_module()
class RTMDetInsHeadEAP(RTMDetInsHead):
    """带有Enhanced Parallel Attention的RTMDetIns检测头
    
    在RTMDetInsHead的基础上增加了EAP注意力机制，分别应用于:
    1. 掩码原型生成（通过MaskFeatModuleEAP）
    2. 卷积核生成前（controller）

    Args:
        使用与RTMDetInsHead相同的参数，增加:
        use_eap (bool): 是否使用EAP
        eap_controller (bool): 是否在controller中使用EAP
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 use_eap=True,
                 eap_controller=True,
                 exp_on_reg=False,  # 添加exp_on_reg参数
                 **kwargs):
        self.use_eap = use_eap
        self.eap_controller = eap_controller
        self.exp_on_reg = exp_on_reg  # 保存exp_on_reg参数
        
        # 从kwargs中移除exp_on_reg，避免传递给父类
        if 'exp_on_reg' in kwargs:
            kwargs.pop('exp_on_reg')
        
        # 保存mask_head配置，但不传递给父类
        if 'mask_head' in kwargs:
            self.mask_head_cfg = kwargs.pop('mask_head')
        else:
            self.mask_head_cfg = dict(
                type='MaskFeatModuleEAP',
                in_channels=in_channels,
                feat_channels=256,
                stacked_convs=4,
                num_levels=3,
                num_prototypes=8,
                act_cfg=dict(type='ReLU', inplace=True),
                norm_cfg=dict(type='BN'),
                use_eap=use_eap
            )
            
        super().__init__(num_classes=num_classes, 
                         in_channels=in_channels, 
                         **kwargs)
        
        if self.use_eap and self.eap_controller:
            # 为每个尺度级别创建一个EAP模块，用于控制器
            self.kernel_eap = nn.ModuleList()
            for _ in range(len(self.prior_generator.strides)):
                self.kernel_eap.append(Enhanced_Parallel_Attention(self.feat_channels))
    
    def _init_layers(self) -> None:
        """初始化层"""
        # 调用父类的_init_layers，但不包括mask_head的初始化
        super(RTMDetHead, self)._init_layers()
        
        # 初始化kernel_convs和rtm_kernel，与父类相同
        self.kernel_convs = nn.ModuleList()
        # calculate num dynamic parameters
        weight_nums, bias_nums = [], []
        for i in range(self.num_dyconvs):
            if i == 0:
                weight_nums.append(
                    (self.num_prototypes + 2) * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels)
            elif i == self.num_dyconvs - 1:
                weight_nums.append(self.dyconv_channels)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dyconv_channels * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.kernel_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        pred_pad_size = self.pred_kernel_size // 2
        self.rtm_kernel = nn.Conv2d(
            self.feat_channels,
            self.num_gen_params,
            self.pred_kernel_size,
            padding=pred_pad_size)
            
        # 初始化mask_head
        self._init_mask_head()
    
    def _init_mask_head(self):
        """初始化掩码头"""
        if self.mask_head_cfg.get('type', None) == 'MaskFeatModuleEAP':
            self.mask_head = MODELS.build(self.mask_head_cfg)
        else:
            # 确保使用MaskFeatModuleEAP
            from mmdet.models.dense_heads.mask_feat_module_eap import MaskFeatModuleEAP
            
            # 复制配置但更改类型
            mask_head_cfg = self.mask_head_cfg.copy()
            if 'type' in mask_head_cfg:
                mask_head_cfg.pop('type')
            
            self.mask_head = MaskFeatModuleEAP(
                in_channels=self.feat_channels,
                feat_channels=self.feat_channels,
                stacked_convs=self.stacked_convs,
                num_levels=self.num_levels,
                num_prototypes=self.num_prototypes,
                use_eap=self.use_eap,
                **mask_head_cfg
            )

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """前向传播
        
        Args:
            feats (tuple[Tensor]): 特征图元组
            
        Returns:
            tuple: 分类分数、边界框预测、核预测和掩码特征的元组
        """
        # 使用掩码头生成mask_feat
        mask_feat = self.mask_head(feats)
        
        cls_scores = []
        bbox_preds = []
        kernel_preds = []
        
        for idx, (x, scale, stride) in enumerate(
                zip(feats, self.scales, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x
            kernel_feat = x

            # 处理分类特征
            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls(cls_feat)

            # 处理内核特征（添加EAP）
            for kernel_layer in self.kernel_convs:
                kernel_feat = kernel_layer(kernel_feat)
                
            # 在控制器前应用EAP
            if self.use_eap and self.eap_controller:
                kernel_feat = self.kernel_eap[idx](kernel_feat)
                
            kernel_pred = self.rtm_kernel(kernel_feat)

            # 处理回归特征
            for reg_layer in self.reg_convs:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj(reg_feat)
                cls_score = inverse_sigmoid(
                    sigmoid_geometric_mean(cls_score, objectness))

            # 根据exp_on_reg决定是否对回归预测进行指数变换
            if self.exp_on_reg:
                reg_dist = self.rtm_reg(reg_feat).exp() * stride[0]
            else:
                reg_dist = scale(self.rtm_reg(reg_feat)) * stride[0]

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            kernel_preds.append(kernel_pred)
            
        return tuple(cls_scores), tuple(bbox_preds), tuple(kernel_preds), mask_feat


@MODELS.register_module()
class RTMDetInsSepBNHeadEAP(RTMDetInsHeadEAP):
    """带有增强并行注意力和独立BN层的RTMDetIns头
    
    在RTMDetInsSepBNHead的基础上增加了EAP注意力机制
    """
    
    def __init__(self, 
                 num_classes,
                 in_channels,
                 share_conv=False,  # 默认为False，与原始RTMDetInsSepBNHead不同
                 exp_on_reg=False,  # 添加exp_on_reg参数
                 **kwargs):
        self.share_conv = share_conv  # 先保存share_conv参数
        
        # 从kwargs中移除share_conv，避免传递给父类
        if 'share_conv' in kwargs:
            kwargs.pop('share_conv')
            
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            exp_on_reg=exp_on_reg,  # 将exp_on_reg传递给RTMDetInsHeadEAP
            **kwargs)  # 不传递share_conv给父类
        
    def _init_layers(self):
        """初始化层"""
        # 不调用直接父类的_init_layers，因为我们需要完全重写
        # 初始化基本组件
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        
        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.rtm_kernel = nn.ModuleList()
        if self.with_objectness:
            self.rtm_obj = nn.ModuleList()
            
        self.scales = nn.ModuleList()

        # 计算动态参数数量
        weight_nums, bias_nums = [], []
        for i in range(self.num_dyconvs):
            if i == 0:
                weight_nums.append(
                    (self.num_prototypes + 2) * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels)
            elif i == self.num_dyconvs - 1:
                weight_nums.append(self.dyconv_channels)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dyconv_channels * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        for _ in range(len(self.prior_generator.strides)):
            self.scales.append(nn.ModuleList([Scale(1.0) for _ in range(1)]))
            
        for i in range(len(self.prior_generator.strides)):
            cls_convs = nn.ModuleList()
            for j in range(self.stacked_convs):
                chn = self.in_channels if j == 0 else self.feat_channels
                cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.cls_convs.append(cls_convs)
            
            reg_convs = nn.ModuleList()
            for j in range(self.stacked_convs):
                chn = self.in_channels if j == 0 else self.feat_channels
                reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.reg_convs.append(reg_convs)
            
            kernel_convs = nn.ModuleList()
            for j in range(self.stacked_convs):
                chn = self.in_channels if j == 0 else self.feat_channels
                kernel_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.kernel_convs.append(kernel_convs)
            
            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.cls_out_channels,
                    3,
                    padding=1))
            self.rtm_reg.append(
                nn.Conv2d(
                    self.feat_channels, self.num_base_priors * 4, 3, padding=1))
            self.rtm_kernel.append(
                nn.Conv2d(
                    self.feat_channels, self.num_gen_params, 
                    3, padding=1))
            
            if self.with_objectness:
                self.rtm_obj.append(
                    nn.Conv2d(
                        self.feat_channels, self.num_base_priors * 1, 3,
                        padding=1))
        
        # 如果share_conv为True，共享卷积层
        if self.share_conv:
            for n in range(len(self.prior_generator.strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv
                    self.kernel_convs[n][i].conv = self.kernel_convs[0][i].conv
                
        # 初始化EAP模块
        if self.use_eap and self.eap_controller:
            self.kernel_eap = nn.ModuleList()
            for _ in range(len(self.prior_generator.strides)):
                self.kernel_eap.append(Enhanced_Parallel_Attention(self.feat_channels))
                
        # 初始化mask_head
        self._init_mask_head()
    
    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """前向传播
        
        Args:
            feats (tuple[Tensor]): 特征图元组
            
        Returns:
            tuple: 分类分数、边界框预测、核预测和掩码特征的元组
        """
        # 使用掩码头生成mask_feat
        mask_feat = self.mask_head(feats)
        
        cls_scores = []
        bbox_preds = []
        kernel_preds = []
        
        for idx, (x, stride) in enumerate(
                zip(feats, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x
            kernel_feat = x

            # 处理分类特征
            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            # 处理内核特征（添加EAP）
            for kernel_layer in self.kernel_convs[idx]:
                kernel_feat = kernel_layer(kernel_feat)
                
            # 在控制器前应用EAP
            if self.use_eap and self.eap_controller:
                kernel_feat = self.kernel_eap[idx](kernel_feat)
                
            kernel_pred = self.rtm_kernel[idx](kernel_feat)

            # 处理回归特征
            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj[idx](reg_feat)
                cls_score = inverse_sigmoid(
                    sigmoid_geometric_mean(cls_score, objectness))

            # 根据exp_on_reg决定是否对回归预测进行指数变换
            if self.exp_on_reg:
                reg_dist = self.rtm_reg[idx](reg_feat).exp() * stride[0]
            else:
                reg_dist = self.rtm_reg[idx](reg_feat) * stride[0]

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            kernel_preds.append(kernel_pred)
            
        return tuple(cls_scores), tuple(bbox_preds), tuple(kernel_preds), mask_feat 