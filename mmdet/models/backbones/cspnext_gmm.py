# # mmdet/models/backbones/cspnext_gmm.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmdet.registry import MODELS
# from mmdet.models.backbones import CSPNeXt # 导入父类
# from timm.models.layers import trunc_normal_
# from typing import Sequence, Tuple
# from torch import Tensor
# from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig # 导入缺失的类型

# # --- 1. GMM 相关模块 (这部分是独立的，保持不变) ---
# class RelativePosition(nn.Module):
#     # ... (你之前的 RelativePosition 代码，无需修改) ...
#     def __init__(self, num_units: int, max_relative_position: int):
#         super().__init__()
#         self.num_units = num_units
#         self.max_relative_position = max_relative_position
#         self.embeddings_table = nn.Parameter(
#             torch.Tensor(max_relative_position * 2 + 1, num_units))
#         trunc_normal_(self.embeddings_table, std=.02)

#     def forward(self, length_q: int, length_k: int) -> Tensor:
#         device = self.embeddings_table.device
#         range_vec_q = torch.arange(length_q, device=device)
#         range_vec_k = torch.arange(length_k, device=device)
#         distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
#         distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
#         final_mat = distance_mat_clipped + self.max_relative_position
#         final_mat = final_mat.long()
#         embeddings = self.embeddings_table[final_mat]
#         return embeddings

# # --- 2. 经过精炼和简化的 GMM_Flexible 模块 ---
# @MODELS.register_module()
# class GMM_Flexible(nn.Module):
#     def __init__(self, channels: int, patch: int = 4, max_relative_position: int = 50):
#         super().__init__()
#         self.channels = channels
#         self.patch = patch
#         if channels % patch != 0:
#             raise ValueError(f"Channels ({channels}) must be divisible by patch ({patch}).")
#         self.Cg = channels // patch  # Channels per group

#         self.fuse_h = nn.Conv2d(channels * 2, channels, 1, bias=False)
#         self.fuse_w = nn.Conv2d(channels * 2, channels, 1, bias=False)
#         self.activation = nn.GELU()
#         self.bn = nn.BatchNorm2d(channels)
#         self.rel_pos_h = RelativePosition(self.Cg, max_relative_position)
#         self.rel_pos_w = RelativePosition(self.Cg, max_relative_position)
        
#         # 投影层，现在是标准的 1x1 卷积，作用于通道维度
#         self.proj_h = nn.Conv2d(self.Cg, self.Cg, 1)
#         self.proj_w = nn.Conv2d(self.Cg, self.Cg, 1)

#     def forward(self, x: Tensor) -> Tensor:
#         N, C, H, W = x.shape
#         identity = x

#         # 1. 通道分组
#         x_grouped = x.view(N, self.patch, self.Cg, H, W)
        
#         # --- 2. 水平混合 (Horizontal Mixing) ---
#         # 交换 H 和 Cg 维度 -> (N, patch, H, Cg, W)
#         x_h = x_grouped.permute(0, 1, 3, 2, 4).contiguous()
#         # 添加水平相对位置编码 (需要调整形状以匹配)
#         rel_h = self.rel_pos_h(H, H).unsqueeze(0).unsqueeze(0) # (1, 1, H, H, Cg)
#         # x_h shape: (N, patch, H, Cg, W)
#         # rel_h shape: (1, 1, H, H, Cg) -> (N, patch, H, H, Cg)
#         # TODO: 相对位置编码与特征的融合方式需要更仔细的设计
#         # 简化版：我们暂时不加相对位置编码，以确保主干逻辑正确
        
#         # 展平 H 维度以进行全连接层（用1x1卷积模拟）
#         x_h = x_h.view(N * self.patch, H, self.Cg, W).permute(0, 2, 1, 3) # (N*p, Cg, H, W)
#         x_h = self.proj_h(x_h)
#         # 恢复形状
#         x_h = x_h.permute(0, 2, 1, 3).view(N, self.patch, H, self.Cg, W)
#         # 恢复原始通道顺序
#         x_h = x_h.permute(0, 1, 3, 2, 4).contiguous().view(N, C, H, W)
        
#         # 融合
#         x_h = self.fuse_h(torch.cat([x_h, identity], dim=1))
#         x_h = self.activation(self.bn(x_h))
        
#         # --- 3. 垂直混合 (Vertical Mixing) ---
#         x_w_input = x_h # 使用水平混合后的结果
#         x_w_grouped = x_w_input.view(N, self.patch, self.Cg, H, W)
        
#         # 交换 W 和 Cg 维度 -> (N, patch, W, Cg, H)
#         x_w = x_w_grouped.permute(0, 1, 4, 2, 3).contiguous()
        
#         # 展平 W 维度
#         x_w = x_w.view(N * self.patch, W, self.Cg, H).permute(0, 2, 1, 3) # (N*p, Cg, W, H)
#         x_w = self.proj_w(x_w)
#         # 恢复形状
#         x_w = x_w.permute(0, 2, 1, 3).view(N, self.patch, W, self.Cg, H)
#         # 恢复原始通道顺序
#         x_w = x_w.permute(0, 1, 3, 4, 2).contiguous().view(N, C, H, W)

#         # 最终融合
#         x = self.fuse_w(torch.cat([identity, x_w], dim=1))

#         return x
# # cspnext_gmm.py

# # ... (imports 和其他类保持不变) ...
# # ... (RelativePosition 和 GMM_Flexible 保持不变) ...

# # cspnext_gmm.py

# # ... (imports, RelativePosition, GMM_Flexible 保持不变) ...

# # cspnext_gmm.py

# # ... (imports, RelativePosition, GMM_Flexible 保持不变) ...

# @MODELS.register_module()
# class CSPNeXt_GMM(CSPNeXt):
#     # __init__ 方法保持我们上次校准后的最终版本，它是正确的
#     def __init__(self,
#                  arch: str = 'P5',
#                  deepen_factor: float = 1.0,
#                  widen_factor: float = 1.0,
#                  out_indices: Sequence[int] = (2, 3, 4),
#                  frozen_stages: int = -1,
#                  use_depthwise: bool = False,
#                  expand_ratio: float = 0.5,
#                  arch_ovewrite: dict = None,
#                  spp_kernel_sizes: Sequence[int] = (5, 9, 13),
#                  channel_attention: bool = True,
#                  conv_cfg: OptConfigType = None,
#                  norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
#                  act_cfg: ConfigType = dict(type='SiLU'),
#                  norm_eval: bool = False,
#                  init_cfg: OptMultiConfig = None,
#                  gmm_stages: Sequence[int] = (2, 3, 4),
#                  gmm_patch: int = 4,
#                  gmm_max_relative_position: int = 50):
        
#         super().__init__(
#             arch=arch, deepen_factor=deepen_factor, widen_factor=widen_factor,
#             out_indices=out_indices, frozen_stages=frozen_stages,
#             use_depthwise=use_depthwise, expand_ratio=expand_ratio,
#             arch_ovewrite=arch_ovewrite, spp_kernel_sizes=spp_kernel_sizes,
#             channel_attention=channel_attention, conv_cfg=conv_cfg,
#             norm_cfg=norm_cfg, act_cfg=act_cfg, norm_eval=norm_eval,
#             init_cfg=init_cfg)
        
#         self.gmm_stages = gmm_stages
#         self.gmm_modules = nn.ModuleList()

#         # 父类在 __init__ 中已经计算好了 arch_setting
#         arch_setting = self.arch_settings[arch]
#         if arch_ovewrite:
#             arch_setting = arch_ovewrite
            
#         # 父类在 __init__ 中已经创建了 self.stage1, self.stage2, ...
#         # 我们只需要根据它们的输出通道数来创建 GMM 模块
#         # 注意：CSPNeXt 的 out_channels 是一个动态计算的属性，我们最好自己算一遍
#         current_channels = int(arch_setting[0][0] * widen_factor)
#         all_stage_out_channels = []
#         for i in range(len(arch_setting)):
#             out_channels = int(arch_setting[i][1] * widen_factor)
#             all_stage_out_channels.append(out_channels)
#             current_channels = out_channels

#         for i, stage_out_channels in enumerate(all_stage_out_channels):
#             stage_idx = i + 1
#             if stage_idx in self.gmm_stages:
#                 self.gmm_modules.append(
#                     GMM_Flexible(
#                         channels=stage_out_channels, patch=gmm_patch,
#                         max_relative_position=gmm_max_relative_position))
#             else:
#                 self.gmm_modules.append(nn.Identity())

#     # --- 【关键修复】重写 forward 方法，严格模仿父类 ---
#     def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
#         outs = []
        
#         # self.layers 是由父类 __init__ 创建的，例如 ['stem', 'stage1', 'stage2', 'stage3', 'stage4']
#         for i, layer_name in enumerate(self.layers):
#             layer = getattr(self, layer_name)
#             x = layer(x)
            
#             # 只有当 layer_name 是 'stageX' 时，我们才应用 GMM
#             if 'stage' in layer_name:
#                 stage_idx_in_list = int(layer_name.replace('stage', '')) -1 # 0, 1, 2, 3
#                 x = self.gmm_modules[stage_idx_in_list](x)

#             # self.out_indices 是 [2, 3, 4]，它指的是 layers 列表的索引
#             # 所以当 i = 2, 3, 4 时，对应的是 'stage2', 'stage3', 'stage4'
#             if i in self.out_indices:
#                 outs.append(x)
                
#         return tuple(outs)




# mmdet/models/backbones/cspnext_gmm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.models.backbones import CSPNeXt
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple
from torch import Tensor
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

# --- 1. RelativePosition 模块 (保持不变) ---
class RelativePosition(nn.Module):
    def __init__(self, num_units: int, max_relative_position: int):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(
            torch.Tensor(max_relative_position * 2 + 1, num_units))
        trunc_normal_(self.embeddings_table, std=.02)

    def forward(self, length_q: int, length_k: int) -> Tensor:
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings

# --- 2. 恢复了核心逻辑的 GMM_Flexible 模块 ---
@MODELS.register_module()
class GMM_Flexible(nn.Module):
    def __init__(self, channels: int, patch: int = 4, max_relative_position: int = 50):
        super().__init__()
        self.channels = channels
        self.patch = patch
        if channels % patch != 0:
            raise ValueError(f"Channels ({channels}) must be divisible by patch ({patch}).")
        self.Cg = channels // patch

        self.fuse_h = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.fuse_w = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.activation = nn.GELU()
        self.bn = nn.BatchNorm2d(channels)
        self.rel_pos_h = RelativePosition(self.Cg, max_relative_position)
        self.rel_pos_w = RelativePosition(self.Cg, max_relative_position)
        
        # 这里的 proj 层就是实现全局交互的关键
        self.proj_h = nn.Conv2d(self.Cg, self.Cg, 3, padding=1, groups=self.Cg)
        self.proj_w = nn.Conv2d(self.Cg, self.Cg, 3, padding=1, groups=self.Cg)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        identity = x

        # 1. 通道分组
        x_grouped = x.view(N, self.patch, self.Cg, H, W)

        # --- 2. 水平混合 (Horizontal Mixing) ---
        # 交换 H 和 Cg 维度 -> (N, patch, H, Cg, W)
        x_h = x_grouped.permute(0, 1, 3, 2, 4).contiguous()
        
        # TODO: 如何融合相对位置编码是一个开放问题，暂时简化
        # rel_h = self.rel_pos_h(H, H).unsqueeze(0).unsqueeze(0)
        # x_h = x_h + rel_h # 维度不匹配，融合方式需要设计
        
        # 将 Batch 和 H 维度展平，以便进行卷积
        # 形状变为 (N*patch*H, Cg, 1, W)
        x_h = x_h.view(N * self.patch * H, self.Cg, 1, W)
        x_h = self.proj_h(x_h)
        # 恢复形状 (N, patch, H, Cg, W)
        x_h = x_h.view(N, self.patch, H, self.Cg, W)

        # 恢复原始通道顺序
        x_h = x_h.permute(0, 1, 3, 2, 4).contiguous().view(N, C, H, W)
        
        x_h = self.fuse_h(torch.cat([x_h, identity], dim=1))
        x_h = self.activation(self.bn(x_h))
        
        # --- 3. 垂直混合 (Vertical Mixing) ---
        x_w_input = x_h
        x_w_grouped = x_w_input.view(N, self.patch, self.Cg, H, W)
        
        # 交换 W 和 Cg 维度 -> (N, patch, W, Cg, H)
        x_w = x_w_grouped.permute(0, 1, 4, 2, 3).contiguous()
        
        # 将 Batch 和 W 维度展平，以便进行卷积
        # 形状变为 (N*patch*W, Cg, H, 1)
        x_w = x_w.view(N * self.patch * W, self.Cg, H, 1)
        x_w = self.proj_w(x_w)
        # 恢复形状 (N, patch, W, Cg, H)
        x_w = x_w.view(N, self.patch, W, self.Cg, H)
        
        # 恢复原始通道顺序
        x_w = x_w.permute(0, 1, 3, 4, 2).contiguous().view(N, C, H, W)

        x = self.fuse_w(torch.cat([identity, x_w], dim=1))

        return x

# --- 3. 最终的、健壮的自定义 Backbone (保持不变) ---
@MODELS.register_module()
class CSPNeXt_GMM(CSPNeXt):
    # __init__ 方法保持我们上次校准后的最终版本，它是正确的
    def __init__(self,
                 arch: str = 'P5',
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 out_indices: Sequence[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 use_depthwise: bool = False,
                 expand_ratio: float = 0.5,
                 arch_ovewrite: dict = None,
                 spp_kernel_sizes: Sequence[int] = (5, 9, 13),
                 channel_attention: bool = True,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU'),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None,
                 gmm_stages: Sequence[int] = (2, 3, 4),
                 gmm_patch: int = 4,
                 gmm_max_relative_position: int = 50):
        
        super().__init__(
            arch=arch, deepen_factor=deepen_factor, widen_factor=widen_factor,
            out_indices=out_indices, frozen_stages=frozen_stages,
            use_depthwise=use_depthwise, expand_ratio=expand_ratio,
            arch_ovewrite=arch_ovewrite, spp_kernel_sizes=spp_kernel_sizes,
            channel_attention=channel_attention, conv_cfg=conv_cfg,
            norm_cfg=norm_cfg, act_cfg=act_cfg, norm_eval=norm_eval,
            init_cfg=init_cfg)
        
        self.gmm_stages = gmm_stages
        self.gmm_modules = nn.ModuleList()

        # 父类在 __init__ 中已经计算好了 arch_setting
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
            
        # 父类在 __init__ 中已经创建了 self.stage1, self.stage2, ...
        # 我们只需要根据它们的输出通道数来创建 GMM 模块
        # 注意：CSPNeXt 的 out_channels 是一个动态计算的属性，我们最好自己算一遍
        current_channels = int(arch_setting[0][0] * widen_factor)
        all_stage_out_channels = []
        for i in range(len(arch_setting)):
            out_channels = int(arch_setting[i][1] * widen_factor)
            all_stage_out_channels.append(out_channels)
            current_channels = out_channels

        for i, stage_out_channels in enumerate(all_stage_out_channels):
            stage_idx = i + 1
            if stage_idx in self.gmm_stages:
                self.gmm_modules.append(
                    GMM_Flexible(
                        channels=stage_out_channels, patch=gmm_patch,
                        max_relative_position=gmm_max_relative_position))
            else:
                self.gmm_modules.append(nn.Identity())

    # --- 【关键修复】重写 forward 方法，严格模仿父类 ---
    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        outs = []
        
        # self.layers 是由父类 __init__ 创建的，例如 ['stem', 'stage1', 'stage2', 'stage3', 'stage4']
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            
            # 只有当 layer_name 是 'stageX' 时，我们才应用 GMM
            if 'stage' in layer_name:
                stage_idx_in_list = int(layer_name.replace('stage', '')) -1 # 0, 1, 2, 3
                x = self.gmm_modules[stage_idx_in_list](x)

            # self.out_indices 是 [2, 3, 4]，它指的是 layers 列表的索引
            # 所以当 i = 2, 3, 4 时，对应的是 'stage2', 'stage3', 'stage4'
            if i in self.out_indices:
                outs.append(x)
                
        return tuple(outs)