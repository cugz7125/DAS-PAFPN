_base_ = '../rtmdet/rtmdet-ins_s_8xb32-300e_coco.py'

model = dict(
    backbone=dict(
        type='CSPNeXt_GMM',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        gmm_stages=(2, 3, 4),  # 在哪些stage应用GMM
        gmm_patch=4,  # GMM的patch参数
        gmm_max_relative_position=50,  # GMM的最大相对位置参数
    ),
)

# 修改训练配置
train_cfg = dict(max_epochs=300, val_interval=10)

# 给这个配置文件一个有意义的名称
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=10, max_keep_ckpts=3,
        save_best='coco/segm_mAP'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# 可视化配置
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# 给配置文件添加一些注释
# GMM (Global Mixed Module) 是一种全局混合模块，用于增强特征提取
# 该模块通过行列重组、跨区卷积和特征融合，捕获长距离依赖关系和全局上下文信息
# 参考论文：STMNet: Single-Temporal Mask-based Network for Self-Supervised Hyperspectral Change Detection 