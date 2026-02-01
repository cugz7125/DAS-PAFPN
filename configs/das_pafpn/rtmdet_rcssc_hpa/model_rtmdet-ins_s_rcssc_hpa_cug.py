_base_ = '/.conda/envs/newmm/mmdetection-main/configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py'

custom_imports = dict(
    imports=[
        'mmdet.models.necks.cspnext_pafpn_rcssc',
        'mmdet.models.necks.cspnext_pafpn_rcssc_hpa'
    ],
    allow_failed_imports=False
)
# 修改使用的neckRCSSC_HPA版本
model = dict(
    neck=dict(
        _delete_=True,  
        type='CSPNeXtPAFPN_RCSSC_HPA',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        # RCSSC模块位置配置（继承自CSPNeXtPAFPN_RCSSC）
        rcssc_locations=['top_down'],
        # 是否使用HPA模块进行特征增强
        use_hpa=True,
        # HPA模块的分组因子
        hpa_factor=32
    )
)

# 数据加载器配置
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='CachedMosaic', img_scale=(480, 480), pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(480, 480),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(480, 480),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(480, 480), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(480, 480),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict( 
        type='RandomResize',
        scale=(480, 480),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(480, 480),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(480, 480), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(480, 480), keep_ratio=True),
    dict(type='Pad', size=(480, 480), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True,with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    batch_sampler=None,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    persistent_workers=True,
    batch_size=4, num_workers=8, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader



val_evaluator = dict(metric=['bbox', 'segm'])
test_evaluator = val_evaluator



default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=3,
        save_best='coco/segm_mAP',
        rule='greater')
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=280,
        switch_pipeline=train_pipeline_stage2)
]

