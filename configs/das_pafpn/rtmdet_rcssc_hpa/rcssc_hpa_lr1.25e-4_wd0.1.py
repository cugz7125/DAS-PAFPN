_base_ = './model_rtmdet-ins_s_rcssc_hpa_cug.py'

max_epochs = 120
stage2_num_epochs = 10
interval = 5
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=2,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])
base_lr = 0.000125
weight_decay = 0.1


# 优化器
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper', # 使用混合精度训练
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
    )

# 学习率
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]



# --- 重写学习率和优化器配置 ---
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR', eta_min=base_lr * 0.05,
        begin=max_epochs // 2, end=max_epochs,
        T_max=max_epochs // 2, by_epoch=True, convert_to_iter_based=True)
]
optim_wrapper = dict(
    optimizer=dict(lr=base_lr, weight_decay=weight_decay),
    clip_grad=dict(max_norm=35, norm_type=2)
)



work_dir = './work_dirs/rcssc_hpa_lr1.25e-4_wd0.1'