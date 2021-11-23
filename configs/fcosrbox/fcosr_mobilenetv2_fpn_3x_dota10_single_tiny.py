_base_ = ['./dataset/dota10_single.py', './schedules/3x_schedules.py']
image_size = (1024, 1024)
model = dict(
    type='FCOSR',
    backbone=dict(
        type='MobileNetV2_N',
        out_indices=(2, 4, 6),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')),
    neck=dict(
        type='FPN',
        in_channels=[32, 96, 320],
        out_channels=128,
        start_level=0,
        num_outs=5),
    rbbox_head=dict(
        type='FCOSRboxHead',
        num_classes=15,
        in_channels=128,
        feat_channels=128,
        stacked_convs=4,
        strides=(8, 16, 32, 64, 128),
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                        (512, 100000000.0)),
        image_size=image_size,
        use_sim_ota=False,
        dcn_on_last_conv=False,
        gauss_factor=12.0,
        drop_positive_sample=dict(
            enable=False, mode='local', iou_threshold=0.6, keep_min=1),
        loss_cfg=dict(
            regress=[dict(type='ProbiouLoss', mode='l1', loss_weight=1.0)],
            classify=dict(
                type='QualityFocalLoss',
                use_sigmoid=True,
                beta=2.0,
                reduction='mean',
                loss_weight=1.0),
            classify_score=dict(type='iou'),
            regress_weight=dict(type='iou')),
        init_cfg=dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=dict(
                type='Normal', name='fcos_cls', std=0.01, bias_prob=0.01))),
    train_cfg=dict(gamma=2.0, alpha=0.25),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=8,
        score_thr=0.1,
        nms=dict(type='py_cpu_nms_poly_fast', iou_thr=0.1),
        max_per_img=2000,
        totoal_nms=dict(enable=False, iou_thr=0.8),
        rotate_test=dict(enable=False, rot90=[0, 1, 2, 3]),
        clip_result=False))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
checkpoint_config = dict(interval=1)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/DOTA10/FCOSR-tiny/FCOSR_mobilenetv2_fpn_3x_dota10_single_tiny'
find_unused_parameters = True
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 4)
