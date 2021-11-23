_base_ = ['./dataset/hrsc2016.py', './schedules/40k_iter_hrsc.py']
image_size = (800, 800)
model = dict(
    type='FCOSR',
    backbone=dict(
        type='ResNeXt',
        groups=32,
        base_width=4,
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='torchvision://resnext50_32x4d')),
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True),
    rbbox_head=dict(
        type='FCOSRboxHead',
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        stacked_convs=4,
        strides=(8, 16, 32, 64, 128),
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                        (512, 100000000.0)),
        image_size=(800, 800),
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
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/HRSC/FCOSR_rx50_32x4d_fpn_40k_hrsc2016'
find_unused_parameters = True
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 4)
