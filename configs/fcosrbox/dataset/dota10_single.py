# dataset settings
image_size = (1024, 1024)
dataset_type = 'DOTADataset_10'
data_root = 'data/dota10_1024_4/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(type='RandomFlipPoly', mode='horizontal'),
    dict(type='RandomRotateRbox', rotate_mode='value', auto_bound=True, rotate_values=[0, 90, 180, -90]),
    dict(type='RandomRotateRbox', rotate_mode='value', auto_bound=True, rotate_ratio=0.5,
         rotate_values=[30, 60]),
    dict(type='Poly2Rbox', key='gt_rboxes', to_rbox=True, filter_cfg=dict(enable=True, mode='edge', threshold=8)),
    dict(type='Resize', img_scale=image_size, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='CustomFormatBundle', formatting_keywords=['img', 'gt_rboxes', 'gt_labels']),
    dict(type='Collect', keys=['img', 'gt_rboxes', 'gt_labels'], meta_keys=['filename', 'ori_filename', 'img_shape'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=image_size, keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval1024/DOTA1_0_trainval1024.json',
        img_prefix=data_root + 'trainval1024/images/',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval1024/DOTA1_0_trainval1024.json',
        img_prefix=data_root + 'trainval1024/images/',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test1024/DOTA1_0_test1024.json',
        img_prefix=data_root + 'test1024/images',
        pipeline=test_pipeline
    )
)