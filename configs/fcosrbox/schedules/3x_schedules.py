lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0/3,
    step=[24, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)