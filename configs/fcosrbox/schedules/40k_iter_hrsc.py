lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=400,
    warmup_ratio=1.0/3,
    step=[30000, 36000])
runner = dict(type='IterBasedRunner', max_iters=40000)

checkpoint_config = dict(interval=1000)