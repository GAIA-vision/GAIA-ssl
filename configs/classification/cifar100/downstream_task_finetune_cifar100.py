manipulate_arch = False
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DynamicResNet',
        in_channels=3,
        stem_width=64,
        body_depth=[2, 6, 9, 3],
        body_width=[48, 128, 192, 512],
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        conv_cfg=dict(type='DynConv2d'),
        norm_eval=False,
        norm_cfg=dict(type='DynBN', requires_grad=True)),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=100,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_soft=True)),
    train_cfg=dict(
        augments=dict(type='BatchCutMix', alpha=0.1, num_classes=100,
                      prob=1.0)))

dataset_type = 'CIFAR100'
img_norm_cfg = dict(
    mean=[129.304, 124.070, 112.434],
    std=[68.170, 65.392, 70.418],
    to_rgb=False)
train_pipeline = [
    dict(type='Resize', size=(160,160)),
    dict(type='RandomCrop', size=128),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Resize', size=(128,128)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=64, # according to 8 GPUs
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/cifar100',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/cifar100',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='data/cifar100',
        pipeline=test_pipeline,
        test_mode=True))

evaluation = dict(interval=1000, metric='accuracy')
# lr, 0.03, 0.01, 0.01
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[3000, 6000, 9000])
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(interval=5000)
log_config = dict(interval=2001, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/ai020/ckpt/30/c3fa1f19_backbone.pth'
resume_from = None#'/mnt/diske/qing_chang/GAIA/workdirs/gaia-cls-imagenet-train-supernet-moreepoch/epoch_40.pth'
workflow = [('train', 1)]
work_dir = '/home/ai020/111367'



gpu_ids = range(0, 4)
