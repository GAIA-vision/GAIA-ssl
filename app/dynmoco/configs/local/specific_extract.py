import copy
import os.path as osp
_root_ = '../../../..'
_base_ = [osp.join(_root_, 'configs/base.py')]
_data_root_ = '/home/data'

R_specific = {
    'arch.encoder_q.stem.width': 64,
    'arch.encoder_q.body.width': [64, 128, 256, 512],
    'arch.encoder_q.body.depth': [3,4,6,3],
}
R_TN = {
    'arch.encoder_k.stem.width': 32,
    'arch.encoder_k.body.width': [48, 96, 192, 640],
    'arch.encoder_k.body.depth': [2, 4, 15, 2],
}
train_sampler = dict(
    type='concat',
    model_samplers=[
        dict(
            type='anchor',
            anchors=[
               dict(
                    name='RSPECIFIC',
                    **R_specific,
                ),
            ],
        ),
    ],
)

# model settings
model = dict(
    type='DynamicMOCO',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    same_arch=True,
    backbone=dict(
        type='DynamicResNet',
        in_channels=3,
        stem_width=64,
        body_depth=[4, 6, 29, 4],
        body_width=[80, 160, 320, 640],
        num_stages=4,
        out_indices=[3],  # 0: conv-1, x: stage-x
        conv_cfg=dict(type='DynConv2d'),
        norm_cfg=dict(type='DynBN', requires_grad=True),
        style='pytorch',
    ),
    neck=dict(
        type='DynamicNonLinearNeckV1',
        in_channels=2560,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    return_label=False,
    )
data_train_list = osp.join(_data_root_, 'imagenet/train_list_sample_with_image.txt')
data_train_root = osp.join(_data_root_, 'imagenet/train')
dataset_type = 'ContrastiveDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=0.5),
    dict(type='RandomHorizontalFlip'),
]
# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=32,  
    workers_per_gpu=2,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ))
# optimizer
optimizer = dict(type='SGD', lr=0.03, weight_decay=0.0001, momentum=0.9)

lr_config = dict(policy='CosineAnnealing', min_lr=0.)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 100

