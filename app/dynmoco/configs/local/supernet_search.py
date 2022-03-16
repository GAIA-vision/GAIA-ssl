import copy
import os.path as osp
_root_ = '../../../..'
_base_ = [osp.join(_root_, 'configs/base.py'),'../model_samplers/ar50to101v2.py']
#_base_ = [osp.join(_root_, 'configs/base.py')]


_data_root_ = '/data1/imagenet'
# model settings
model = dict(
    type='DynamicMOCO',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
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
data_train_list = osp.join(_data_root_, 'train_10percent.txt')
data_train_root = osp.join(_data_root_, 'ILSVRC2012_img_train')
dataset_type = 'ContrastiveDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(1., 1.)), 
    dict(type='RandomHorizontalFlip'),
]
# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=128,  # total 32*8=256
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
#use_fp16 = True
#optimizer_config = dict(use_fp16=use_fp16)
# optimizer
optimizer = dict(type='SGD', lr=0.12, weight_decay=0.0001, momentum=0.9)

lr_config = dict(policy='CosineAnnealing', min_lr=0.)
checkpoint_config = dict(interval=20)
# runtime settings
total_epochs = 1
work_dir = "/data2/OpenSelfSup-gaia/workdirs/moco"

model_space_path = '../../../../hubs/flops.json' #
model_sampling_rules = dict(
    type='sequential',
    rules=[
        dict(
            type='parallel',
            rules=[

                 dict(func_str='lambda x: x[\'overhead.flops\'] >=3000000000 and x[\'overhead.flops\']<4000000000'),
            ]
        ),
        dict(func_str='lambda x: x[\'data.input_shape\'] == 224'),

        dict(
            type='sample',
            operation='random',
            value=40, 
            mode='number',
        ),

        dict(type='merge'),
    ]
)
