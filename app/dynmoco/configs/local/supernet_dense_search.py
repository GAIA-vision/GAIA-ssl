import copy
import os.path as osp
_root_ = '../../../..'
_base_ = [osp.join(_root_, 'configs/base.py'),'../model_samplers/ar50to101v2.py']


manipulate_arch = False
_data_root_ = '/data1/Data/imagenet'
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
        out_indices=[3],
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

# ImageNet
data_train_list = osp.join(_data_root_, 'train_10percent.txt')
data_train_root = osp.join(_data_root_, 'ILSVRC2012_img_train')


dataset_type = 'ContrastiveDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# for other dataset, please keep the same test pipeline  except for the crop, too large crop will case memory out
train_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=16,  # total 32*8=256
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

optimizer = dict(type='SGD', lr=0.12, weight_decay=0.0001, momentum=0.9)

lr_config = dict(policy='CosineAnnealing', min_lr=0.)
checkpoint_config = dict(interval=20)
# runtime settings
total_epochs = 1
work_dir = "/data2/OpenSelfSup-gaia/workdirs/moco"

model_space_path = 'hubs/flops.json'
model_sampling_rules = dict(
    type='sequential',
    rules=[
        dict(
            type='parallel',
            rules=[
                 dict(func_str='lambda x: x[\'overhead.flops\'] >=5000000000 and x[\'overhead.flops\']<6000000000'),

            ]
        ),

        dict(func_str='lambda x: x[\'data.input_shape\'] == 224'),

        
        # sample
        dict(
            type='sample',
            operation='random',
            value=100, 
            mode='number',
        ),

        # merge all groups if more than one
        dict(type='merge'),
    ]
)
