import copy

_base_ = ['base.py','../model_samplers/specific.py']
_data_root_ = '/data/cls/'
# model settings
model = dict(
    type='DynamicBYOL',
    pretrained=None,
    base_momentum=0.996,
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
        type='DynamicNonLinearNeckV2',
        in_channels=2560,
        hid_channels=4096,
        out_channels=256,
        with_avg_pool=True),
    head=dict(type='LatentPredictHead',
              size_average=True,
              predictor=dict(type='NonLinearNeckV2',
                             in_channels=256, hid_channels=4096,
                             out_channels=256, with_avg_pool=False)))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    return_label=False,
    )
data_train_list = osp.join(_data_root_, 'imagenet/train_list.txt')
data_train_root = osp.join(_data_root_, 'imagenet/train')
dataset_type = 'BYOLDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, interpolation=3), 
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
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
        p=1.),
    dict(type='RandomAppliedTrans',
         transforms=[dict(type='Solarization')], p=0.),
]
# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
train_pipeline1 = copy.deepcopy(train_pipeline)
train_pipeline2 = copy.deepcopy(train_pipeline)
train_pipeline2[4]['p'] = 0.1 # gaussian blur
train_pipeline2[5]['p'] = 0.2 # solarization

data = dict(
    imgs_per_gpu=32,  # total 64*2*2=256
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline1=train_pipeline1,
        pipeline2=train_pipeline2,
        prefetch=prefetch,
    ))
# additional hooks
update_interval = 4  # interval for accumulate gradient
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1.),
]
optimizer = dict(type='LARS', lr=0.2, weight_decay=0.0000015, momentum=0.9,
                 paramwise_options={
                    '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
                    'bias': dict(weight_decay=0., lars_exclude=True)})
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=2,
    warmup_ratio=0.0001, # cannot be 0
    warmup_by_epoch=True)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 100
