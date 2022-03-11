R50 = {
    'arch.online_net.stem.width': 64,
    'arch.online_net.body.width': [64, 128, 256, 512],
    'arch.online_net.body.depth': [3, 4, 6, 3],
    'data.input_shape': 224,
}
R50_TN = {
    'arch.target_net.stem.width': 64,
    'arch.target_net.body.width': [64, 128, 256, 512],
    'arch.target_net.body.depth': [3, 4, 6, 3],
}
# config of model samplers
train_sampler = dict(
    type='concat',
    model_samplers=[
        dict(
            type='anchor',
            anchors=[
               dict(
                    name='R50',
                    **R50,
                    **R50_TN,
                ),
            ],
        ),
    ],
)