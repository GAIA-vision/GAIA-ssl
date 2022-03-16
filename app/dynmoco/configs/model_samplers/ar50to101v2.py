input_shape_cands = dict(
    key='data.input_shape', candidates=(192, 208, 224, 240, 256))
# encoder_q
stem_width_range = dict(
    key='arch.encoder_q.stem.width',
    start=32,
    end=64,
    step=16,
)
body_width_range = dict(
    key='arch.encoder_q.body.width',
    start=[48, 96, 192, 384],
    end=[80, 160, 320, 640],
    step=[16, 32, 64, 128],
    ascending=True,
)
body_depth_range = dict(
    key='arch.encoder_q.body.depth',
    start=[2, 2, 5, 2],
    end=[4, 6, 29, 4],
    step=[1, 2, 2, 1],
)

MAX_TN = {
    'arch.encoder_k.stem.width': stem_width_range['end'],
    'arch.encoder_k.body.width': body_width_range['end'],
    'arch.encoder_k.body.depth': body_depth_range['end'],
}
MAX = {
    'arch.encoder_q.stem.width': stem_width_range['end'],
    'arch.encoder_q.body.width': body_width_range['end'],
    'arch.encoder_q.body.depth': body_depth_range['end'],
}
MIN = {
    'arch.encoder_q.stem.width': stem_width_range['start'],
    'arch.encoder_q.body.width': body_width_range['start'],
    'arch.encoder_q.body.depth': body_depth_range['start'],
}
R50 = {
    'arch.encoder_q.stem.width': 64,
    'arch.encoder_q.body.width': [64, 128, 256, 512],
    'arch.encoder_q.body.depth': [3, 4, 6, 3],
}
R77 = {
    'arch.encoder_q.stem.width': 64,
    'arch.encoder_q.body.width': [64, 128, 256, 512],
    'arch.encoder_q.body.depth': [3, 4, 15, 3],
}
R101 = {
    'arch.encoder_q.stem.width': 64,
    'arch.encoder_q.body.width': [64, 128, 256, 512],
    'arch.encoder_q.body.depth': [3, 4, 23, 3],
}


# config of model samplers
train_sampler = dict(
    type='concat',
    model_samplers=[
        dict(
            type='anchor',
            anchors=[
                dict(
                    name='MAX',
                    **MAX,
                    **MAX_TN,
                ),
                dict(
                    name='MIN',
                    **MIN,
                    **MAX_TN,
                )
            ]
        ),
        # random model samplers
        dict(
            type='repeat',
            times=3,
            model_sampler=dict(
                type='composite',
                model_samplers=[
                    dict(
                        type='anchor',
                        anchors=[
                            dict(
                                name='MAX_TN',
                                **MAX,
                            ),
                        ]
                    ),
                    dict(
                        type='range',
                        **stem_width_range,
                    ),
                    dict(
                        type='range',
                        **body_width_range,
                    ),
                    dict(
                        type='range',
                        **body_depth_range,
                    ),
                ]
            )
        )
    ]
)


