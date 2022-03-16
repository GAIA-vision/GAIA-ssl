# encoder_q

extra_subnet_num = 1
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
                ),
                dict(
                    name='MIN',
                    **MIN,
                )
            ]
        ),
        # random model samplers
        dict(
            type='repeat',
            times=extra_subnet_num,
            model_sampler=dict(
                type='composite',
                model_samplers=[
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

