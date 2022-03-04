# standard lib
from collections.abc import Sequence
import math


class ScaleManipulator():
    def __init__(self, scale, base=224):
        self.scale = scale
        self.ratio = scale/224

    def __call__(self, dataset_cfg):
        pipelines = dataset_cfg.pipeline
        for i, p in enumerate(pipelines):
            if p['type'] == 'Resize':
                img_scale = p['size']
                new_scale = int(math.ceil(img_scale*self.ratio))
                p['size'] = new_scale
            elif p['type'] == 'CenterCrop':
                p['size'] = self.scale


def manipulate_dataset(dataset, manipulator):
    if hasattr(dataset, 'dataset'):
        dataset = getattr(dataset, 'dataset')
        manipulate_dataset(dataset, manipulator)
    elif hasattr(dataset, 'datasets'):
        for dataset in getattr(dataset, 'datasets'):
            manipulate_dataset(dataset, manipulator)
    else:
        manipulator(dataset)


