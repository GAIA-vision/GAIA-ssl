import copy
import torch
from torchvision.transforms import Compose
from PIL import Image
from openselfsup.datasets.registry import DATASETS
from openselfsup.datasets.base import BaseDataset
from openselfsup.datasets.utils import to_numpy
from openselfsup.datasets.registry import DATASETS, PIPELINES
from openselfsup.datasets.builder import build_datasource
from openselfsup.utils import print_log, build_from_cfg

@DATASETS.register_module
class DenseRelationDataset(BaseDataset):
    """
    Dataset for our dense relation
    """

    def __init__(self, data_source, pipeline,same_pipeline,aug_pipeline,prefetch=False):
        img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.data_source = build_datasource(data_source)
        data_source['return_label'] = False

        pipeline_encoder = [build_from_cfg(p, PIPELINES) for p in pipeline]
        self.pipeline_encoder = Compose(pipeline_encoder)

        same_pipeline = [build_from_cfg(p, PIPELINES) for p in same_pipeline]
        self.same_pipeline = Compose(same_pipeline)

        aug_pipeline = [build_from_cfg(p, PIPELINES) for p in aug_pipeline]
        self.aug_pipeline = Compose(aug_pipeline)

        to_tensor_pipeline = [dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)]
        to_tensor_pipeline = [build_from_cfg(p, PIPELINES) for p in to_tensor_pipeline]
        self.to_tensor_pipeline = Compose(to_tensor_pipeline)
        self.prefetch = prefetch

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        img1 = self.pipeline_encoder(img)
        img2 = self.pipeline_encoder(img)
        img3 = self.same_pipeline(img)
        img4 = self.to_tensor_pipeline(copy.deepcopy(img3))
        img3 = self.aug_pipeline(img3)
        #img4 = self.aug_pipeline(img4) 这个地方感觉用weak是更make sense的。
        if self.prefetch:
            img1 = torch.from_numpy(to_numpy(img1))
            img2 = torch.from_numpy(to_numpy(img2))
            img3 = torch.from_numpy(to_numpy(img3))
            img4 = torch.from_numpy(to_numpy(img4))
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0), img3.unsqueeze(0), img4.unsqueeze(0)), dim=0)
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented
