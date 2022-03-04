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
class DifferentPipeline(BaseDataset):
    """Dataset for ReSSL, encoder_k and encoder_q's input are processed differently
    """

    def __init__(self, data_source, pipeline_encoder_q, pipeline_encoder_k, prefetch=False):

        self.data_source = build_datasource(data_source)
        data_source['return_label'] = False
        pipeline_encoder_q = [build_from_cfg(p, PIPELINES) for p in pipeline_encoder_q]
        pipeline_encoder_k = [build_from_cfg(p, PIPELINES) for p in pipeline_encoder_k]
        self.pipeline_encoder_q = Compose(pipeline_encoder_q)
        self.pipeline_encoder_k = Compose(pipeline_encoder_k)
        self.prefetch = prefetch

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        img1 = self.pipeline_encoder_q(img)
        img2 = self.pipeline_encoder_k(img)
        if self.prefetch:
            img1 = torch.from_numpy(to_numpy(img1))
            img2 = torch.from_numpy(to_numpy(img2))
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented
