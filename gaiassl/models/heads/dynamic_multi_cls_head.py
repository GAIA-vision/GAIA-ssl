import torch.nn as nn
import pdb

from openselfsup.models.utils import accuracy, build_norm_layer, MultiPooling
from openselfsup.models import HEADS

from ..utils import CustomMultiPooling


@HEADS.register_module
class CustomMultiClsHead(nn.Module):
    """Multiple classifier heads.
    """

    def __init__(self,
                 pool_type='adaptive',
                 in_indices=(0, ),
                 backbone='i224-s5', # input_scale: 224, stages: 5
                 with_last_layer_unpool=False,
                 feat_channels=[64, 256, 512, 1024, 2048],
                 feat_last_unpool=2048*7*7,
                 norm_cfg=dict(type='BN'),
                 num_classes=1000):
        super().__init__()
        assert norm_cfg['type'] in ['BN', 'SyncBN', 'GN', 'null']

        self.with_last_layer_unpool = with_last_layer_unpool
        self.with_norm = norm_cfg['type'] != 'null'

        self.feat_channels = feat_channels
        self.feat_last_unpool = feat_last_unpool

        self.criterion = nn.CrossEntropyLoss()

        self.multi_pooling = CustomMultiPooling(pool_type, in_indices, backbone)

        if self.with_norm:
            self.norms = nn.ModuleList([
                build_norm_layer(norm_cfg, self.feat_channels[l])[1]
                for l in in_indices
            ])

        self.fcs = nn.ModuleList([
            nn.Linear(
                self.multi_pooling.POOL_SIZES[backbone][l]**2*self.feat_channels[l],
                num_classes)
            for l in in_indices
        ])
        if with_last_layer_unpool:
            self.fcs.append(
                nn.Linear(self.feat_last_unpool, num_classes))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,
                            (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        if self.with_last_layer_unpool:
            last_x = x[-1]
        x = self.multi_pooling(x)
        if self.with_norm:
            x = [n(xx) for n, xx in zip(self.norms, x)]
        if self.with_last_layer_unpool:
            x.append(last_x)
        x = [xx.view(xx.size(0), -1) for xx in x]
        x = [fc(xx) for fc, xx in zip(self.fcs, x)]
        return x

    def loss(self, cls_score, labels):
        losses = dict()
        for i, s in enumerate(cls_score):
            # keys must contain "loss"
            losses['loss.{}'.format(i + 1)] = self.criterion(s, labels)
            losses['acc.{}'.format(i + 1)] = accuracy(s, labels)
        return losses
