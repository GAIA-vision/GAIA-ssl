# starndard lib
import pdb
from packaging import version

# 3rd party lib
import torch
import torch.nn as nn

# mm lib
from openselfsup.models import NECKS
from mmcv.cnn import kaiming_init, normal_init

# gaia lib
from gaiavision.core import DynamicMixin, DynamicLinear, DynamicConv2d


def _init_weights(module, init_linear='normal', std=0.01, bias=0.):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.modules():
        if isinstance(m, (nn.Linear, DynamicLinear)):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


@NECKS.register_module
class DynamicDenseCLNeck(nn.Module,DynamicMixin):
    '''The non-linear neck in DenseCL.
        Single and dense in parallel: fc-relu-fc, conv-relu-conv
    '''
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_grid=None):
        super(DynamicDenseCLNeck, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            DynamicLinear(in_channels, hid_channels), nn.ReLU(inplace=True),
            DynamicLinear(hid_channels, out_channels))

        self.with_pool = num_grid != None
        if self.with_pool:
            self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid))
        self.mlp2 = nn.Sequential(
            DynamicConv2d(in_channels, hid_channels, 1), nn.ReLU(inplace=True),
            DynamicConv2d(hid_channels, out_channels, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        pdb.set_trace()
        #assert len(x) == 1
        x = x[0]

        avgpooled_x = self.avgpool(x) #[N,C]
        avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1)) #[N,C]

        if self.with_pool:
            x = self.pool(x) # sxs
        x = self.mlp2(x) # sxs: bxdxsxs [N,C,H,W]
        avgpooled_x2 = self.avgpool2(x) # 1x1: bxdx1x1 [N,C,1,1]
        x = x.view(x.size(0), x.size(1), -1) # bxdxs^2 [N,C,H*W] 
        avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1) # bxd [N,C]

        # 略微有点奇怪，仔细看了下DenseCL的源代码，这个avgpooled_x2根本没用到。。。
        # 原来是自己没有注意完全，对于dense的队列，竟然不是存储pixel level的feature，
        # 而是把feature做一个池化。。。感觉可以跑一版看一下，是用pooling的更好，还是
        # 不pooling的更好，讲道理这个很直觉的实现，作者应该是考虑过，难道pooling后更具有代表性？
        # pooling之后的不就相当于全局的嘛。。。。
        return [avgpooled_x, x, avgpooled_x2]