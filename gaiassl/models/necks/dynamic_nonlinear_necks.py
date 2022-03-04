# starndard lib
import pdb
from packaging import version

# 3rd party lib
import torch
import torch.nn as nn

# mm lib
from mmcv.cnn import kaiming_init, normal_init
from openselfsup.models import NECKS, build_norm_layer

# gaia lib
from gaiavision.core import DynamicMixin, DynamicLinear


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
class DynamicNonLinearNeckV0(nn.Module, DynamicMixin):
    """The non-linear neck in ODC, fc-bn-relu-dropout-fc-relu.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 sync_bn=False,
                 with_avg_pool=True):
        super().__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if version.parse(torch.__version__) < version.parse("1.4.0"):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.fc0 = DynamicLinear(in_channels, hid_channels)
        # TODO: replace with dynamic bn
        if sync_bn:
            _, self.bn0 = build_norm_layer(
                dict(type='SyncBN', momentum=0.001, affine=False),
                hid_channels)
        else:
            self.bn0 = nn.BatchNorm1d(
                hid_channels, momentum=0.001, affine=False)

        self.fc1 = nn.Linear(hid_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()
        self.sync_bn = sync_bn

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        if self.sync_bn:
            x = self._forward_syncbn(self.bn0, x)
        else:
            x = self.bn0(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.relu(x)
        return [x]


@NECKS.register_module
class DynamicNonLinearNeckV1(nn.Module, DynamicMixin):
    """The non-linear neck in MoCo v2: fc-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super().__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            DynamicLinear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]


@NECKS.register_module
class DynamicNonLinearNeckV2(nn.Module, DynamicMixin):
    """The non-linear neck in byol: fc-bn-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super().__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            DynamicLinear(in_channels, hid_channels),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1, "Got: {}".format(len(x))
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]



@NECKS.register_module
class DynamicPatchNonLinearNeckV1(nn.Module, DynamicMixin):
    """The non-linear neck in Detco for patch
    """

    def __init__(self,
                 patch_num,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super().__init__()
        self.patch_num = patch_num
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            DynamicLinear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        #pdb.set_trace()
        assert len(x) == 1
        x = x[0] # 这个是因为backbone的输出是可能包含多个stage的原因
        if self.with_avg_pool:
            x = self.avgpool(x) # [N*P,C,H,W] -> [N*P,C]
        x = x.view(x.size(0)//self.patch_num,-1) # [N*P,C] -> [N, P*C]
        return [self.mlp(x)]

@NECKS.register_module
class DynamicPatchNonLinearNeckV1_multistage(nn.Module, DynamicMixin):
    """The non-linear neck in Detco for patch
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 patch_num=1, # patch_num=1 就是DynamicNonLinearNeckV1_multistage
                 with_avg_pool=True):
        super().__init__()
        self.patch_num = patch_num
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.ModuleList()
        for each_in_channels in in_channels:
            self.mlp.append(
                    nn.Sequential(
                        DynamicLinear(each_in_channels, hid_channels), nn.ReLU(inplace=True),
                        nn.Linear(hid_channels, out_channels)
                )
            )
    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        #pdb.set_trace()
        
        assert len(x) == len(self.mlp)
        if self.with_avg_pool:
            x = [self.avgpool(each) for each in x] # [N*P,C,H,W] -> [N*P,C]
        x = [each.view(each.size(0)//self.patch_num, -1) for each in x]
        for i in range(len(x)):
            x[i] = self.mlp[i](x[i])
        return x