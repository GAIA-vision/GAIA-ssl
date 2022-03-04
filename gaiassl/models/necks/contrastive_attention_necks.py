# starndard lib
import pdb
import math
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

# 还没加dropout, layer normalization, 残差连接( )
# 初始化得思考一下，应该要让初始时，attention是avgpooling的效果
# 那么验证attention写的有没有bug的一个方式也可以是，把attention的score
# 变成一个均值，看是否和原来结果保持一致。

# neck是否也要是momentum的，保持一致可不可以？ 
# 以及是否需要warmup一下，因为loss看起来比较奇怪，先升后降然后又上升。。。。
@NECKS.register_module
class ContrastiveAttentionNeck(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_heads=1,
                 atten_drop=0):
        super(ContrastiveAttentionNeck, self).__init__()
        self.object_query = nn.Parameter(torch.rand(num_heads, in_channels//num_heads, 1)) # [num_heads, C//num_heads, 1]
        # 选择什么样的初始化应该不可能是一个一个的尝试，而是根据先验信息选取才对吧
        self.qkv = nn.Linear(in_channels, 3*in_channels)# 这个用什么样的初始化是否有很大的影响？
        # self.attn_drop = nn.Dropout(atten_trop) # 这个待定，不能明确有它
        self.proj = nn.Sequential(nn.Linear(in_channels, hid_channels), nn.Linear(hid_channels, out_channels))
        self.num_heads = num_heads

        
    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        #assert len(x) == 1
        
        x = x[0] #[N,C,H,W]
        
        x = x.view(x.size(0), x.size(1), -1) # [N, C, H*W]
        x = x.permute(0,2,1) # [N, H*W, C] 是否contiguous？这个怎么准确判断？
        N, num, C = x.shape
        qkv = self.qkv(x) # [N, H*W, 3*C]
        qkv = qkv.reshape(N,num,3,self.num_heads, C//self.num_heads) # [N,H*W,3,head_num,C//head_num]
        qkv = qkv.permute(2,0,3,1,4) # [3, N, head_num, H*W, C//head_num]
        q, k, v = qkv[0], qkv[1], qkv[2] # q好像用不太到, [N, head_num, H*W, C//head_num]
        #pdb.set_trace()
        
        score = torch.matmul(k,self.object_query)/math.sqrt(q.size(-1)) # [N, head_num, H*W, 1]
        score = torch.nn.Softmax(dim=2)(score) # [N, head_num, H*W, 1]
        x = torch.matmul(score.transpose(2,3), v).contiguous() # [N, head_num, 1, C//head_num]
        x = x.view(x.size(0),-1)
        x = self.proj(x)
        return [x]