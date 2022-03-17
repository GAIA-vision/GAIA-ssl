# standard lib
import pdb

# 3rd parth lib
import torch
import torch.nn as nn

# mm lib
from openselfsup.utils import print_log
from openselfsup.models import builder, MODELS

# local lib
from .base import BaseSSLearner


@MODELS.register_module
class DynamicBYOL(BaseSSLearner):
    """DynamicBYOL.

    Implementation based on "Bootstrap Your Own Latent: A New Approach to
    Self-Supervised Learning (https://arxiv.org/abs/2006.07733)".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        base_momentum (float): The base momentum coefficient for the target network.
            Default: 0.996.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 base_momentum=0.996,
                 same_arch=False,
                 **kwargs):
        super().__init__()
        self.online_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.target_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.backbone = self.online_net[0]
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

        self.base_momentum = base_momentum
        self.momentum = base_momentum
        self.same_arch = same_arch

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.online_net[0].init_weights(pretrained=pretrained) # backbone
        self.online_net[1].init_weights(init_linear='kaiming') # projection
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)
        # init the predictor in the head
        self.head.init_weights()

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the target network."""
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

    @torch.no_grad()
    def momentum_update(self):
        self._momentum_update()

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()
        # compute query features
        proj_online_v1 = self.online_net(img_v1)[0]
        proj_online_v2 = self.online_net(img_v2)[0]
        with torch.no_grad():
            proj_target_v1 = self.target_net(img_v1)[0].clone().detach()
            proj_target_v2 = self.target_net(img_v2)[0].clone().detach()

        loss = self.head(proj_online_v1, proj_target_v2)['loss'] + \
               self.head(proj_online_v2, proj_target_v1)['loss']
        return dict(loss=loss)

    def forward_test(self, img, **kwargs):
        pass

    def forward_get_embedding(self, img, extract_from='online_net', **kwargs):
        
        # TO DO: the label return need to be change
        label = kwargs.get('label', None)
        if img.dim() == 5:
            img = img[:, 0, ...].contiguous()

        with torch.no_grad():
            
            if extract_from == 'online_net':
                if label is not None:
                    # tensor [N, D], tensor[N]
                    return self.online_net(img)[0],label
                return self.online_net(img)[0]
            else:
                if label is not None:
                    return self.target_net(img)[0],label
                return self.target(img)[0]

    def forward(self, img, mode='train', **kwargs):
        #pdb.set_trace()
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        elif mode == 'get_embedding':
            return self.forward_get_embedding(img, **kwargs) 
        else:
            raise Exception("No such mode: {}".format(mode))

    # TODO: deal with neck in encoder
    def manipulate_online_net(self, arch_meta):
        self.online_net[0].manipulate_arch(arch_meta)

    def manipulate_target_net(self, arch_meta):
        if self.same_arch:
            state = self.online_net[0].state()
            self.target_net[0].manipulate_arch(state)
        else:
            self.target_net[0].manipulate_arch(arch_meta)

    def manipulate_head(self, arch_meta):
        raise NotImplementedError
