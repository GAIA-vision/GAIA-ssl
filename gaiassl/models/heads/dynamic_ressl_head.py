# 3rd party lib
import torch
import torch.nn as nn
import torch.nn.functional as F

# mm lib
from openselfsup.models import HEADS


@HEADS.register_module
class DynamicResslHead(nn.Module):
    """Head for contrastive learning.
    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self, temperature=0.07):
        super(DynamicResslHead, self).__init__()
        self.temperature = temperature

    def forward(self, logits_q, logits_k):
        """Forward head.
        Args:
            logits_q (Tensor): Nxk 
            logits_k (Tensor): Nxk 
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = dict()
        # -plogq 算cross entropy, 有个问题一直没想清楚，-qlogp 和 -plogq 有本质区别嘛？
        losses['loss_contra'] = - torch.sum( F.softmax(logits_k.detach() / self.temperature, dim=1) \
                                             * F.log_softmax(logits_q / 0.1, dim=1), dim=1).mean()
        return losses