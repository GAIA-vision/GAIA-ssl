# 3rd party lib
import torch
import torch.nn as nn

# mm lib
from openselfsup.models import HEADS


@HEADS.register_module
class SimilarCosHead(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self):
        super(SimilarCosHead, self).__init__()

    def forward(self, pos, neg):
        """Forward head.

        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N = pos.size(0)
        K = neg.size(1)
        logits = torch.cat((pos, neg), dim=1)
        #logits /= self.temperature
        labels = torch.zeros((N, K+1), dtype=torch.long).cuda()
        labels[:,0] = 1
        losses = dict()
        logits[labels==1] *= -1
        logits *= 2 
        losses['loss'] = torch.mean(logits) 
        return losses
