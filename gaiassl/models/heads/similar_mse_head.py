# 3rd party lib
import torch
import torch.nn as nn

# mm lib
from openselfsup.models import HEADS


@HEADS.register_module
class SimilarHeadMse(nn.Module):
    """Head for contrastive learning.
    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self):
        super(SimilarHeadMse, self).__init__()
        self.criterion = nn.MSELoss(reduction='sum')


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
        labels = torch.zeros((N, K+1), dtype=torch.float).cuda()
        labels -= 1.0
        labels[:,0] = 1.0
        losses = dict()
        #logits = (logits+1)/2 # 如果用BCE的话，需要它们的数值[0,1]之间，
        # 突然有个问题，对于feature 是正交比较好，还是完全反向比较好？
        # 这倒是可以考虑下，如果是正交比较好，那只需要把现在的logits取个relu，然后算BCE就好了。
        losses['loss'] = self.criterion(logits, labels)/N
        return losses