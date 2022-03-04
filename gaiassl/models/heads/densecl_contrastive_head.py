# 3rd party lib
import torch
import torch.nn as nn

# mm lib
from openselfsup.models import HEADS


@HEADS.register_module
class DenseCLContrastiveHead(nn.Module):
    """Head for contrastive learning.
    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self, temperature=0.1):
        super(DenseCLContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, pos, neg):
        """Forward head.
        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        losses = dict()
        losses['loss_contra'] = self.criterion(logits, labels)
        return losses

        