import pdb
import torch
import torch.nn as nn

from openselfsup.models import HEADS

class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0, open_smooth=True):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 
        self.smoothing = smoothing
        self.open_smooth = open_smooth

    def forward(self, x, target, smooth_length):
        #pdb.set_trace()
        if self.open_smooth is False:
            logprobs = torch.nn.functional.log_softmax(x[:,:-smooth_length], dim=-1)
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            return nll_loss.mean()
            
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        
        smooth_loss = -logprobs[:, -smooth_length:].mean(dim=1) # 这不是包含了 label那一项吗？
        # -qlogp 相当于label那项的q是1, 其他都是0.1。 并不是全部加起来是1，
        # https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
        # 这里面的头两个实现倒是加起来都是1
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

@HEADS.register_module
class LabelSmoothHead(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self, smoothing=0.1, temperature=0.2, open_smooth=True):
        super(LabelSmoothHead, self).__init__()
        self.criterion = LabelSmoothing(smoothing, open_smooth)
        self.temperature = temperature
        self.open = open_smooth

    def forward(self, pos, neg, neg_smooth):
        """Forward head.

        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.
            neg_smooth (Tensor): NxHW negative similarity with label smooth
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        #pdb.set_trace()
        N = pos.size(0)
        smooth_length = neg_smooth.size(1)
        logits = torch.cat((pos, neg, neg_smooth), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        losses = dict()
        losses['loss'] = self.criterion(logits, labels, smooth_length)
        return losses
