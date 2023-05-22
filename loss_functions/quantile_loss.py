from typing import Optional
import torch
from torch import Tensor

import ipdb
from .loss_utils import agg_loss


def quantile_loss(output, target, masks, alpha):
    single_loss = masks * quantile_noreduction_loss(output, target, alpha)
    loss = torch.mean(torch.sum(single_loss, dim=1) / torch.sum(masks, dim=1))

    return loss

#def quantile_noreduction_loss(output, target,alpha):
    #return ((output - target) * (output >= target) * alpha + (target - output) * (output < target) * (1 - alpha))
def quantile_noreduction_loss(output,alpha):
    return ((output ) * (output >= 0) * alpha + (- output) * (output < 0) * (1 - alpha))



class QuantileLoss(torch.nn.Module):
    reduction: str
    def __init__(self, q, ignore_index: int = -100, reduction: str = 'mean') -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.q = q

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        #dim = input.dim()
        logit, t = input['logit'], input['out']
        assert logit.dim() == 2, f"Expected 2 dimensions (got {logit.dim()})"

        offset = 1 - logit.min()
        output = torch.max((logit + offset) * (1-target), 1, keepdim=True)[0] - offset
        loss_in = quantile_noreduction_loss(-output - t[:, 0], self.q)

        offset = -1 - logit.max()
        output = torch.min((logit + offset) * target, 1, keepdim=True)[0] - offset
        loss_out = quantile_noreduction_loss(output - t[:, 1], self.q)

        return agg_loss(loss_in + loss_out, self.reduction)
