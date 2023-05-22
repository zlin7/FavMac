import torch
from typing import Optional, Tuple, Union
import torch.nn.functional as F
from torch import Tensor
import ipdb

import numpy as np

LOG_EPSILON = 1e-5

def agg_loss(loss, reduction):
    if torch.isnan(loss).any(): ipdb.set_trace()
    if reduction == 'mean': return loss.mean()
    if reduction == 'sum': return loss.sum()
    return loss

def safe_binary_cross_entropy_with_probs(input_p, target):
    input_p = input_p.clip(LOG_EPSILON, 1 - LOG_EPSILON)
    return -(target * torch.log(input_p) + (1.0 - target) * torch.log(1.0 - input_p))

def safe_binary_cross_entropy_with_logits(input, target):
    return safe_binary_cross_entropy_with_probs(torch.sigmoid(input), target)

if __name__ == '__main__':
    torch.manual_seed(15)
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    target_onehot = F.one_hot(target, 5).float()
    p = torch.softmax(input, 1)

    l1 = torch.binary_cross_entropy_with_logits(input, target_onehot)
    l2 = safe_binary_cross_entropy_with_logits(input, target_onehot)
    assert torch.allclose(l1,l2)
