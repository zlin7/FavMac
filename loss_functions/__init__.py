import torch
from .quantile_loss import QuantileLoss
def get_criterion(name):
    if not isinstance(name, str): return name
    if name == 'BCE': return torch.nn.BCEWithLogitsLoss
    if name == 'PinBall': return QuantileLoss
    raise ValueError()