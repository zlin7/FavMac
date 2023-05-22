import ipdb
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=200, hdims=[100,100], nclass=8):
        super(MLP, self).__init__()
        last_dim = input_dim
        self.fcs = nn.ModuleList()
        for dim in hdims:
            self.fcs.append(nn.Linear(last_dim, dim, bias=True))
            last_dim = dim
        self.readout = nn.Linear(last_dim, nclass, bias=True)

    def get_readout_layer(self):
        return self.readout

    def get_embed_size(self):
        return self.readout.in_features

    def forward(self, x, embed_only=False, **kwargs):
        for fc in self.fcs:
            x = torch.relu(fc(x))
        if embed_only: return x
        return self.readout(x)


class MNISTCNN(nn.Module):
    def __init__(self, nclass, size=48, debug=False) -> None:
        super().__init__()
        self.convbn1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=0),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.convbn2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.convbn3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.fc = torch.nn.Linear(32 * 4 * 4, nclass)
        self.debug = debug

    def forward(self, input, embed_only=False, all_input=None):
        x = self.convbn1(input)
        x = self.convbn2(x)
        x = self.convbn3(x)
        return self.fc(x.flatten(start_dim=1))


if __name__ == '__main__':
    x = torch.rand([4,3,48,48])
    model = MNISTCNN(10)
    out = model(x)