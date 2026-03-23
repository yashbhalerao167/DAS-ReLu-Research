# src/activations.py

import torch
import torch.nn as nn


class DASReLU(nn.Module):
    def __init__(self, beta=0.1):
        super(DASReLU, self).__init__()
        self.beta = beta

    def forward(self, x, epoch=None):
        return torch.where(x >= 0, x, self.beta * x)