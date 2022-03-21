import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

import math
import itertools

class IBPLinear(nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        bias=True,
        ibp=False
        ):
        super().__init__()
        self.in_features = in_dim
        self.out_features = out_dim
        self.M = nn.Linear(in_dim, out_dim, bias=bias)
        self.ibp = ibp

    def forward(
        self,
        *z):
        #print(len(z))
        assert(len(z) == 3 if self.ibp else 1)
        if not self.ibp:
            return (self.M(z[0]),)
        else:
            z0 = self.M(z[0])

            M_abs = torch.abs(self.M.weight)

            mu = (z[1] + z[2])/2
            r = (z[1] - z[2])/2
            
            mu = mu @ self.M.weight.T + self.M.bias
            r = r @ M_abs.T

            return (z0, mu + r, mu - r)

class IBPConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True, ibp=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.M = nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        self.stride = stride
        self.padding = padding
        self.ibp = ibp

    def forward(
        self, 
        *z):
        assert(len(z) == 3 if self.ibp else 1)
        if not self.ibp:
            return (F.conv2d(z[0], self.M.weight, bias=self.M.bias, stride=self.stride, padding=self.padding),)
        else:
            z0 = F.conv2d(z[0], self.M.weight, bias=self.M.bias, stride=self.stride, padding=self.padding)

            M_abs = torch.abs(self.M.weight)

            mu = (z[1] + z[2])/2
            r = (z[1] - z[2])/2

            mu = F.conv2d(mu, self.M.weight, bias=self.M.bias, stride=self.stride, padding=self.padding)
            r = F.conv2d(r, M_abs, bias=None, stride=self.stride, padding=self.padding)

            return (z0, mu + r, mu - r)

