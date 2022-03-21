# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:41:32 2022

@author: Alexander Davydov
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import MM_NEMON
import fixed_point
import splitting as sp

class NEMON_Inference(nn.Module):

    def __init__(self, in_dim, out_dim, m=0.05, cond=10):
        super().__init__()
        self.U = nn.Linear(in_dim, out_dim, bias=True) # input matrix
        self.M = nn.Linear(out_dim, out_dim, bias=False) # matrix parametrization variable
        self.m = m # contraction factor
        self.eta = nn.Parameter(torch.zeros(out_dim)) # diagonal weights
        self.a = nn.Parameter(torch.zeros(out_dim)) # negative diagonal
        self.cond = cond # condition number on diagonal weights
        self.cond_min = 1.0/self.cond
        self.cond_scale = 1 - self.cond_min
        
    def x_shape(self, n_batch):
        return (n_batch, self.U.in_features)

    def z_shape(self, n_batch):
        return ((n_batch, self.M.in_features),)

    def forward(self, x, *z):
        return (self.U(x) + self.multiply(*z)[0],)

    def bias(self, x):
        return (self.U(x),)

    def multiply(self, *z):
        #ATAz = self.A(z[0]) @ self.A.weight
        
        one = torch.ones(self.M.weight.shape[0], dtype=self.M.weight.dtype,
                      device=self.M.weight.device)
        rowSums = torch.diagflat(torch.abs(self.M.weight) @ one).to(device = self.M.weight.device)
        #print(self.d)
        diagweights = self.cond_scale*F.sigmoid(self.eta) + self.cond_min
        diagweights.to(device = self.M.weight.device)
        
        diagweightsinverse = torch.reciprocal(diagweights).to(device = self.M.weight.device)
        #diagweights = torch.exp(self.d).to(device = self.A.weight.device)
        #diagweightsinverse = torch.reciprocal(diagweights).to(device = self.A.weight.device)
        #print(torch.diag(diagweightsinverse))
        transformedA = torch.diagflat(diagweightsinverse) @ self.M.weight @ torch.diagflat(diagweights)
        transformedA.to(device = self.M.weight.device)
        negdiag = torch.diagflat(torch.square(self.a)).to(device = self.M.weight.device)
        z_out = self.m * z[0] + z[0] @ transformedA.T - z[0] @ rowSums - z[0] @ negdiag
        #- ATAz + self.B(z[0]) - z[0] @ self.B.weight
        return (z_out,)

    def multiply_transpose(self, *g):
        #ATAg = self.A(g[0]) @ self.A.weight
        one = torch.ones(self.M.weight.shape[0], dtype=self.M.weight.dtype,
                      device=self.M.weight.device)
        rowSums = torch.diagflat(torch.abs(self.M.weight) @ one).to(device = self.M.weight.device)
        #diagweights = torch.exp(self.d).to(device = self.A.weight.device)
        diagweights = self.cond_scale*F.sigmoid(self.eta) + self.cond_min
        diagweights.to(device = self.M.weight.device)
        
        diagweightsinverse = torch.reciprocal(diagweights).to(device = self.M.weight.device)
        transformedA = torch.diagflat(diagweightsinverse) @ self.M.weight @ torch.diagflat(diagweights)
        transformedA.to(device = self.M.weight.device)
        negdiag = torch.diagflat(torch.square(self.a)).to(device = self.M.weight.device)
        g_out = self.m * g[0] + g[0] @ transformedA - g[0] @ rowSums.T - g[0] @ negdiag
        return (g_out,)

    def get_max_optimal_alpha(self):
        one = torch.ones(self.M.weight.shape[0], dtype=self.M.weight.dtype, device=self.M.weight.device)
        
        #Lambda = 1/(self.M.weight.shape[0]-1)*(torch.abs(self.M.weight) - torch.diag(torch.diag(torch.abs(self.M.weight)))) @ one
        #Lambda.to(device = self.M.weight.device)
        #offdiag = self.M.weight - torch.diag(torch.diag(self.M.weight))
        #offdiag.to(device = self.M.weight.device)
        #T = F.relu(torch.abs(offdiag) - Lambda @ one.T)*torch.sign(offdiag)
        #T.to(device = self.M.weight.device)
        #T += torch.diag(torch.diag(self.M.weight)) - torch.diag(Lambda)
        
        #rowSums = torch.diag(torch.abs(self.M.weight) @ one).to(device = self.M.weight.device)
        rowSums = torch.diagflat(torch.abs(self.M.weight) @ one).to(device = self.M.weight.device)
        diagweights = self.cond_scale*F.sigmoid(self.eta) + self.cond_min
        diagweights.to(device = self.M.weight.device)
        diagweightsinverse = torch.reciprocal(diagweights).to(device = self.M.weight.device)
        negdiag = torch.diagflat(torch.square(self.a)).to(device = self.M.weight.device)
        transformedA = torch.diagflat(diagweightsinverse) @ self.M.weight @ torch.diagflat(diagweights)
        transformedA.to(device = self.M.weight.device)
        A = torch.diagflat(one)*self.m + transformedA - rowSums - negdiag
        
        alpha = 1/(1 - torch.min(torch.diag(A)))
        return alpha

class NEMON(nn.Module):

    def __init__(self, in_dim=784, out_dim=100, m=0.1, **kwargs):
        super().__init__()
        linear_module = NEMON_Inference(in_dim, out_dim, m=m)
        nonlin_module = MM_NEMON.NEMONReLU()
        self.nemon = sp.NEmonForwardStep(linear_module, nonlin_module, alpha=0.5, tol=1e-3, max_iter=50)
        self.Wout = nn.Linear(out_dim, 10, bias=True)
        #self.D = nn.Linear(in_dim, 10, bias=False)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        #print(len(x))
        z = self.nemon(x)
        return self.Wout(z[0]) # + self.D(x)
    


class FFInference(nn.Module):
    def __init__(self, in_dim=784, out_dim=40, **kwargs):
        super().__init__()
        self.W1 = nn.Linear(in_dim, 100, bias=True)
        self.W2 = nn.Linear(100, 75, bias=True)
        self.W3 = nn.Linear(75, 50, bias=True)
        self.W4 = nn.Linear(50, 40, bias=True)
        self.W5 = nn.Linear(40, out_dim, bias=True)
        self.Wout = nn.Linear(out_dim, 10, bias=True)
        #self.D = nn.Linear(in_dim, 10, bias=False)

    def forward(self, x, eps=0.0):
        x = x.view(x.shape[0], -1)
        z = self.W1(x)
        z = F.relu(z)
        z = self.W2(z)
        z = F.relu(z)
        z = self.W3(z)
        z = F.relu(z)
        z = self.W4(z)
        z = F.relu(z)
        z = self.W5(z)
        z = F.relu(z)
        return self.Wout(z) # + self.D(x)