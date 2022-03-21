import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from basic_layers import IBPLinear

class MM_NEMON(nn.Module):

    def __init__(self, in_dim, out_dim, m=0.05, cond=10):
        super().__init__()
        self.U = IBPLinear(in_dim, out_dim, bias=True, ibp=True) # input matrix
        self.M = nn.Linear(out_dim, out_dim, bias=False) # matrix parametrization variable
        self.m = m # contraction factor
        self.eta = nn.Parameter(torch.zeros(out_dim)) # diagonal weights
        self.a = nn.Parameter(torch.zeros(out_dim)) # negative diagonal
        self.cond = cond # condition number on diagonal weights
        self.cond_min = 1.0/self.cond
        self.cond_scale = 1 - self.cond_min
        
    def x_shape(self, n_batch):
        return ((n_batch, self.U.in_features),)*3

    def z_shape(self, n_batch):
        return ((n_batch, self.M.in_features),)*3

    def forward(self, x, *z):
        # Todo, construct positive and negative parts of B and make two copies of bias
        a = self.bias(*x) 
        b = self.multiply(*z)
        return [ai + bi for ai, bi in zip(a, b)]
        #return (self.B(x) + self.multiply(*z)[0],)

    def bias(self, x):
        # reapeat bias twice horizontally
        return self.U(*x)

    def _multiply(self, M, *z, transpose=False):
        
        one = torch.ones(self.M.weight.shape[0], dtype=self.M.weight.dtype, device=self.M.weight.device)
        
        #Lambda = 1/(self.M.weight.shape[0]-1)*(torch.abs(self.M.weight) - torch.diag(torch.diag(torch.abs(self.M.weight)))) @ one
        #Lambda.to(device = self.M.weight.device)
        #offdiag = self.M.weight - torch.diag(torch.diag(self.M.weight))
        #offdiag.to(device = self.M.weight.device)
        #T = F.relu(torch.abs(offdiag) - Lambda @ one.T)*torch.sign(offdiag)
        #T.to(device = self.M.weight.device)
        #T += torch.diag(torch.diag(self.M.weight)) - torch.diag(Lambda)
        
        #rowSums = torch.diag(torch.abs(self.M.weight) @ one).to(device = self.M.weight.device)
        rowSums = torch.diag(torch.abs(self.M.weight) @ one).to(device = self.M.weight.device)
        #diagweights = self.cond_scale*F.sigmoid(self.eta) + self.cond_min
        diagweights = torch.exp(self.eta)
        diagweights.to(device = self.M.weight.device)
        diagweightsinverse = torch.reciprocal(diagweights).to(device = self.M.weight.device)
        negdiag = torch.diagflat(torch.square(self.a)).to(device = self.M.weight.device)
        #transformedA = torch.diag(diagweightsinverse) @ self.M.weight @ torch.diag(diagweights)
        transformedA = torch.diagflat(diagweightsinverse) @ self.M.weight @ torch.diagflat(diagweights)
        transformedA.to(device = self.M.weight.device)
        A = torch.diag(one)*self.m + transformedA - rowSums - negdiag
        A.to(device = self.M.weight.device)
        AnonMzr = torch.minimum(A, torch.zeros_like(A)) - torch.diagflat(torch.diag(torch.minimum(A, torch.zeros_like(A))))
        AnonMzr.to(device = self.M.weight.device)
        AMzr = A - AnonMzr
        AMzr.to(device = self.M.weight.device)
        
        if transpose:
            z0 = z[0] @ A
            
            zover = z[1] @ AMzr + z[2] @ AnonMzr
            zunder = z[1] @ AnonMzr + z[2] @ AMzr
        else:
            z0 = z[0] @ A.T

            zover = z[1] @ AMzr.T + z[2] @ AnonMzr.T
            zunder = z[1] @ AnonMzr.T + z[2] @ AMzr.T
		
        out = (z0, zover, zunder)
        return out 

    def multiply(self, *z):
        return self._multiply(self.M.weight, *z, transpose=False)
    
    def multiply_transpose(self, *z):
        return self._multiply(self.M.weight, *z, transpose=True)

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
        rowSums = torch.diag(torch.abs(self.M.weight) @ one).to(device = self.M.weight.device)
        #diagweights = self.cond_scale*F.sigmoid(self.eta) + self.cond_min
        diagweights = torch.exp(self.eta)
        diagweights.to(device = self.M.weight.device)
        diagweightsinverse = torch.reciprocal(diagweights).to(device = self.M.weight.device)
        negdiag = torch.diagflat(torch.square(self.a)).to(device = self.M.weight.device)
        #transformedA = torch.diag(diagweightsinverse) @ self.M.weight @ torch.diag(diagweights)
        transformedA = torch.diagflat(diagweightsinverse) @ self.M.weight @ torch.diagflat(diagweights)
        transformedA.to(device = self.M.weight.device)
        A = torch.diagflat(one)*self.m + transformedA - rowSums - negdiag
        
        
        #one = torch.ones(self.M.weight.shape[0], dtype=self.M.weight.dtype, device=self.M.weight.device)
        #rowSums = torch.diag(torch.abs(self.M.weight) @ one).to(device = self.M.weight.device)
        #diagweights = self.cond_scale*F.sigmoid(self.eta) + self.cond_min
        #diagweights.to(device = self.M.weight.device)
        #diagweightsinverse = torch.reciprocal(diagweights).to(device = self.M.weight.device)
        #negdiag = torch.diag(torch.square(self.a)).to(device = self.M.weight.device)
        #transformedA = torch.diag(diagweightsinverse) @ self.M.weight @ torch.diag(diagweights)
        #transformedA.to(device = self.M.weight.device)
        #A = torch.diag(one)*self.m + transformedA - rowSums - negdiag
        #A.to(device = self.M.weight.device)
        alpha = 1/(1 - torch.min(torch.diag(A)))
        return alpha
        

class NEMONReLU(nn.Module):
    def forward(self, *z):
        return tuple(F.relu(z_) for z_ in z)
        #return tuple(F.leaky_relu(z_,negative_slope=0.1) for z_ in z)

    def derivative(self, *z):
        return tuple((z_ > 0).type_as(z[0]) for z_ in z)
        #return tuple(0.9*(z_ > 0).type_as(z[0]) + 0.1*torch.ones_like(z_) for z_ in z)