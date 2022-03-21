# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 10:34:48 2022

@author: Alexander Davydov
"""

import torch
import torch.nn as nn
from torch.autograd import Function
import time
from functools import reduce

import numpy as np

class Meter(object):
    """Computes and stores the min, max, avg, and current values"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -float("inf")
        self.min = float("inf")

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)
        self.min = min(self.min, val)

class SplittingMethodStats(object):
    def __init__(self):
        self.fwd_iters = Meter()
        self.bkwd_iters = Meter()
        self.fwd_time = Meter()
        self.bkwd_time = Meter()

    def reset(self):
        self.fwd_iters.reset()
        self.fwd_time.reset()
        self.bkwd_iters.reset()
        self.bkwd_time.reset()

    def report(self):
        print('Fwd iters: {:.2f}\tFwd Time: {:.4f}\tBkwd Iters: {:.2f}\tBkwd Time: {:.4f}\n'.format(
                self.fwd_iters.avg, self.fwd_time.avg,
                self.bkwd_iters.avg, self.bkwd_time.avg))

class AverageIteration(nn.Module):

    def __init__(self, linear_module, nonlin_module, limiting_alpha=0.5, max_alpha=0.5, tol=1e-3, max_iter=75, verbose=False):
        super().__init__()
        self.linear_module = linear_module
        self.nonlin_module = nonlin_module
        self.max_alpha = max_alpha
        self.limiting_alpha = limiting_alpha
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.stats = SplittingMethodStats()
        self.save_abs_err = False
        self.norm_names = [] if not hasattr(self.linear_module, 'norm_names') else self.linear_module.norm_names

    def forward(self, *x):
        """ Forward pass of NEMON, find fixed-point with average iterations"""
        self.limiting_alpha = self.linear_module.get_max_optimal_alpha()
        #print(self.alpha)
        start = time.time()
        # Run the forward pass _without_ tracking gradients
        with torch.no_grad():
            #self.linear_module.prep_model()
            #print(x[0][0])
            z = tuple(torch.zeros(s, dtype=x[0][0].dtype, device=x[0][0].device)
                      for s in self.linear_module.z_shape(x[0][0].shape[0]))
            
            n = len(z)
            running_alpha = torch.tensor(self.max_alpha)
            bias = self.linear_module.bias(*x)
            
            err = 100.0
            it = 0
            errs = []
            
            while (err > self.tol and it < self.max_iter):
                zn = self.linear_module.multiply(*z)
                zn = tuple((zn[i] + bias[i]) for i in range(n))
                zn = self.nonlin_module(*zn)
                #print(type(zn[0]))
                #print(type(self.alpha))
                zn = tuple((1 - running_alpha) * z[i] + running_alpha * zn[i] for i in range(n))
                #itplus1 = it+1
                #phik = (1 - gamma**(2*itplus1 + 2))/(1 - gamma**2)
                #zn = tuple((1 - 1/phik)*zn[i] + 1/phik*z0[i] for i in range(n))
                #zn = tuple(m(z[i]) for i in range(n))
                
                if self.save_abs_err:
                    fn = self.nonlin_module(*self.linear_module(x, *zn))
                    err_new = sum((zn[i] - fn[i]).norm().item() / (zn[i].norm().item()) for i in range(n))
                    errs.append(err_new)
                else:
                    err_new = sum((zn[i] - z[i]).norm().item() / (1e-6 + zn[i].norm().item()) for i in range(n))
                ratio = torch.div(err_new,err)
                if torch.max(ratio) > 0.9:
                    running_alpha = torch.max(running_alpha/1.75, self.limiting_alpha)
                
                err = err_new

                z = zn
                it = it + 1

        #if self.verbose:
        #print("Forward: ", it, err, running_alpha)

        # Run the forward pass one more time, tracking gradients, then backward placeholder
        #self.linear_module.prep_model()
        self.limiting_alpha = running_alpha
        zn = self.linear_module(x, *z)
        zn = self.nonlin_module(*zn)
        #zn = tuple(m(z[i]) for i in range(n))
        zn = self.Backward.apply(self, *zn)
        self.stats.fwd_iters.update(it)
        self.stats.fwd_time.update(time.time() - start)
        self.errs = errs
        return zn

    class Backward(Function):
        @staticmethod
        def forward(ctx, splitter, *z):
            ctx.splitter = splitter
            ctx.save_for_backward(*z)
            return z

        @staticmethod
        # assume that prep_model has already been called with some associated forward pass
        def backward(ctx, *g):
            start = time.time()
            sp = ctx.splitter
            n = len(g)
            z = ctx.saved_tensors
            j = sp.nonlin_module.derivative(*z)
            I = [j[i] == 0 for i in range(n)]
            d = [(1 - j[i]) / j[i] for i in range(n)]
            v = tuple(j[i] * g[i] for i in range(n))
            u = tuple(torch.zeros(s, dtype=g[0][0].dtype, device=g[0][0].device)
                      for s in sp.linear_module.z_shape(g[0].shape[0]))
            #print(u[0].size())
            
            err = 1.0
            it = 0
            errs = []
            while (err > sp.tol and it < sp.max_iter):
                un = sp.linear_module.multiply_transpose(*u)
                #print(un[0].size())
                un = tuple((1 - sp.limiting_alpha) * u[i] + sp.limiting_alpha * un[i] for i in range(n))
                un = tuple((un[i] + sp.limiting_alpha * (1 + d[i]) * v[i]) / (1 + sp.limiting_alpha * d[i]) for i in range(n))
                for i in range(n):
                    un[i][I[i]] = v[i][I[i]]
                    
                    err = sum((un[i] - u[i]).norm().item() / (1e-6 + un[i].norm().item()) for i in range(n))
                    errs.append(err)
                    u = un
                    it = it + 1

            if sp.verbose:
                print("Backward: ", it, err)

            dg = sp.linear_module.multiply_transpose(*u)
            dg = tuple(g[i] + dg[i] for i in range(n))

            sp.stats.bkwd_iters.update(it)
            sp.stats.bkwd_time.update(time.time() - start)
            sp.errs = errs
            return (None,) + dg

    def get_norms(self):
        return self.linear_module.get_norms()
    
class MONForwardBackward(nn.Module):

    def __init__(self, linear_module, nonlin_module, alpha=1.0, tol=1e-3, max_iter=50, verbose=False):
        super().__init__()
        self.linear_module = linear_module
        self.nonlin_module = nonlin_module
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.stats = SplittingMethodStats()
        self.save_abs_err = False
        self.norm_names = [] if not hasattr(self.linear_module, 'norm_names') else self.linear_module.norm_names

    def forward(self, *x):
        """ Forward pass of the MON, find an equilibirum with forward-backward splitting"""

        start = time.time()
        # Run the forward pass _without_ tracking gradients
        with torch.no_grad():
            self.linear_module.prep_model()
            z = tuple(torch.zeros(s, dtype=x[0][0].dtype, device=x[0][0].device)
                      for s in self.linear_module.z_shape(x[0][0].shape[0]))
            n = len(z)
            bias = self.linear_module.bias(*x, update_bn=False)

            err = 1.0
            it = 0
            errs = []
            
            while (err > self.tol and it < self.max_iter):
                zn = self.linear_module.multiply(*z)
                zn = tuple((1 - self.alpha) * z[i] + self.alpha * (zn[i] + bias[i]) for i in range(n))
                zn = self.nonlin_module(*zn)
                if self.save_abs_err:
                    fn = self.nonlin_module(*self.linear_module(x, *zn, update_bn=False), alpha=1)
                    err = sum((zn[i] - fn[i]).norm().item() / (zn[i].norm().item()) for i in range(n))
                    errs.append(err)
                else:
                    err = sum((zn[i] - z[i]).norm().item() / (1e-6 + zn[i].norm().item()) for i in range(n))

                z = zn
                it = it + 1

        #if self.verbose:
        #print("Forward: ", it, err)

        # Run the forward pass one more time, tracking gradients, then backward placeholder
        self.linear_module.prep_model()
        zn = self.linear_module(x, *z, update_bn=True)
        zn = self.nonlin_module(*zn)
        zn = self.Backward.apply(self, *zn)
        self.stats.fwd_iters.update(it)
        self.stats.fwd_time.update(time.time() - start)
        self.errs = errs
        return zn
    
    class Backward(Function):
        @staticmethod
        def forward(ctx, splitter, *z):
            ctx.splitter = splitter
            ctx.save_for_backward(*z)
            return z

        @staticmethod
        # assume that prep_model has already been called with some associated forward pass
        def backward(ctx, *g):
            start = time.time()
            sp = ctx.splitter
            n = len(g)
            z = ctx.saved_tensors
            j = sp.nonlin_module.derivative(*z)
            I = [j[i] == 0 for i in range(n)]
            d = [(1 - j[i]) / j[i] for i in range(n)]
            v = tuple(j[i] * g[i] for i in range(n))
            u = tuple(torch.zeros(s, dtype=g[0][0].dtype, device=g[0][0].device)
                      for s in sp.linear_module.z_shape(g[0].shape[0]))
            #print(u[0].size())
            
            err = 1.0
            it = 0
            errs = []
            while (err > sp.tol and it < sp.max_iter):
                un = sp.linear_module.multiply_transpose(*u)
                #print(un[0].size())
                un = tuple((1 - sp.alpha) * u[i] + sp.alpha * un[i] for i in range(n))
                un = tuple((un[i] + sp.alpha * (1 + d[i]) * v[i]) / (1 + sp.alpha * d[i]) for i in range(n))
                for i in range(n):
                    un[i][I[i]] = v[i][I[i]]
                    
                    err = sum((un[i] - u[i]).norm().item() / (1e-6 + un[i].norm().item()) for i in range(n))
                    errs.append(err)
                    u = un
                    it = it + 1

            if sp.verbose:
                print("Backward: ", it, err)

            dg = sp.linear_module.multiply_transpose(*u)
            dg = tuple(g[i] + dg[i] for i in range(n))

            sp.stats.bkwd_iters.update(it)
            sp.stats.bkwd_time.update(time.time() - start)
            sp.errs = errs
            return (None,) + dg

    def get_norms(self):
        return self.linear_module.get_norms()