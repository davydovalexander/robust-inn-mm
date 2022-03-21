# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:50:36 2022

@author: Alexander Davydov
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import sys
import os
import MM_NEMON
import fixed_point
from basic_layers import IBPLinear
from logger import Logger

from ibp_loss import *

from data import DATA_SHAPE

def cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor

def get_optim(model, lr):
    print('Using Adam optimizer.')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer 

def get_lr_lambda(anneal_at, anneal_factor):
    def f(epoch):
        fac = 1
        for ep in anneal_at:
            if epoch > ep:
                fac *= anneal_factor
        return fac
    return f


def train(train_loader, test_loader, model, epochs, train_eps=0.2, test_eps=0.1, log=True): 
    lr = 5e-4
    optimizer = get_optim(model, lr)

    lr_scheduler = None
    #if lr_mode == 'step':
    #    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda(step, anneal_factor))
    #elif lr_mode != 'constant':
    #    raise Exception('lr mode one of constant, step')
    if log:
        loggr = Logger(printstr=["epoch: {}. ce_loss: {:.2f}, ibp_loss: {:.2f}, acc: {}, ibp_acc: {:.2f}%, eps: {:.2f}", "epoch", "ce_loss", "ibp_loss", "acc", "ibp_acc", "eps"],
                       dir_name='MON_Test_eps01')
    
    model = cuda(model)

    scatter_mat = compute_sa(n_class=10)
    running_eps = 0.0
    train_eps /= 0.3081
    test_eps /= 0.3081
    kappa = 0.0
    train_kappa = 0.75
    for epoch in range(0, 1 + epochs):
        model.train()
        start = time.time()
        if epoch >= 10 and epoch < 20:
            running_eps += train_eps/10
            kappa += train_kappa/10
        if epoch == 30:
            lr *= 0.2
            optimizer = get_optim(model, lr)
        for batch_idx, batch in enumerate(train_loader):
            #eps = get_eps(epoch, batch_idx, len(train_loader), args)
            
            # Write brief linear epsilon ramp-up and kappa ramp-up
            

            data, target = cuda(batch[0]), cuda(batch[1])
            optimizer.zero_grad()
            if epoch < 10:
                z = model(data, eps=0.0)
                zhat = z[0]
                preds = model.Wout(zhat)
                ibp_preds = None
                #print(preds)
                #print(preds.size())
                #print(target.size())
                ce_loss = F.cross_entropy(preds, target)
                ce_loss.backward()
                ibp_loss = 0
            else:
                z = model(data, eps=running_eps)
                zhat = z[0]
                preds = model.Wout(zhat)
                ibp_preds = compute_ibp_elide_z(model.Wout, z[1], z[2], target, scatter_mat, n_class=10)
                ce_loss = F.cross_entropy(preds, target)
                ibp_loss = F.cross_entropy(ibp_preds, target)
                total_loss = (1-kappa)*ce_loss + kappa*ibp_loss
                total_loss.backward()
            
            optimizer.step()
        incorrect = 0
        ibp_incorrect = 0
        test_loss = 0
        ibp_test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                data, target = cuda(batch[0]), cuda(batch[1])
                z = model(data, eps=test_eps)
                zhat = z[0]
                #summ = 0
                #for i in range(len(z[1])):
                #    summ += torch.norm(z[1][i] - z[2][i])
                #print(summ)
                preds = model.Wout(zhat)
                ibp_preds = compute_ibp_elide_z(model.Wout, z[1], z[2], target, scatter_mat, n_class=10)
                ce_loss = F.cross_entropy(preds, target)
                ibp_loss = F.cross_entropy(ibp_preds, target)
                test_loss += ce_loss
                ibp_test_loss += ibp_loss
                incorrect += preds.float().argmax(1).ne(target.data).sum()
                ibp_incorrect += ibp_preds.float().argmax(1).ne(target.data).sum()
            #test_loss /= len(test_loader.dataset)
            #ibp_test_loss /= len(test_loader.dataset)
            nTotal = len(test_loader.dataset)
            err = 100. * incorrect.float() / float(nTotal)
            ibp_err = 100. * ibp_incorrect.float() / float(nTotal)
        #print()
        if log:
            log_dict = {
                        "epoch": epoch + 1,
                        "ce_loss": test_loss,
                        "ibp_loss": ibp_test_loss,
                        "acc": 100 - err,
                        "ibp_acc": 100 - ibp_err,
                        "eps": running_eps*0.3081
                        }
            loggr.log(log_dict, model, "acc")
        print('epoch: %d, loss: %.3f, ibp_loss: %5f, error: %5f, ibp_error: %5f, eps: %5f' %(epoch + 1, test_loss, ibp_test_loss, err, ibp_err, running_eps*0.3081))
        #tot_iter += 1


        #if lr_mode == 'step':
        #    lr_scheduler.step()

        print("Tot train time: {}".format(time.time() - start))

        #val(test_loader, model, log_writer, epoch, tot_iter)

        #if epoch % args.ckpt_every == 0:
        #    save_ckpt(model, optimizer, epoch, tot_iter, lr_scheduler, log_writer, args, is_latest=False) 
        #save_ckpt(model, optimizer, epoch, tot_iter, lr_scheduler, log_writer, args, is_latest=True)
        sys.stdout.flush()

class NEMON_MM_Net(nn.Module):

    def __init__(self, in_dim=784, out_dim=100, m=0.1, **kwargs):
        super().__init__()
        linear_module = MM_NEMON.MM_NEMON(in_dim, out_dim, m=m)
        nonlin_module = MM_NEMON.NEMONReLU()
        self.nemon = fixed_point.AverageIteration(linear_module, nonlin_module, limiting_alpha=0.5, max_alpha=0.5, tol=1e-3, max_iter=50)
        self.Wout = nn.Linear(out_dim, 10, bias=True)
        #self.D = nn.Linear(in_dim, 10, bias=False)

    def forward(self, x, eps=0.0):
        x = x.view(x.shape[0], -1)
        maxdata = torch.max(x)
        mindata = torch.min(x)
        x = (x, torch.minimum(x + eps, torch.ones_like(x)*maxdata), torch.maximum(x - eps, torch.ones_like(x)*mindata))
        #print(len(x))
        
        z = self.nemon(x)
        return z # + self.D(x)

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim=784, out_dim=40, **kwargs):
        super().__init__()
        self.W1 = IBPLinear(in_dim, 100, bias=True, ibp=True)
        self.W2 = IBPLinear(100, 75, bias=True, ibp=True)
        self.W3 = IBPLinear(75, 50, bias=True, ibp=True)
        self.W4 = IBPLinear(50, 40, bias=True, ibp=True)
        self.W5 = IBPLinear(40, out_dim, bias=True, ibp=True)
        self.Wout = nn.Linear(out_dim, 10, bias=True)
        #self.D = nn.Linear(in_dim, 10, bias=False)

    def forward(self, x, eps=0.0):
        x = x.view(x.shape[0], -1)
        maxdata = torch.max(x)
        mindata = torch.min(x)
        x = (x, torch.minimum(x + eps, torch.ones_like(x)*maxdata), torch.maximum(x - eps, torch.ones_like(x)*mindata))
        #print(len(x))
        z = self.W1(*x)
        z = tuple(F.relu(zi) for zi in z)
        z = self.W2(*z)
        z = tuple(F.relu(zi) for zi in z)
        z = self.W3(*z)
        z = tuple(F.relu(zi) for zi in z)
        z = self.W4(*z)
        z = tuple(F.relu(zi) for zi in z)
        z = self.W5(*z)
        z = tuple(F.relu(zi) for zi in z)
        return z # + self.D(x)

