# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:59:09 2022

@author: Alexander Davydov
"""

# Create experiment on MNIST

#import fixed_point as fp
import os
import trainNEMON
import torchvision
import torchvision.transforms as transforms
import torch
from data import mnist_loaders, cifar_loaders, mnist_transform, cifar_transform


trainLoader, testLoader = mnist_loaders(train_batch_size=100, test_batch_size=1000)

torch.manual_seed(37)
save_model = True

#trainNEMON.train(train_loader=trainLoader, 
#                 test_loader=testLoader, 
#                 model=trainNEMON.NEMON_MM_Net(in_dim=784, out_dim=100, m=0.0), 
#                 epochs=60, train_eps = 0.1, test_eps = 0.1, log=True)
#model=trainNEMON.NEMON_MM_Net(in_dim=784, out_dim=100, m=0.5)
model = trainNEMON.FeedForwardNN(in_dim=784, out_dim=25)
#model=trainNEMON.ibp_monDEQ(in_dim=784, out_dim=100, m=0.5)


print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Num of trainable parameters:", pytorch_total_params)
trainNEMON.train(train_loader=trainLoader, 
                 test_loader=testLoader, 
                 model=model, 
                 epochs=40, train_eps = 0.1, test_eps = 0.1, log=False)

if save_model:
    torch.save(model.state_dict(), 'ff_robust5.pt')