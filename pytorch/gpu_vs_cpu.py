# -*- coding: utf-8 -*-
"""
Criado em Mon Apr  5 21:13:38 2021

@author: Jasmine Moreira

conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
https://medium.com/analytics-vidhya/4-steps-to-install-anaconda-and-pytorch-onwindows-10-5c9cb0c80dfe

"""
import torch
import time 

t = 1000
dim = []
cpu = []
gpu = []

for n in (1,5,10,20,40,80,160,320,640,1280,2560,5120,10240,20480):

    start = time.time()
    b = torch.ones(n,n)
    for _ in range(t):
        b += b  
    c = time.time()-start
    
    start = time.time()
    b = torch.ones(n,n).cuda()
    for _ in range(t):
        b += b  
    g = time.time()-start
    
    dim.append(n)
    cpu.append(c)
    gpu.append(g)
    
    
import matplotlib.pyplot as plt
plt.scatter(dim,cpu, c='b', marker='x', label='cpu')
plt.scatter(dim,gpu, c='r', marker='s', label='gpu')
plt.legend(loc='upper left')
plt.xscale('log')
plt.yscale('log')
plt.show()
