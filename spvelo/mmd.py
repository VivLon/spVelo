#!/usr/bin/env python
# encoding: utf-8
import torch
import torch.nn
import jax
import jax.numpy as jnp

min_var_est = 1e-8

import torch

def gaussian_kernel_gpu(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # Move tensors to GPU
    source = source.cuda()
    target = target.cuda()

    n_samples = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)
    
    # Efficiently compute L2 distance
    L2_distance = torch.cdist(total, total, p=2).pow(2)
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    
    # Compute the kernel values for all bandwidths and sum them up
    kernel_val = torch.zeros_like(L2_distance).cuda()
    for i in range(kernel_num):
        bandwidth_i = bandwidth * (kernel_mul ** i)
        kernel_val.add_(-L2_distance, alpha=1/bandwidth_i)
    
    kernel_val.exp_()  # Compute the exponential
    
    return kernel_val

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def calculate_mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, use_gpu=False):
    #new method of calculating mmd
    batch_size = int(source.size()[0])
    if use_gpu==False:
        kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)
    elif use_gpu==True:
        kernels = gaussian_kernel_gpu(source, target, kernel_mul, kernel_num, fix_sigma)
    XX = torch.mean(kernels[:batch_size, :batch_size])
    YY = torch.mean(kernels[batch_size:, batch_size:])
    XY = torch.mean(kernels[:batch_size, batch_size:])
    YX = torch.mean(kernels[batch_size:, :batch_size])
    loss = torch.mean(XX + YY - XY - YX)
    return loss
    
