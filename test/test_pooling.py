import rivertensor.base as rt
import torch
import torch.nn as nn
import numpy as np
import random

def T_pooling(N, C, H, W):
    assert H >= 2 and W >= 2
    im = np.random.randn(N, C, H, W)
    P = nn.MaxPool2d(2)
    pool = P(torch.from_numpy(im)).numpy()
    pool_rt = rt.maxpool2d_2x2(rt.from_numpy(im))
    pool_rt = pool_rt.numpy()
    assert np.max(np.abs(pool - pool_rt)) <= 1e-3

def T_pooling_d(N, C, H, W):
    assert H >= 2 and W >= 2
    im = np.random.randn(N, C, H, W)
    coeff = np.random.randn(N, C, H // 2, W // 2)
    P = nn.MaxPool2d(2)
    im_torch = torch.from_numpy(im)
    im_torch.requires_grad = True
    pool = P(im_torch)
    # pool_rt = rt.maxpool2d_2x2(rt.from_numpy(im))
    loss = torch.inner(pool.ravel(), torch.from_numpy(coeff).ravel())
    loss.backward()
    dx = im_torch.grad.numpy()
    dx_rt = rt.maxpool2d_2x2_grad(rt.from_numpy(im), rt.from_numpy(coeff))
    dx_rt = dx_rt.numpy()
    print(dx)
    print(dx_rt)
    assert np.max(np.abs(dx_rt - dx)) < 1e-3

def test_pooling():
    for i in range(100):
        N = random.randint(1, 10)
        C = random.randint(1, 10)
        H = random.randint(2, 20)
        W = random.randint(2, 20)
        T_pooling(N, C, H, W)

def test_pooling_grad():
    for i in range(100):
        N = random.randint(1, 10)
        C = random.randint(1, 10)
        H = random.randint(2, 20)
        W = random.randint(2, 20)
        T_pooling_d(N, C, H, W)

