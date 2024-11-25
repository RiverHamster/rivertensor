import random
import numpy as np
import pytensor as pt

random.seed(0)
np.random.seed(0)

def T_matmul(N, M, K):
    a = np.random.randn(N, M)
    b = np.random.randn(M, K)
    c = a @ b
    c_pt = pt.matmul(pt.from_numpy(a), pt.from_numpy(b)).numpy()
    assert np.max(np.abs(c - c_pt)) < 2e-3

def T_fc_grad_w(N, C, K):
    a = np.random.randn(N, C)
    b = np.random.randn(N, K)
    c = a.T @ b
    c_pt = pt.zeros([C, K])
    pt.fc_update_grad_w(pt.from_numpy(a), pt.from_numpy(b), 1.0, c_pt)
    c_pt = c_pt.numpy()
    assert np.max(np.abs(c - c_pt)) < 2e-3

def test_matmul():
    for i in range(50):
        N = random.randint(1, 32)
        M = random.randint(1, 32)
        K = random.randint(1, 32)
        T_matmul(N, M, K)

def test_fc_grad_w():
    for i in range(50):
        N = random.randint(1, 32)
        M = random.randint(1, 32)
        K = random.randint(1, 32)
        T_fc_grad_w(N, M, K)
