import random
import numpy as np
import rivertensor as rt

random.seed(0)
np.random.seed(0)

def T_matmul(N, M, K):
    a = np.random.randn(N, M)
    b = np.random.randn(M, K)
    c = a @ b
    c_rt = rt.matmul(rt.from_numpy(a), rt.from_numpy(b)).numpy()
    assert np.max(np.abs(c - c_rt)) < 2e-3

def T_fc_grad_w(N, C, K):
    a = np.random.randn(N, C)
    b = np.random.randn(N, K)
    c = a.T @ b
    c_rt = rt.zeros([C, K])
    rt.fc_update_grad_w(rt.from_numpy(a), rt.from_numpy(b), 1.0, c_rt)
    c_rt = c_rt.numpy()
    assert np.max(np.abs(c - c_rt)) < 2e-3

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
