import rivertensor.base as rt
import numpy as np
import scipy
import random

np.random.seed(0)
random.seed(0)

def T_c2d(N, C, H, W, K, output=False):
    # image = np.arange(1, N * C * H * W + 1).reshape(N, C, H, W).astype('float32')
    # fil = np.arange(1, 9 * C * K + 1).reshape(9, C, K).astype('float32')
    image = np.random.randn(N, C, H, W).astype('float32')
    fil = np.random.randn(9, C, K).astype('float32')
    if output:
        print("Image\n", image)
        print("Filter\n", fil)
    # calculate with rivertensor
    res = rt.conv2d_3x3(rt.from_numpy(image), rt.from_numpy(fil)).numpy()
    print(res.shape)

    # calculate with scipy
    fil = fil.reshape([3, 3, C, K])
    res_sp = np.zeros([N, K, H, W])
    for n in range(N):
        for c in range(C):
            for k in range(K):
                res_sp[n][k] += scipy.signal.correlate2d(image[n][c], fil[:, :, c, k], mode='full', boundary='fill')[2:, 2:]
                # print(f"correlate {image[n][c]} {fil[:, :, c, k]}")
    if output:
        print("res\n", res)
        print("res_sp\n", res_sp)
    assert np.max(np.abs(res - res_sp)) < 1e-2

def T_c2d_dx(N, C, H, W, K, output=False):
    print(f"T_c2d_dx {C} {H} {W} {K}")
    x = rt.randn([N, C, H, W])
    if output: 
        print(f"x: {x.numpy()}")
    ker = rt.randn([9, C, K])
    if output:
        print(f"ker: {ker.numpy()}")
    eps = 1e-3
    delta = rt.randn([N, C, H, W]) * eps
    coeff = rt.randn([N, K, H, W])
    if output:
        print(f"coeff: {coeff.numpy()}")
    y = rt.conv2d_3x3(x, ker)
    y1 = rt.conv2d_3x3(x + delta, ker)
    Y = rt.inner(y, coeff)
    Y1 = rt.inner(y1, coeff)
    dY = Y1 - Y
    gradX = rt.conv2d_3x3_grad_x(coeff, ker)
    if output:
        print(f"gradX: {gradX.numpy()}")
    dY_pred = rt.inner(delta, gradX)
    print(f"Y = {Y}, dY = {dY}, dY_pred = {dY_pred}")
    # compare with relative err
    assert np.abs(dY_pred - dY) / np.max(np.abs([Y, Y1, dY])) / eps < 1

def T_c2d_dk(N, C, H, W, K, output=False):
    print(f"T_c2d_dx {C} {H} {W} {K}")
    x = rt.randn([N, C, H, W])
    if output: 
        print(f"x: {x.numpy()}")
    ker = rt.randn([9, C, K])
    if output:
        print(f"ker: {ker.numpy()}")
    eps = 1e-3
    delta = rt.randn([9, C, K]) * eps
    coeff = rt.randn([N, K, H, W])
    if output:
        print(f"coeff: {coeff.numpy()}")
    y = rt.conv2d_3x3(x, ker)
    y1 = rt.conv2d_3x3(x, ker + delta)
    Y = rt.inner(y, coeff.newaxis)
    Y1 = rt.inner(y1, coeff.newaxis)
    dY = Y1 - Y
    gradK = rt.conv2d_3x3_grad_k(coeff, x)
    dY_pred = rt.inner(delta, gradK)
    print(f"Y = {Y}, dY = {dY}, dY_pred = {dY_pred}")
    # compare with relative err
    assert np.abs(dY_pred - dY) / np.max(np.abs([Y, Y1, dY / eps])) / eps < 0.2

def test_conv2d():
    for w in range(1, 25):
        T_c2d(1, 1, w, w, 1)
    for t in range(50):
        N = random.randint(1, 4)
        H = random.randint(1, 32)
        W = random.randint(1, 32)
        C = random.randint(1, 24)
        K = random.randint(1, 96)
        T_c2d(N, C, H, W, K)

def test_conv2d_grad_x():
    for w in range(1, 25):
        T_c2d_dx(1, 1, w, w, 1)
    for i in range(200):
        N = random.randint(1, 4)
        C = random.randint(1, 16)
        H = random.randint(1, 32)
        W = random.randint(1, 32)
        K = random.randint(1, 128)
        T_c2d_dx(N, C, H, W, K)

def test_conv2d_grad_k():
    for w in range(1, 25):
        T_c2d_dx(1, 1, w, w, 1)
    for i in range(200):
        N = random.randint(1, 4)
        C = random.randint(1, 16)
        H = random.randint(1, 32)
        W = random.randint(1, 32)
        K = random.randint(1, 128)
        T_c2d_dx(N, C, H, W, K)
