import pytensor as pt
import numpy as np
import scipy
import random

np.random.seed(0)
random.seed(0)

def T(N, C, H, W, K, output=False):
    # image = np.arange(1, N * C * H * W + 1).reshape(N, C, H, W).astype('float32')
    # fil = np.arange(1, 9 * C * K + 1).reshape(9, C, K).astype('float32')
    image = np.random.randn(N, C, H, W).astype('float32')
    fil = np.random.randn(9, C, K).astype('float32')
    if output:
        print("Image\n", image)
        print("Filter\n", fil)
    # calculate with pytensor
    res = pt.zeros([N, K, H, W])
    pt.conv2d_3x3(pt.from_numpy(image), pt.from_numpy(fil), res)
    res = res.numpy()
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

def test_conv2d():
    for w in range(1, 25):
        T(1, 1, w, w, 1)
    for t in range(50):
        N = random.randint(1, 4)
        H = random.randint(1, 32)
        W = random.randint(1, 32)
        C = random.randint(1, 24)
        K = random.randint(1, 96)
        T(N, C, H, W, K)
