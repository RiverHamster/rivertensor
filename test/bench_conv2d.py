import pytensor as pt
import time

N = 8
C = 128
K = 128
H = 48
W = 48

im = pt.randn([N, C, H, W])
ker = pt.randn([9, C, K])
out = pt.zeros([N, K, H, W])

niter = 1000

st = time.time()
for i in range(niter):
    pt.conv2d_3x3(im, ker, out)
out = out.numpy()
ed = time.time()

print(f"{(ed - st) * 1e3 / niter} ms/iter, {N * C * K * H * W * 9 * niter / (ed - st) / 1e9} GFLOPs")
