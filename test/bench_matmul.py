import pytensor as pt
import time

N = 2048
M = 2048
K = 2048

A = pt.randn([N, K])
B = pt.randn([K, M])

st = time.time()
niter = 1000
for i in range(niter):
    C = pt.matmul(A, B)
C = C.copy_to(pt.TensorDevice.cpu)
et = time.time()

print(f"{(et - st) * 1e3 / niter} ms/iter, {2 * N * M * K * niter / (et - st) / 1e9} GFLOPs")
