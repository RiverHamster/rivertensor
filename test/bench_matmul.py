import rivertensor.base as rt
import time

N = 2048
M = 2048
K = 2048

A = rt.randn([N, K])
B = rt.randn([K, M])

st = time.time()
niter = 1000
for i in range(niter):
    C = rt.matmul(A, B)
C = C.copy_to(rt.TensorDevice.cpu)
et = time.time()

print(f"{(et - st) * 1e3 / niter} ms/iter, {2 * N * M * K * niter / (et - st) / 1e9} GFLOPs")
