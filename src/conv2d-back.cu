#include "cuda_runtime.h"
#include "macros.h"
#include "tensor.h"
#include <cassert>
#include <cstdio>

namespace ten {
// NB: we interchanged the meaning of C and K, and C now refers to the
// dimension of the input of backward pass (y), to maintain code consistency
template <ssize_t CBLK, ssize_t HBLK, ssize_t WBLK, ssize_t KBLK>
KERNEL conv2d_3x3_dx_ker(const float *y, const float *ker, float *dx, int C,
                         int nblkC, int H, int nblkH, int W, int nblkW, int K,
                         int nblkK) {
    __shared__ float t_in[CBLK][HBLK + 2][WBLK + 2], t_ker[CBLK * 9][KBLK],
        col[HBLK * WBLK][CBLK * 9];

    int batch = blockIdx.x;
    int off_k = (batch % nblkK) * KBLK;
    batch /= nblkK;
    int off_c = (batch % nblkC) * CBLK;
    batch /= nblkC;
    int off_h = blockIdx.y * HBLK, off_w = blockIdx.z * WBLK;
    int tid = threadIdx.x;
    // BACK PASS: no batch size

    // load data
    // 256 threads
    if (tid < CBLK * (HBLK + 2) * (WBLK + 2)) {
        int c = tid / ((HBLK + 2) * (WBLK + 2));
        int h = (tid / (WBLK + 2)) % (HBLK + 2);
        int w = tid % (WBLK + 2);
        // BACK PASS: shift the convolution window
        if (off_c + c < C && off_h + h - 2 < H && off_w + w - 2 < W &&
            off_h + h - 2 >= 0 && off_w + w - 2 >= 0)
            t_in[c][h][w] =
                y[(off_c + c) * H * W + (off_h + h - 2) * W + off_w + w - 2];
        else
            t_in[c][h][w] = 0.0;
    }

    // load kernel
    // 288 threads
    if (tid < KBLK * 9) {
        int phase = tid / KBLK;
        int k = tid % KBLK;
        // BACK PASS: reverse the kernel, and exchange C, K
        // kernel is (9, K, C)
        // TODO: irregular access patterns
        for (int c = 0; c < CBLK; ++c) {
            if (off_c + c < C && off_k + k < K)
                t_ker[c * 9 + (8 - phase)][k] =
                    ker[phase * K * C + (off_k + k) * C + off_c + c];
            else
                t_ker[c * 9 + (8 - phase)][k] = 0.0;
        }
    }
    __syncthreads();

    // im2col transformation
    // 324 threads
    if (tid < HBLK * WBLK * 9) {
        int phase = tid % 9, phase_h = phase / 3, phase_w = phase % 3;
        int _quot = tid / 9;
        int h = _quot / WBLK, w = _quot % WBLK;
        for (int c = 0; c < CBLK; ++c) {
            col[_quot][c * 9 + phase] = t_in[c][h + phase_h][w + phase_w];
        }
    }

    __syncthreads();

    // matrix multiplication and write-back
    // 288 threads
    if (tid < HBLK * WBLK * (KBLK / 4)) {
        int k0 = tid / (HBLK * WBLK), pos = tid % (HBLK * WBLK);
        int h = pos / WBLK, w = pos % WBLK;
        for (int kstep = 0; kstep < 4; ++kstep) {
            int k = kstep * (KBLK / 4) + k0;
            float sum = 0;
            for (int m = 0; m < 9 * CBLK; ++m) {
                sum += col[pos][m] * t_ker[m][k];
            }
            // TODO: use atomicAdd, optimize to reduction when necessary
            if (off_k + k < K && off_h + h < H && off_w + w < W)
                atomicAdd(
                    &dx[(off_k + k) * H * W + (off_h + h) * W + off_w + w],
                    sum);//, printf("update [%d, %d, %d]: %f\n", k, h, w, sum);
        }
    }
}

void conv2d_3x3_grad_x(const Tensor &y, const Tensor &ker, Tensor dx) {
    constexpr int CBLK = 4, HBLK = 6, WBLK = 6, KBLK = 32;
    assert(y.ndim() == 4);
    assert(dx.ndim() == 4);
    assert(ker.ndim() == 3);
    unsigned N = y.shape()[0], K = y.shape()[1], H = y.shape()[2],
             W = y.shape()[3], C = ker.shape()[1];
    assert(ker.shape() == (shape_t{9, C, K}));
    assert(dx.shape() == (shape_t{N, C, H, W}));

    // BACK PASS: exchange C, K
    unsigned nblkC = (K + CBLK - 1) / CBLK;
    unsigned nblkH = (H + HBLK - 1) / HBLK;
    unsigned nblkW = (W + WBLK - 1) / WBLK;
    unsigned nblkK = (C + KBLK - 1) / KBLK;
    dim3 grid{N * nblkC * nblkK, nblkH, nblkW};
    ssize_t block = 324;
    cudaMemsetAsync(dx.data(), 0, sizeof(float) * dx.size());
    conv2d_3x3_dx_ker<CBLK, HBLK, WBLK, KBLK>
        <<<grid, block>>>(y.data(), ker.data(), dx.data(), K, nblkC, H, nblkH,
                          W, nblkW, C, nblkK);
    cudaDeviceSynchronize();
}
} // namespace ten
