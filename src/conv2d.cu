#include "tensor.h"
#include <cassert>
#include "cuda_runtime.h"
#include <cstdio>

namespace ten {
// we assume KBLK is a multiple of 4
static const ssize_t CBLK = 4, HBLK = 6, WBLK = 6, KBLK = 32;

__global__ void conv2d_3x3_ker(const float *in, const float *ker, float *out,
                               int C, int nblkC, int H, int nblkH, int W,
                               int nblkW, int K, int nblkK) {
    __shared__ float t_in[CBLK][HBLK + 2][WBLK + 2], t_ker[CBLK * 9][KBLK],
        col[HBLK * WBLK][CBLK * 9];

    int batch = blockIdx.x;
    int off_k = (batch % nblkK) * KBLK;
    batch /= nblkK;
    int off_c = (batch % nblkC) * CBLK;
    batch /= nblkC;
    int off_h = blockIdx.y * HBLK, off_w = blockIdx.z * WBLK;
    int tid = threadIdx.x;
    in += (ssize_t)batch * C * H * W;
    out += (ssize_t)batch * K * H * W;

    // load data
    // 256 threads, HARDCODED
    if (tid < 256) {
        int c = tid >> 6;
        int h = (tid >> 3) & 7;
        int w = tid & 7;
        if (off_c + c < C && off_h + h < H && off_w + w < W)
            t_in[c][h][w] =
                in[(off_c + c) * H * W + (off_h + h) * W + off_w + w];
        else
            t_in[c][h][w] = 0.0;
    }

    // load kernel
    // 288 threads
    if (tid < KBLK * 9) {
        int phase = tid / KBLK;
        int k = tid % KBLK;
        for (int c = 0; c < CBLK; ++c) {
            if (off_c + c < C && off_k + k < K)
                t_ker[c * 9 + phase][k] =
                    ker[phase * C * K + (off_c + c) * K + off_k + k];
            else
                t_ker[c * 9 + phase][k] = 0.0;
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
        // int k0 = tid % (KBLK / 4), pos = tid / (KBLK / 4);
        // int h = pos / WBLK, w = pos % WBLK;
        int k0 = tid / (HBLK * WBLK), pos = tid % (HBLK * WBLK);
        int h = pos / WBLK, w = pos % WBLK;
        for (int kstep = 0; kstep < 4; ++kstep) {
            int k = kstep * (KBLK / 4) + k0;
            float sum = 0;
            for (int m = 0; m < 9 * CBLK; ++m) {
                sum += col[pos][m] * t_ker[m][k];
            }
            // use atomicAdd, optimize to reduction
            if (k < K && off_h + h < H && off_w + w < W)
                atomicAdd(&out[(off_k + k) * H * W + (off_h + h) * W + off_w + w], sum);
        }
    }
}

// (N, C, H, W), (9, C, K) -> (N, K, H, W)
void conv2d_3x3(const Tensor &t, const Tensor &ker, Tensor out) {
    assert(t.ndim() == 4);
    assert(out.ndim() == 4);
    assert(ker.ndim() == 3);
    unsigned N = t.shape()[0], C = t.shape()[1], H = t.shape()[2], W = t.shape()[3],
        K = ker.shape()[2];
    assert(ker.shape() == (shape_t{9, C, K}));
    assert(out.shape() == (shape_t{N, K, H, W}));
    // tiling: C = 4, H = 6, W = 6, K = 32
    unsigned nblkC = (C + CBLK - 1) / CBLK;
    unsigned nblkH = (H + HBLK - 1) / HBLK;
    unsigned nblkW = (W + WBLK - 1) / WBLK;
    unsigned nblkK = (K + KBLK - 1) / KBLK;
    dim3 grid{N * nblkC * nblkK, nblkH, nblkW};
    ssize_t block = 324;
    conv2d_3x3_ker<<<grid, block>>>(t.data(), ker.data(), out.data(), C, nblkC, H, nblkH, W, nblkW, K, nblkK);
}
} // namespace ten
