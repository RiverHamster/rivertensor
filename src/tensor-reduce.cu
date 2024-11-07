#include "tensor.h"
#include <cassert>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

namespace ten {
float sum(const Tensor &t) {
    assert(t.device() == TensorDevice::gpu);
    return thrust::reduce(thrust::device, t.data(), t.data() + t.size(), 0.0);
}

float max(const Tensor &t) {
    assert(t.device() == TensorDevice::gpu);
    return thrust::reduce(thrust::device, t.data(), t.data() + t.size(),
                          (float)(-INFINITY), thrust::maximum<float>());
}

float min(const Tensor &t) {
    assert(t.device() == TensorDevice::gpu);
    return thrust::reduce(thrust::device, t.data(), t.data() + t.size(),
                          (float)(INFINITY), thrust::minimum<float>());
}

template <ssize_t reduce_factor>
static void __global__ block_sum_ker(const float *__restrict__ in,
                                     float *__restrict__ out, ssize_t blk_size,
                                     ssize_t nblk) {
    ssize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float *__restrict__ in_p =
        in + blk_size * reduce_factor * blockIdx.y + idx;
    float *__restrict__ out_p = out + blk_size * blockIdx.y + idx;
    if (idx < blk_size) {
        float s = *in_p;
        // s[threadIdx.x] = *in_p;
        in_p += blk_size;
        #pragma unroll
        for (ssize_t i = 1; i < reduce_factor; ++i, in_p += blk_size) {
            if (blockIdx.y * reduce_factor + i < nblk)
                s += *in_p;
        }
        // printf("out_p[%d] = %f\n", (int)idx, s[threadIdx.x]);
        *out_p = s;
    }
}

static ssize_t cdiv(ssize_t a, ssize_t b) { return (a + b - 1) / b; }

Tensor sum_d0(const Tensor &t) {
    assert(t.device() == TensorDevice::gpu);
    assert(t.shape().size() > 0);
    constexpr ssize_t reduce_factor = 16;
    float *buf1, *buf2, *buf;
    ssize_t nblk = cdiv(t.shape()[0], reduce_factor);
    ssize_t nblk2 = cdiv(nblk, reduce_factor);
    cudaMalloc(&buf, (nblk + nblk2) * t.stride()[0] * sizeof(float));
    buf1 = buf;
    buf2 = buf1 + nblk * t.stride()[0];
    block_sum_ker<reduce_factor>
        <<<dim3(cdiv(t.stride()[0], BLOCK_SIZE),
                cdiv(t.shape()[0], reduce_factor)),
           BLOCK_SIZE>>>(t.data(), buf1, t.stride()[0], t.shape()[0]);
    for (; nblk != 1; nblk = cdiv(nblk, reduce_factor), std::swap(buf1, buf2)) {
        // cudaDeviceSynchronize();
        // printf("nblk = %zd\n", nblk);
        block_sum_ker<reduce_factor>
            <<<dim3(cdiv(t.stride()[0], BLOCK_SIZE), cdiv(nblk, reduce_factor)),
               BLOCK_SIZE>>>(buf1, buf2, t.stride()[0], nblk);
    }
    Tensor r(shape_t(t.shape().begin() + 1, t.shape().end()),
             TensorDevice::gpu);
    cudaMemcpy(r.data(), buf1, t.stride()[0] * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaFree(buf);
    return r;
}
} // namespace ten