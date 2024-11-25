// Common constructors of Tensor
// Only GPU version is supported here
#include "macros.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <random>

__global__ void fill_1_kernel(float *f, intptr_t n) {
    for (intptr_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        f[i] = 1.0f;
}

namespace ten {
Tensor zeros(const shape_t &shape) {
    Tensor t(shape, TensorDevice::gpu);
    cudaMemsetAsync(t.data(), 0, t.aligned_size() * sizeof(float));
    return t;
}
Tensor ones(const shape_t &shape) {
    Tensor t(shape, TensorDevice::gpu);
    fill_1_kernel<<<t.size() + BLOCK_SIZE - 1, BLOCK_SIZE>>>(t.data(),
                                                             t.aligned_size());
    return t;
}

static uint64_t seed_random() {
    return (uint64_t)std::random_device()() << 32 | std::random_device()();
}

Tensor rand(const shape_t &shape) {
    Tensor t(shape, TensorDevice::gpu);
    curandGenerator_t gen;
    randChkerr(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    randChkerr(curandSetPseudoRandomGeneratorSeed(gen, seed_random()));
    randChkerr(curandGenerateUniform(gen, t.data(), t.aligned_size()));
    cudaDeviceSynchronize();
    return t;
}

Tensor randn(const shape_t &shape) {
    Tensor t(shape, TensorDevice::gpu);
    curandGenerator_t gen;
    randChkerr(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    randChkerr(curandSetPseudoRandomGeneratorSeed(gen, seed_random()));
    randChkerr(
        curandGenerateNormal(gen, t.data(), t.aligned_size(), 0.0f, 1.0f));
    cudaDeviceSynchronize();
    return t;
}
} // namespace ten
