#include "tensor.h"
#include <cassert>

namespace ten {
#define DEF_MAP_KERNEL(name, op)                                               \
    __global__ static void name##_kernel(                                      \
        intptr_t n, const float *__restrict__ in, float *__restrict__ out) {   \
        for (intptr_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;        \
             i += blockDim.x * gridDim.x)                                      \
            op;                                                                \
    }
#define DEF_MAP_OPT(name, ker_name, op)                                        \
    DEF_MAP_KERNEL(ker_name, op)                                               \
    Tensor name(const Tensor &x) {                                             \
        assert(x.device() == TensorDevice::gpu);                               \
        Tensor out(x.shape(), x.device());                                     \
        ker_name##_kernel<<<(x.size() + BLOCK_SIZE - 1) / BLOCK_SIZE,          \
                            BLOCK_SIZE>>>(x.size(), x.data(), out.data());     \
        return out;                                                            \
    }
#define DEF_BINARY_KERNEL(name, arg1, arg2, op)                                \
    __global__ static void name##_kernel(                                      \
        intptr_t n, const float *__restrict__ arg1,                            \
        const float *__restrict__ arg2, float *__restrict__ out) {             \
        for (intptr_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;        \
             i += blockDim.x * gridDim.x)                                      \
            op;                                                                \
    }

#define DEF_BINARY_OPT(name, ker_name, arg1, arg2, op)                         \
    DEF_BINARY_KERNEL(ker_name, arg1, arg2, op)                                \
    Tensor name(const Tensor &arg1, const Tensor &arg2) {                      \
        assert(arg1.device() == TensorDevice::gpu);                            \
        assert(arg2.device() == TensorDevice::gpu);                            \
        assert(arg1.shape() == arg2.shape());                                  \
        Tensor out(arg1.shape(), arg1.device());                               \
        ker_name##_kernel<<<(arg1.size() + BLOCK_SIZE - 1) / BLOCK_SIZE,       \
                            BLOCK_SIZE>>>(arg1.size(), arg1.data(),            \
                                          arg2.data(), out.data());            \
        return out;                                                            \
    }

#define DEF_FUN_OPT(f) DEF_MAP_OPT(f, f, out[i] = ::f(in[i]))

DEF_MAP_OPT(relu, relu, out[i] = in[i] < 0.0 ? 0.0 : in[i])
DEF_MAP_OPT(sigmoid, sigmoid, out[i] = 1.0 / (1.0 + ::exp(-in[i])))
DEF_MAP_OPT(operator-, neg, out[i] = -in[i])
DEF_BINARY_OPT(reluGrad, reluGrad, x, grad, out[i] = x[i] < 0.0 ? 0.0 : grad[i])
DEF_BINARY_OPT(sigmoidGrad, sigmoidGrad, y, grad,
               out[i] = grad[i] * y[i] * (1.0 - y[i]))
DEF_FUN_OPT(exp)
DEF_FUN_OPT(log)
DEF_FUN_OPT(log2)
DEF_FUN_OPT(sqrt)
// DEF_MAP_OPT(exp, exp, out[i] = ::exp(in[i]))
// DEF_MAP_OPT(log, log, out[i] = ::exp(in[i]))
// DEF_MAP_OPT(sqrt, sqrt, out[i] = ::sqrt(in[i]))
DEF_MAP_OPT(square, square, out[i] = in[i] * in[i])
DEF_BINARY_OPT(operator+, pointwise_plus, x, y, out[i] = x[i] + y[i])
DEF_BINARY_OPT(operator-, pointwise_minus, x, y, out[i] = x[i] - y[i])
DEF_BINARY_OPT(operator*, pointwise_mult, x, y, out[i] = x[i] * y[i])
DEF_BINARY_OPT(operator/, pointwise_div, x, y, out[i] = x[i] / y[i])
}