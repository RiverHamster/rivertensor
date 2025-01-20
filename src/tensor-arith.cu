#include "macros.h"
#include "tensor.h"
#include <cassert>

namespace ten {
#define DEF_MAP_KERNEL(name, op)                                               \
    KERNEL name##_kernel(ssize_t n, const float *__restrict__ in,              \
                         float *__restrict__ out) {                            \
        ssize_t i = blockIdx.x * blockDim.x + threadIdx.x;                     \
        if (i < n)                                                             \
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
    KERNEL name##_kernel(ssize_t n, const float *__restrict__ arg1,            \
                         const float *__restrict__ arg2,                       \
                         float *__restrict__ out) {                            \
        ssize_t i = blockIdx.x * blockDim.x + threadIdx.x;                     \
        if (i < n)                                                             \
            op;                                                                \
    }

#define DEF_BINARY_KERNEL_EXT(name, arg1, arg2, op)                            \
    template <int xstep, int ystep>                                            \
    KERNEL name##_kernel(ssize_t n, const float *__restrict__ arg1,            \
                         const float *__restrict__ arg2,                       \
                         float *__restrict__ out) {                            \
        ssize_t i = blockIdx.x * blockDim.x + threadIdx.x;                     \
        if (i < n) {                                                           \
            ssize_t i1 = i * xstep, i2 = i * ystep;                            \
            op;                                                                \
        }                                                                      \
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

#define DEF_BINARY_OPT_EXT(name, ker_name, arg1, arg2, op)                     \
    DEF_BINARY_KERNEL_EXT(ker_name, arg1, arg2, op)                            \
    Tensor name(const Tensor &arg1, const Tensor &arg2) {                      \
        assert(arg1.device() == TensorDevice::gpu);                            \
        assert(arg2.device() == TensorDevice::gpu);                            \
        if (arg1.shape().empty() && arg2.shape().empty()) {                    \
            Tensor out(std::vector<ssize_t>{}, arg1.device());                 \
            ker_name##_kernel<0, 0>                                            \
                <<<(arg1.size() + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>( \
                    arg1.size(), arg1.data(), arg2.data(), out.data());        \
            return out;                                                        \
        }                                                                      \
        if (arg1.shape().empty()) {                                            \
            Tensor out(arg2.shape(), arg2.device());                           \
            ker_name##_kernel<0, 1>                                            \
                <<<(arg2.size() + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>( \
                    arg2.size(), arg1.data(), arg2.data(), out.data());        \
            return out;                                                        \
        }                                                                      \
        if (arg2.shape().empty()) {                                            \
            Tensor out(arg1.shape(), arg1.device());                           \
            ker_name##_kernel<1, 0>                                            \
                <<<(arg1.size() + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>( \
                    arg1.size(), arg1.data(), arg2.data(), out.data());        \
            return out;                                                        \
        }                                                                      \
        assert(arg1.shape() == arg2.shape());                                  \
                                                                               \
        Tensor out(arg1.shape(), arg1.device());                               \
        ker_name##_kernel<1, 1>                                                \
            <<<(arg1.size() + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(     \
                arg1.size(), arg1.data(), arg2.data(), out.data());            \
        return out;                                                            \
    }

#define DEF_FUN_OPT(f) DEF_MAP_OPT(f, f, out[i] = ::f(in[i]))

#define DEF_SCALAR_KERNEL(name, op)                                            \
    KERNEL name##_kernel(ssize_t n, const float *x, float y, float *out) {     \
        ssize_t i = blockIdx.x * blockDim.x + threadIdx.x;                     \
        if (i < n)                                                             \
            out[i] = x[i] op y;                                                \
    }

#define DEF_COMM_SCALAR_OPT(ker_name, op)                                      \
    DEF_SCALAR_KERNEL(ker_name, op)                                            \
    Tensor operator op(const Tensor &x, float y) {                             \
        assert(x.device() == TensorDevice::gpu);                               \
        Tensor out(x.shape(), x.device());                                     \
        ker_name##_kernel<<<(x.size() + BLOCK_SIZE - 1) / BLOCK_SIZE,          \
                            BLOCK_SIZE>>>(x.size(), x.data(), y, out.data());  \
        return out;                                                            \
    }                                                                          \
    Tensor operator op(float x, const Tensor &y) { return y op x; }

DEF_MAP_OPT(relu, relu, out[i] = in[i] < 0.0 ? 0.0 : in[i])
DEF_MAP_OPT(sigmoid, sigmoid, out[i] = 1.0 / (1.0 + ::exp(-in[i])))
DEF_MAP_OPT(operator-, neg, out[i] = -in[i])
DEF_BINARY_OPT(relu_grad, relu_grad, x, grad,
               out[i] = x[i] < 0.0 ? 0.0 : grad[i])
DEF_BINARY_OPT(sigmoid_grad, sigmoid_grad, y, grad,
               out[i] = grad[i] * y[i] * (1.0 - y[i]))
DEF_FUN_OPT(exp)
DEF_FUN_OPT(log)
DEF_FUN_OPT(log2)
DEF_FUN_OPT(sqrt)
DEF_MAP_OPT(square, square, out[i] = in[i] * in[i])
DEF_BINARY_OPT_EXT(operator+, pointwise_plus, x, y, out[i] = x[i1] + y[i2])
DEF_BINARY_OPT_EXT(operator-, pointwise_minus, x, y, out[i] = x[i1] - y[i2])
DEF_BINARY_OPT_EXT(operator*, pointwise_mult, x, y, out[i] = x[i1] * y[i2])
DEF_BINARY_OPT_EXT(operator/, pointwise_div, x, y, out[i] = x[i1] / y[i2])
DEF_COMM_SCALAR_OPT(scalar_add, +)
DEF_COMM_SCALAR_OPT(scalar_mult, *)
} // namespace ten
