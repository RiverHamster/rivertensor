#include "macros.h"
#include "tensor.h"
#include <cassert>
#include "cublas-handle.h"
#include <cublas_v2.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>

// TODO: Check async errors?

static const float f32_1 = 1.0f, f32_0 = 0.0f;

namespace ten {
Tensor matmul(const Tensor &a, const Tensor &b) {
    assert(a.device() == TensorDevice::gpu);
    assert(b.device() == TensorDevice::gpu);
    assert(a.ndim() <= 2 && a.ndim() >= 1);
    assert(b.ndim() <= 2 && b.ndim() >= 1);
    assert(a.ndim() + b.ndim() >= 3);
    cublasHandle_t handle = get_cublas_handle();
    if (a.ndim() == 1) {
        assert(a.shape()[0] == b.shape()[0]);
        Tensor c(shape_t{b.shape()[1]}, TensorDevice::gpu);
        blasChkerr(cublasSgemv_64(
            handle,
            CUBLAS_OP_N,
            b.shape()[1],
            b.shape()[0],
            &f32_1,
            b.data(),
            b.stride()[0],
            a.data(),
            1,
            &f32_0,
            c.data(),
            1
        ));
        return c;
    }
    else if (b.ndim() == 1) {
        assert(a.shape()[1] == b.shape()[0]);
        Tensor c(shape_t{a.shape()[0]}, TensorDevice::gpu);
        blasChkerr(cublasSgemv_64(
            handle,
            CUBLAS_OP_T,
            a.shape()[1],
            a.shape()[0],
            &f32_1,
            a.data(),
            a.stride()[0],
            b.data(),
            1,
            &f32_0,
            c.data(),
            1
        ));
        return c;
    }
    else {
        assert(a.shape()[1] == b.shape()[0]);
        // TODO: make a global handle
        Tensor c({a.shape()[0], b.shape()[1]}, TensorDevice::gpu);
        // cuBLAS use F-order
        blasChkerr(cublasSgemm_64(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            b.shape()[1],
            a.shape()[0],
            a.shape()[1],
            &f32_1,
            b.data(),
            b.stride()[0],
            a.data(),
            a.stride()[0],
            &f32_0,
            c.data(),
            c.stride()[0]
        ));
        return c;
    }
}

void outer_update(const Tensor &x, const Tensor &y, float alpha, Tensor t) {
    assert(x.device() == TensorDevice::gpu);
    assert(y.device() == TensorDevice::gpu);
    assert(x.ndim() == 1);
    assert(y.ndim() == 1);
    assert(t.shape()[0] == x.size());
    assert(t.shape()[1] == y.size());
    auto handle = get_cublas_handle();
    blasChkerr(cublasSger_64(
        handle,
        y.size(),
        x.size(),
        &alpha,
        y.data(), 1,
        x.data(), 1,
        t.data(), t.stride()[0]
    ));
}

Tensor outer(const Tensor &x, const Tensor &y) {
    assert(x.device() == TensorDevice::gpu);
    assert(y.device() == TensorDevice::gpu);
    assert(x.ndim() == 1);
    assert(y.ndim() == 1);
    Tensor t({x.size(), y.size()}, TensorDevice::gpu);
    auto handle = get_cublas_handle();
    blasChkerr(cublasSger_64(
        handle,
        y.size(),
        x.size(),
        &f32_1,
        y.data(), 1,
        x.data(), 1,
        t.data(), t.stride()[0]
    ));
    return t;
}

float inner(const Tensor &a, const Tensor &b) {
    assert(a.shape() == b.shape());
    return thrust::inner_product(thrust::device, a.data(), a.data() + a.size(), b.data(), 0.0f);
}
} // namespace ten
