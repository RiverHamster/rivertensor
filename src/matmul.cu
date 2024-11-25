#include "cublas-handle.h"
#include "macros.h"
#include "tensor.h"
#include <algorithm>
#include <cassert>
#include <cublas_v2.h>
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>

static const float f32_1 = 1.0f, f32_0 = 0.0f;

namespace ten {
void matmul_(const Tensor &A, const Tensor &B, Tensor C, bool transA,
             bool transB, float alpha, float beta) {
    assert(A.ndim() == 2);
    assert(B.ndim() == 2);
    assert(A.device() == TensorDevice::gpu);
    assert(B.device() == TensorDevice::gpu);
    ssize_t A0 = A.shape()[0], A1 = A.shape()[1], B0 = B.shape()[0],
            B1 = B.shape()[1];
    if (transA)
        std::swap(A0, A1);
    if (transB)
        std::swap(B0, B1);
    assert(A1 == B0);
    assert(C.shape() == (shape_t{A0, B1}));

    auto handle = get_cublas_handle();
    blasChkerr(cublasSgemm_64(handle, transB ? CUBLAS_OP_T : CUBLAS_OP_N,
                              transA ? CUBLAS_OP_T : CUBLAS_OP_N, B1, A0, A1,
                              &alpha, B.data(), B.stride()[0], A.data(),
                              A.stride()[0], &beta, C.data(), C.stride()[0]));
}

Tensor matmul_t(const Tensor &A, const Tensor &B, bool transA, bool transB) {
    assert(A.ndim() == 2);
    assert(B.ndim() == 2);
    Tensor C({A.shape()[0], B.shape()[1]}, TensorDevice::gpu);
    matmul_(A, B, C, transA, transB, 1, 0);
    return C;
}

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
        blasChkerr(cublasSgemv_64(handle, CUBLAS_OP_N, b.shape()[1],
                                  b.shape()[0], &f32_1, b.data(), b.stride()[0],
                                  a.data(), 1, &f32_0, c.data(), 1));
        return c;
    } else if (b.ndim() == 1) {
        assert(a.shape()[1] == b.shape()[0]);
        Tensor c(shape_t{a.shape()[0]}, TensorDevice::gpu);
        blasChkerr(cublasSgemv_64(handle, CUBLAS_OP_T, a.shape()[1],
                                  a.shape()[0], &f32_1, a.data(), a.stride()[0],
                                  b.data(), 1, &f32_0, c.data(), 1));
        return c;
    } else {
        assert(a.shape()[1] == b.shape()[0]);
        Tensor c({a.shape()[0], b.shape()[1]}, TensorDevice::gpu);
        // cuBLAS use F-order
        matmul_(a, b, c, false, false, 1, 0);
        // blasChkerr(cublasSgemm_64(
        //     handle, CUBLAS_OP_N, CUBLAS_OP_N, b.shape()[1], a.shape()[0],
        //     a.shape()[1], &f32_1, b.data(), b.stride()[0], a.data(),
        //     a.stride()[0], &f32_0, c.data(), c.stride()[0]));
        return c;
    }
}

// x: [N, C] dy: [N, K] dw: [C, K]
void fc_update_grad_w(const Tensor &x, const Tensor &dy, float alpha,
                      Tensor dw) {
    assert(x.device() == TensorDevice::gpu);
    assert(dy.device() == TensorDevice::gpu);
    assert(x.ndim() == 2);
    assert(dy.ndim() == 2);
    ssize_t N = x.shape()[0], C = x.shape()[1], K = dy.shape()[1];
    assert(dy.shape()[0] == N);
    assert(dw.shape() == (shape_t{C, K}));

    // auto handle = get_cublas_handle();
    // blasChkerr(cublasSgemm_64(handle, CUBLAS_OP_N, CUBLAS_OP_T, K, C, N, &alpha,
    //                           dy.data(), dy.stride()[0], x.data(),
    //                           x.stride()[0], &f32_1, dw.data(),
    //                           dw.stride()[0]));
    matmul_(x, dy, dw, true, false, alpha, 1);
}

void outer_update(const Tensor &x, const Tensor &y, float alpha, Tensor t) {
    assert(x.device() == TensorDevice::gpu);
    assert(y.device() == TensorDevice::gpu);
    assert(x.ndim() == 1);
    assert(y.ndim() == 1);
    assert(t.shape()[0] == x.size());
    assert(t.shape()[1] == y.size());
    auto handle = get_cublas_handle();
    blasChkerr(cublasSger_64(handle, y.size(), x.size(), &alpha, y.data(), 1,
                             x.data(), 1, t.data(), t.stride()[0]));
}

Tensor outer(const Tensor &x, const Tensor &y) {
    assert(x.device() == TensorDevice::gpu);
    assert(y.device() == TensorDevice::gpu);
    assert(x.ndim() == 1);
    assert(y.ndim() == 1);
    Tensor t({x.size(), y.size()}, TensorDevice::gpu);
    auto handle = get_cublas_handle();
    blasChkerr(cublasSger_64(handle, y.size(), x.size(), &f32_1, y.data(), 1,
                             x.data(), 1, t.data(), t.stride()[0]));
    return t;
}

float inner(const Tensor &a, const Tensor &b) {
    assert(a.shape() == b.shape());
    return thrust::inner_product(thrust::device, a.data(), a.data() + a.size(),
                                 b.data(), 0.0f);
}
} // namespace ten
