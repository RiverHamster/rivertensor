#include "macros.h"
#include "tensor.h"
#include <cassert>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <vector>

namespace ten {
void softmax(const Tensor &t, Tensor out) {
    assert(t.shape() == out.shape());
    assert(t.ndim() == 1 || t.ndim() == 2);
    assert(out.shape() == t.shape());
    ssize_t N = t.ndim() == 1 ? 1 : t.shape()[0], C = t.shape().back();

    for (ssize_t i = 0; i < N; ++i) {
        const float *L = t.data() + C * i, *R = t.data() + C * (i + 1);
        float mx = thrust::reduce(thrust::device, L, R, -INFINITY,
                                  thrust::maximum<float>());
        float exp_sum = thrust::transform_reduce(
            thrust::device, L, R,
            [=] __device__ __host__(float x) -> float {
                return std::exp(x - mx);
            },
            0.0, thrust::plus<float>());
        thrust::transform(thrust::device, L, R, out.data() + i * C,
                          [=] __device__ __host__(float x) {
                              return std::exp(x - mx) / exp_sum;
                          });
    }
}

float CELoss(const Tensor &t, std::vector<int> labels) {
    assert(t.ndim() == 2);
    ssize_t N = t.shape()[0], C = t.shape().back();

    Tensor sm = zeros(t.shape());
    softmax(t, sm);

    float sum = 0;
    for (ssize_t i = 0; i < N; ++i) {
        sum -= std::log(sm.at({i, labels[i]}));
    }
    return sum / N;
}

void CELoss_grad(const Tensor &t, std::vector<int> labels, Tensor dx) {
    assert(t.ndim() == 1 || t.ndim() == 2);
    assert(dx.shape() == dx.shape());
    ssize_t N = t.ndim() == 1 ? 1 : t.shape()[0], C = t.shape().back();

    softmax(t, dx);
    for (int i = 0; i < N; ++i) {
        dx.set({i, labels[i]}, dx.at({i, labels[i]}) - 1);
    }
    dx = dx * (1. / N);
}
} // namespace ten
