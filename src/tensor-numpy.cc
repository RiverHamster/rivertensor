#include "pytensor.h"
#include <cuda_runtime.h>

namespace pyten {
py::array_t<float> numpy(const Tensor &t) {
    py::array_t<float> np(t.shape());
    float *np_data = np.mutable_data();
    cudaMemcpy(np_data, t.data(), t.size() * sizeof(float),
               t.device() == TensorDevice::cpu ? cudaMemcpyHostToHost
                                                : cudaMemcpyDeviceToHost);
    return np;
}

Tensor from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> np) {
    shape_t shape(np.shape(), np.shape() + np.ndim());
    Tensor t(shape, TensorDevice::gpu);
    // we currently force GPU tensors
    cudaMemcpy(t.data(), np.data(), t.size() * sizeof(float),
               t.device() == TensorDevice::cpu ? cudaMemcpyHostToHost
                                               : cudaMemcpyHostToDevice);
    return t;
}
}