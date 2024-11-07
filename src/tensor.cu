#include "macros.h"
#include "tensor.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>
#include <memory>

using std::intptr_t;
using std::size_t;

namespace ten {
static cudaMemcpyKind cu_copydir(TensorDevice src, TensorDevice dst) {
    static cudaMemcpyKind dir[2][2] = {
        {cudaMemcpyHostToHost, cudaMemcpyHostToDevice},
        {cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice}};
    return dir[(int)src][(int)dst];
}

Buffer::~Buffer() { free(); }
Buffer::Buffer() : data(nullptr), dev(TensorDevice::gpu) {}
Buffer::Buffer(ssize_t nelem, TensorDevice dev) : dev(dev) {
    ssize_t nbytes = ((nelem + 7) & (~7)) * sizeof(float);
    // null buffer
    if (nbytes == 0) {
        data = nullptr;
        return;
    }
    if (dev == TensorDevice::cpu) {
        cudaMallocHost(&data, nbytes);
    } else {
        // printf("GPU Buffer of %.6lf MB allocated\n", 4.0 * nelem / 1e6);
        cudaMalloc(&data, nbytes);
    }
}
Buffer::Buffer(Buffer &&r) {
    free();
    data = r.data;
    dev = r.dev;
}
void Buffer::free() {
    if (data == nullptr)
        return;
    if (dev == TensorDevice::cpu) {
        cudaFreeHost(data);
    } else {
        // printf("GPU Buffer freed\n");
        cudaFree(data);
    }
}
Buffer &Buffer::operator=(const Buffer &&r) {
    free();
    data = r.data;
    dev = r.dev;
    return *this;
}

Tensor::Tensor() : _size(0), _shape({0}), _stride({1}), _buf(std::make_shared<Buffer>()) {}
Tensor::Tensor(const shape_t &s, TensorDevice d)
    : _shape(s), _stride(s.size()) {
    assert(s.size() > 0);
    ssize_t prod = 1;
    for (ssize_t i = s.size() - 1; i != -1; --i) {
        _stride[i] = prod;
        prod *= s[i];
    }
    _size = prod;
    _buf = std::make_shared<Buffer>(size(), d);
    cudaMemset(data(), 0, size() * sizeof(float));
}

Tensor Tensor::copy_to(TensorDevice dev) {
    size_t nbytes = size() * sizeof(float);
    Tensor r(shape(), dev);
    if (nbytes)
        cudaMemcpy(r.data(), data(), nbytes, cu_copydir(device(), dev));
    return r;
}
Tensor Tensor::copy() { return copy_to(device()); }

ssize_t Tensor::ndim() const { return _shape.size(); }
ssize_t Tensor::size() const { return _size; }
ssize_t Tensor::aligned_size() const { return (_size + 7) & ~7; }
const std::vector<ssize_t> &Tensor::stride() const { return _stride; }
const std::vector<ssize_t> &Tensor::shape() const { return _shape; }
const float *Tensor::data() const { return _buf->data; }
float *Tensor::data() { return _buf->data; }
TensorDevice Tensor::device() const {
    return _buf ? _buf->dev : TensorDevice::cpu;
}

float Tensor::at(const std::vector<ssize_t> &idx) const {
    assert(idx.size() == _shape.size());
    const float *ptr = data();
    for (ssize_t i = 0; i != idx.size(); ++i) {
        assert(idx[i] < _shape[i]);
        ptr += idx[i] * _stride[i];
    }
    if (device() == TensorDevice::cpu)
        return *ptr;
    float val;
    cudaMemcpy(&val, ptr, sizeof(float), cudaMemcpyDeviceToHost);
    return val;
}

void Tensor::set(const std::vector<ssize_t> &idx, float val) {
    assert(idx.size() == _shape.size());
    float *ptr = data();
    for (ssize_t i = 0; i != idx.size(); ++i) {
        assert(idx[i] < _shape[i]);
        ptr += idx[i] * _stride[i];
    }
    if (device() == TensorDevice::cpu) {
        *ptr = val;
        return;
    }
    cudaMemcpy(ptr, &val, sizeof(float), cudaMemcpyHostToDevice);
}

} // namespace ten