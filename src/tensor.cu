#include "macros.h"
#include "tensor.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

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
        cudaMallocAsync(&data, nbytes, cudaStreamDefault);
        // printf("GPU Buffer of %.6lf MB (%zd elems) allocated: %p\n",
        //        4.0 * nelem / 1e6, nelem, data);
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
        // printf("GPU Buffer %p freed\n", data);
        cudaFreeAsync(data, cudaStreamDefault);
    }
}
Buffer &Buffer::operator=(const Buffer &&r) {
    free();
    data = r.data;
    dev = r.dev;
    return *this;
}

Tensor::Tensor()
    : _size(0), _shape({0}), _stride({1}), _buf(std::make_shared<Buffer>()) {}
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
}

Tensor Tensor::copy_to(TensorDevice dev) {
    size_t nbytes = size() * sizeof(float);
    Tensor r(shape(), dev);
    auto dir = cu_copydir(device(), dev);
    if (dir == cudaMemcpyDeviceToDevice && nbytes)
        cudaMemcpyAsync(r.data(), data(), nbytes, dir);
    else if (nbytes)
        cudaMemcpy(r.data(), data(), nbytes, dir);
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

Tensor Tensor::reshape(shape_t shape) const {
    ssize_t prod = 1;
    shape_t strides(shape.size());
    for (ssize_t i = shape.size() - 1; i != -1; --i) {
        strides[i] = prod;
        prod *= shape[i];
    }
    assert(prod == size());
    Tensor r = *this;
    r._shape = shape;
    r._stride = strides;
    return r;
}

Tensor Tensor::newaxis(ssize_t axis) const {
    assert(0 <= axis && axis <= size());
    Tensor r = *this;
    r._shape.insert(r._shape.begin() + axis, 1);
    r.reshape(r._shape);
    return r;
}
} // namespace ten
