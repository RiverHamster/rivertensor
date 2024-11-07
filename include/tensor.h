#pragma once
#include <cstdint>
#include <vector>
#include <memory>

using std::size_t;

namespace ten {
using ssize_t = std::intptr_t;
using shape_t = std::vector<ssize_t>;

const int BLOCK_SIZE = 256;

enum class TensorDevice { cpu = 0, gpu = 1 };

// Heterogeneous buffer with unique ownership semantics
// for a buffer with size 0, data should be nullptr
struct Buffer {
    float *data;
    TensorDevice dev;

    ~Buffer();
    // initialize with nullptr.
    Buffer();
    // allcoate FP32 * N on CPU or GPU
    Buffer(ssize_t nelem, TensorDevice dev);
    // no copy
    Buffer(const Buffer &r) = delete;
    // free the currently owned buffer
    void free();
    // move buffer
    Buffer(Buffer &&r);
    // no copy
    Buffer &operator=(const Buffer &r) = delete;
    // free the currently owned buffer and move a new buffer in
    Buffer &operator=(const Buffer &&r);
};

// The Tensor class, with shape, stride and shared ownership semantics
// empty tensor: shape = [0], stride = [1], buf points to a empty Buffer object
class Tensor {
  private:
    shape_t _shape;
    shape_t _stride;
    ssize_t _size;
    std::shared_ptr<Buffer> _buf;

  public:
    ssize_t ndim() const;
    // product of dimensions
    ssize_t size() const;
    // size, aligned to 8 elements or 32 bytes
    ssize_t aligned_size() const;
    // strides for each dimension
    const std::vector<ssize_t> &stride() const;
    // dimensions of the array
    const std::vector<ssize_t> &shape() const;
    // data pointer, either CPU or GPU
    const float *data() const;
    float *data();
    // enum indicating the device
    TensorDevice device() const;

    Tensor();
    // initalize a tensor with 0
    Tensor(const shape_t &s, const TensorDevice d);
    // other constructors remain default

    // make a copy of the current Tensor
    Tensor copy();
    // make a copy to a specific device
    Tensor copy_to(TensorDevice dev);

    // get single element
    float at(const std::vector<ssize_t> &idx) const;
    // set single element
    void set(const std::vector<ssize_t> &idx, float val);
};

Tensor matmul(const Tensor &a, const Tensor &b);

#define DEC_MAP_OPT(name) Tensor name(const Tensor &x);
#define DEC_BINARY_OPT(name, arg1, arg2)                                       \
    Tensor name(const Tensor &arg1, const Tensor &arg2);

// we do not implement expression templates. Implement more fused operators!
DEC_MAP_OPT(relu)
DEC_MAP_OPT(sigmoid)
DEC_MAP_OPT(exp)
DEC_MAP_OPT(log)
DEC_MAP_OPT(log2)
DEC_MAP_OPT(sqrt)
DEC_MAP_OPT(square)
DEC_MAP_OPT(operator-)
DEC_BINARY_OPT(reluGrad, x, grad)
DEC_BINARY_OPT(sigmoidGrad, y, grad)
DEC_BINARY_OPT(operator+, x, y)
DEC_BINARY_OPT(operator-, x, y)
DEC_BINARY_OPT(operator*, x, y)
DEC_BINARY_OPT(operator/, x, y)

Tensor zeros(const shape_t &);
Tensor ones(const shape_t &);
Tensor rand(const shape_t &);
Tensor randn(const shape_t &);

// reduction
float sum(const Tensor &t);
float max(const Tensor &t);
float min(const Tensor &t);
Tensor sum_d0(const Tensor &t);

// matmul
Tensor matmul(const Tensor &a, const Tensor &b);
float inner(const Tensor &a, const Tensor &b);
Tensor outer(const Tensor &a, const Tensor &b);
void outer_update(const Tensor &x, const Tensor &y, float alpha, Tensor t);
} // namespace ten