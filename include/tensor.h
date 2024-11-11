#pragma once
#include <cstdint>
#include <memory>
#include <vector>

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
    // add an unit dimension before axis i (non-copy)
    Tensor newaxis(ssize_t axis) const;
    // change dimensions without affecting size (non-copy)
    Tensor reshape(shape_t shape) const;
};

Tensor matmul(const Tensor &a, const Tensor &b);

#define DEC_MAP_OPT(name) Tensor name(const Tensor &x);
#define DEC_BINARY_OPT(name, arg1, arg2)                                       \
    Tensor name(const Tensor &arg1, const Tensor &arg2);
#define DEC_COMM_SCALAR_OPT(name)                                              \
    Tensor name(const Tensor &x, float y);                                     \
    Tensor name(float x, const Tensor &y);

// we do not implement expression templates. Implement more fused operators!
DEC_MAP_OPT(relu)
DEC_MAP_OPT(sigmoid)
DEC_MAP_OPT(exp)
DEC_MAP_OPT(log)
DEC_MAP_OPT(log2)
DEC_MAP_OPT(sqrt)
DEC_MAP_OPT(square)
DEC_MAP_OPT(operator-)
DEC_BINARY_OPT(relu_grad, x, grad)
DEC_BINARY_OPT(sigmoid_grad, y, grad)
DEC_BINARY_OPT(operator+, x, y)
DEC_BINARY_OPT(operator-, x, y)
DEC_BINARY_OPT(operator*, x, y)
DEC_BINARY_OPT(operator/, x, y)
DEC_COMM_SCALAR_OPT(operator+)
DEC_COMM_SCALAR_OPT(operator*)

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
void fc_grad_w(const Tensor &x, const Tensor &dy, Tensor dx);
float inner(const Tensor &a, const Tensor &b);
Tensor outer(const Tensor &a, const Tensor &b);
void outer_update(const Tensor &x, const Tensor &y, float alpha, Tensor t);

// convolution
// t: (N, C, H, W)
// ker: (9, C, K)
// out: (N, K, H, W)
void conv2d_3x3(const Tensor &x, const Tensor &ker, Tensor y);
void conv2d_3x3_grad_x(const Tensor &y, const Tensor &ker, Tensor dx);
void conv2d_3x3_grad_k(const Tensor &y, const Tensor &ker, Tensor dk);

// pooling
void maxpool2d_2x2(const Tensor &x, Tensor y);
void maxpool2d_2x2_grad(const Tensor &x, const Tensor &y, Tensor dx);

// loss functions
void softmax(const Tensor &x, Tensor y);
float CELoss(const Tensor &x, std::vector<int> labels);
void CELoss_grad(const Tensor &x, std::vector<int> labels, Tensor out);
} // namespace ten
