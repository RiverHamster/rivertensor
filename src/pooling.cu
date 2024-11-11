#include "macros.h"
#include "tensor.h"
#include <cassert>

namespace ten {
KERNEL maxpool2d_2x2_ker(const float *__restrict__ in, float *__restrict__ out,
                         ssize_t N, ssize_t C, ssize_t H, ssize_t W,
                         ssize_t tiles) {
    ssize_t tile_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_id >= tiles)
        return;
    int w = tile_id % (W / 2);
    tile_id /= (W / 2);
    int h = tile_id % (H / 2);
    tile_id /= (H / 2);
    int c = tile_id % C;
    tile_id /= C;
    int n = tile_id;

    const float *ptr = in + n * C * H * W + c * H * W + 2 * h * W + 2 * w;
    float mx = *ptr;
    if (ptr[1] > mx)
        mx = ptr[1];
    if (ptr[W] > mx)
        mx = ptr[W];
    if (ptr[W + 1] > mx)
        mx = ptr[W + 1];
    out[n * C * (H / 2) * (W / 2) + c * (H / 2) * (W / 2) + h * (W / 2) + w] =
        mx;
}

void maxpool2d_2x2(const Tensor &in, Tensor out) {
    assert(in.ndim() == 4);
    assert(out.ndim() == 4);
    ssize_t N = in.shape()[0], C = in.shape()[1], H = in.shape()[2],
            W = in.shape()[3];
    ssize_t tiles = N * C * (H / 2) * (W / 2);
    maxpool2d_2x2_ker<<<(tiles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        in.data(), out.data(), N, C, H, W, tiles);
}

KERNEL maxpool2d_2x2_d_ker(const float *__restrict__ x, const float *__restrict__ dy, float *__restrict__ dx,
                         ssize_t N, ssize_t C, ssize_t H, ssize_t W,
                         ssize_t tiles) {
    ssize_t tile_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_id >= tiles)
        return;
    int w = tile_id % (W / 2);
    tile_id /= (W / 2);
    int h = tile_id % (H / 2);
    tile_id /= (H / 2);
    int c = tile_id % C;
    tile_id /= C;
    int n = tile_id;

    int idx = n * C * H * W + c * H * W + 2 * h * W + 2 * w;
    const float *ptr = x + idx;
    float mx = *ptr;
    float grad = dy[n * C * (H / 2) * (W / 2) + c * (H / 2) * (W / 2) + h * (W / 2) + w];
    int mx_idx = 0;
    if (ptr[1] > mx)
        mx = ptr[1], mx_idx = 1;
    if (ptr[W] > mx)
        mx = ptr[W], mx_idx = W;
    if (ptr[W + 1] > mx)
        mx = ptr[W + 1], mx_idx = W + 1;
    dx[idx + mx_idx] = grad;
}

void maxpool2d_2x2_grad(const Tensor &x, const Tensor &dy, Tensor dx) {
    assert(x.ndim() == 4);
    assert(dy.ndim() == 4);
    assert(dx.ndim() == 4);
    ssize_t N = x.shape()[0], C = x.shape()[1], H = x.shape()[2],
            W = x.shape()[3];
    assert(dx.shape() == x.shape());
    assert(dy.shape() == (shape_t{N, C, H / 2, W / 2}));

    ssize_t tiles = N * C * (H / 2) * (W / 2);
    cudaMemsetAsync(dx.data(), 0, sizeof(float) * dx.size());
    maxpool2d_2x2_d_ker<<<(tiles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        x.data(), dy.data(), dx.data(), N, C, H, W, tiles);
}
} // namespace ten
