#include "tensor.h"
#include <cassert>

int main() {
    const int n = 1000000;
    auto t = ten::ones({n});
    assert(t.device() == ten::TensorDevice::gpu);
    assert(ten::sum(t) == n);
}