#pragma once
#include "tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <string>

using ten::Tensor, ten::shape_t, ten::TensorDevice;
namespace py = pybind11;

namespace pyten {
py::array_t<float> numpy(const Tensor &x);
Tensor from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast>);
}