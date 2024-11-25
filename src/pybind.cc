#include "pytensor.h"

PYBIND11_MODULE(pytensor, m) {
    m.doc() = "Tensor library for PKU Programming in AI course";
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<shape_t, TensorDevice>())
        .def("ndim", &Tensor::ndim)
        .def("shape", &Tensor::shape)
        .def("size", &Tensor::size)
        .def("stride", &Tensor::stride)
        .def("device", &Tensor::device)
        .def("copy", &Tensor::copy)
        .def("copy_to", &Tensor::copy_to)
        .def("at", &Tensor::at)
        .def("set", &Tensor::set)
        .def("newaxis", &Tensor::newaxis)
        .def("reshape", &Tensor::reshape)
        .def("numpy", [](const Tensor &t) { return pyten::numpy(t); })
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self + float())
        .def(float() + py::self)
        .def(py::self * float())
        .def(float() * py::self);
    py::enum_<TensorDevice>(m, "TensorDevice")
        .value("cpu", TensorDevice::cpu)
        .value("gpu", TensorDevice::gpu);
    m.def("zeros", &ten::zeros);
    m.def("ones", &ten::ones);
    m.def("rand", &ten::rand);
    m.def("randn", &ten::randn);
    m.def("relu", &ten::relu);
    m.def("relu_grad", &ten::relu_grad);
    m.def("sigmoid", &ten::sigmoid);
    m.def("sigmoid_grad", &ten::sigmoid_grad);
    m.def("sum", &ten::sum);
    m.def("max", &ten::max);
    m.def("min", &ten::min);
    m.def("sum_d0", &ten::sum_d0);
    m.def("from_numpy", &pyten::from_numpy);
    m.def("exp", &ten::exp);
    m.def("log", &ten::log);
    m.def("log2", &ten::log2);
    m.def("sqrt", &ten::sqrt);
    m.def("square", &ten::square);
    m.def("matmul", &ten::matmul);
    m.def("matmul_", &ten::matmul_);
    m.def("matmul_t", &ten::matmul_t);
    m.def("fc_update_grad_w", &ten::fc_update_grad_w);
    m.def("inner", &ten::inner);
    m.def("outer", &ten::outer);
    m.def("outer_update", &ten::outer_update);
    m.def("conv2d_3x3", &ten::conv2d_3x3);
    m.def("conv2d_3x3_grad_x", &ten::conv2d_3x3_grad_x);
    m.def("conv2d_3x3_grad_k", &ten::conv2d_3x3_grad_k);
    m.def("maxpool2d_2x2", &ten::maxpool2d_2x2);
    m.def("maxpool2d_2x2_grad", &ten::maxpool2d_2x2_grad);
    m.def("softmax", &ten::softmax);
    m.def("CELoss", &ten::CELoss);
    m.def("CELoss_grad", &ten::CELoss_grad);
}
