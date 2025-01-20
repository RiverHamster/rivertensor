from .adtensor import Tensor
import numpy as np
from . import base

def tensor(arr, requires_grad=False):
    if isinstance(arr, base.Tensor):
        return Tensor(arr, requires_grad=requires_grad)
    elif isinstance(arr, np.ndarray):
        t = base.from_numpy(arr).copy_to(base.TensorDevice.gpu)
        return Tensor(t, requires_grad=requires_grad)
    else:
        t = base.from_numpy(np.array(arr)).copy_to(base.TensorDevice.gpu)
        return Tensor(t, requires_grad=requires_grad)

def zeros(shape, requires_grad=False):
    return Tensor(base.zeros(shape), requires_grad=requires_grad)

def ones(shape, requires_grad=False):
    return Tensor(base.ones(shape), requires_grad=requires_grad)

def zeros_like(t, requires_grad=False):
    return zeros(t.shape, requires_grad=requires_grad)

def ones_like(t, requires_grad=False):
    return ones(t.shape, requires_grad=requires_grad)

def rand(shape, requires_grad=False):
    return Tensor(base.rand(shape), requires_grad=requires_grad)

def randn(shape, requires_grad=False):
    return Tensor(base.randn(shape), requires_grad=requires_grad)

def rand_like(t, requires_grad=False):
    return rand(t.shape, requires_grad=requires_grad)

def randn_like(t, requires_grad=False):
    return randn(t.shape, requires_grad=requires_grad)

def from_numpy(arr, requires_grad=False):
    t = base.from_numpy(arr).copy_to(base.TensorDevice.gpu)
    return Tensor(t, requires_grad=requires_grad)
