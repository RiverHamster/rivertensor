from . import base
from abc import ABC, abstractmethod
from .opbase import Op
from .op import *
from typing import Optional
import numpy as np

class Op(ABC):
    @abstractmethod
    def forward(self, *inputs):
        pass

    @abstractmethod
    def backward(self, inputs, grad):
        pass

def make_from_op(op: Op, *inputs: base.Tensor, requires_grad = None) -> base.Tensor:
    if requires_grad == None:
        requires_grad = any(inp.requires_grad for inp in inputs)
    return Tensor(op=op, inputs=inputs, requires_grad=requires_grad)

class Tensor:
    def __init__(self, data: Optional[base.Tensor] = None, op: Optional[Op] = None, 
                 inputs: Optional[list['Tensor']] = None,
                 requires_grad: bool = False):
        assert data == None or isinstance(data, float) or isinstance(data, base.Tensor)
        self.data = data
        self.op = op
        self.inputs = inputs
        self.requires_grad = requires_grad
        # we does not implement grad as a CG Tensor
        self.grad: Optional[base.Tensor | float] = None
    
    @property
    def value(self) -> base.Tensor:
        if self.data != None:
            return self.data
        elif self.op != None:
            return self.op.forward(*(inp.value for inp in self.inputs))
        else:
            raise ValueError("No data or op")
    
    # data should not be of type `float`. This is a flaw.
    # Tensor of shape () should be used here instead.
    
    @property
    def size(self):
        if isinstance(self.value, float):
            return 1
        return self.value.size()

    @property
    def shape(self):
        if isinstance(self.value, float):
            return ()
        return self.value.shape()

    @property
    def stride(self):
        if isinstance(self.value, float):
            return ()
        return self.value.stride()

    @property
    def device(self):
        if isinstance(self.value, float):
            return base.TensorDevice.cpu
        return self.value.device()
    
    def numpy(self):
        if isinstance(self.value, float):
            return np.array(self.value)
        return self.value.numpy()

    def backward(self, grad_x: 'Tensor' = None):
        x = self
        assert x.requires_grad
        if grad_x == None:
            grad_x = base.ones(x.shape)
        if isinstance(grad_x, Tensor):
            grad_x = grad_x.value
        vis = set()
        topo: list[Tensor] = []

        # DFS post-order is reversed topo-order
        def DFS(u: Tensor):
            # print(f"DFS id {id(u)}")
            vis.add(id(u))
            if u.inputs != None:
                for v in u.inputs:
                    if id(v) not in vis and v.requires_grad:
                        DFS(v)
            topo.append(u)

        DFS(x)
        for u in topo:
            u.grad = None
        x.grad = grad_x
        for u in reversed(topo):
            # print(f"backward id {id(u)}")
            if u.op == None or u.inputs == None:
                continue
            grads_input = u.op.backward([t.value for t in u.inputs],
                                        u.grad)
            for inp, grad in zip(u.inputs, grads_input):
                if inp.grad == None:
                    inp.grad = grad
                else:
                    inp.grad += grad
    
    def __add__(self, other: 'Tensor') -> 'Tensor':
        return make_from_op(Add(), self, other)

    def __sub__(self, other: 'Tensor') -> 'Tensor':
        return make_from_op(Sub(), self, other)

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        return make_from_op(Mul(), self, other)

    def __truediv__(self, other: 'Tensor') -> 'Tensor':
        return make_from_op(Div(), self, other)
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return make_from_op(Matmul(False, False), self, other)
    
    def reshape(self, shape) -> 'Tensor':
        return make_from_op(Reshape(shape), self)
    
    def __repr__(self) -> str:
        return self.numpy().__repr__()
    
    def __str__(self) -> str:
        return self.numpy().__str__()


def relu(x, requires_grad = None):
    return make_from_op(ReLU(), x, requires_grad=requires_grad)

def sigmoid(x, requires_grad = None):
    return make_from_op(Sigmoid(), x, requires_grad=requires_grad)

def conv2d(x, k, requires_grad = None):
    return make_from_op(Conv2d(), x, k, requires_grad=requires_grad)

def maxpool2d(x, requires_grad = None):
    return make_from_op(MaxPool2d(), x, requires_grad=requires_grad)

def cross_entropy(x, y, requires_grad = None):
    return make_from_op(CELoss(y), x, requires_grad=requires_grad)

def inner(x, y, requires_grad = None):
    return make_from_op(Inner(), x, y, requires_grad=requires_grad)
