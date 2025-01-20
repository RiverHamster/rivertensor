from .opbase import Op
from . import base

class Add(Op):
    def forward(self, a: base.Tensor, b: base.Tensor):
        return a + b

    def backward(self, inputs, grad):
        return grad, grad

class Sub(Op):
    def forward(self, a: base.Tensor, b: base.Tensor):
        return a - b

    def backward(self, inputs, grad):
        return grad, -grad

class Mul(Op):
    def forward(self, a: base.Tensor, b: base.Tensor):
        return a * b

    def backward(self, inputs, grad):
        a, b = inputs
        return grad * b, grad * a

class Div(Op):
    def forward(self, a: base.Tensor, b: base.Tensor):
        return a / b

    def backward(self, inputs, grad):
        a, b = inputs
        return grad / b, -grad * a / (b * b)

class Matmul(Op):
    def __init__(self, tA: bool = False, tB: bool = False):
        self.tA = tA
        self.tB = tB

    def forward(self, a: base.Tensor, b: base.Tensor):
        return base.matmul_t(a, b, self.tA, self.tB)

    def backward(self, inputs, grad):
        a, b = inputs
        return (base.matmul_t(grad, b, False, not self.tB),
                base.matmul_t(a, grad, not self.tA, False))

class Conv2d(Op):
    def forward(self, x: base.Tensor, k: base.Tensor):
        return base.conv2d_3x3(x, k)
    
    def backward(self, inputs, grad):
        x, k = inputs
        return base.conv2d_3x3_grad_x(grad, k), base.conv2d_3x3_grad_k(grad, x)

# optimize with stateful?

class ReLU(Op):
    def forward(self, x: base.Tensor):
        return base.relu(x)

    def backward(self, inputs, grad):
        x, = inputs
        return base.relu_grad(x, grad),

class MaxPool2d(Op):
    def forward(self, x: base.Tensor):
        return base.maxpool2d_2x2(x)

    def backward(self, inputs, grad):
        x = inputs[0]
        return base.maxpool2d_2x2_grad(x, grad),

class Sigmoid(Op):
    def forward(self, x: base.Tensor):
        self.y = base.sigmoid(x)
        return self.y

    def backward(self, inputs, grad):
        y = self.y
        return base.sigmoid_grad(y, grad),

class CELoss(Op):
    def __init__(self, labels: list[int]):
        self.labels = labels
    
    def forward(self, x: base.Tensor):
        return base.CELoss(x, self.labels)
    
    def backward(self, inputs, grad):
        x = inputs[0]
        return base.CELoss_grad(x, self.labels) * grad,

class Reshape(Op):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x: base.Tensor):
        return x.reshape(self.shape)
    
    def backward(self, inputs, grad):
        return grad.reshape(inputs[0].shape()),

class Inner(Op):
    def forward(self, a: base.Tensor, b: base.Tensor):
        return base.inner(a, b)
    
    def backward(self, inputs, grad):
        a, b = inputs
        return grad * b, grad * a

