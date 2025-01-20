import os
import torch
import torchvision
import torchvision.transforms.v2 as v2
from rivertensor import base as pt
import numpy as np
import tqdm

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set the root directory for the MNIST dataset
data_dir = os.path.join(script_dir, 'mnist-data')

batch_size = 32
n_epoch = 10
lr = 2e-2

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.5], [0.5])
])

# Download and load the MNIST dataset
train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True,
                                       transform=transform)
eval_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True,
                                      transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True)
eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size,
                                          shuffle=False)

# Sequential computational graph
class LNode:
    def __call__(self, x: pt.Tensor):
        pass
    def backward(self, dy: pt.Tensor, lr: float):
        pass

class Conv2d(LNode):
    def __init__(self, C, K):
        self.C = C
        self.K = K
        self.ker = pt.randn([9, C, K])

    def __call__(self, x):
        self.X = x
        return pt.conv2d_3x3(x, self.ker)
    
    def backward(self, dy, lr):
        self.ker -= lr * pt.conv2d_3x3_grad_k(dy, self.X)
        return pt.conv2d_3x3_grad_x(dy, self.ker)

class Linear(LNode):
    def __init__(self, C, K):
        self.C = C
        self.K = K
        self.W = pt.randn([C, K])
    def __call__(self, x):
        self.X = x
        return pt.matmul(x, self.W)
    # dy: [N, K]
    def backward(self, dy, lr):
        # print(self.X.shape(), dy.shape(), self.W.shape())
        # print("matmul_")
        pt.matmul_(self.X, dy, self.W, True, False, -lr, 1)
        # print("matmul_t")
        return pt.matmul_t(dy, self.W, False, True)

class ReLU(LNode):
    def __call__(self, x):
        self.X = x
        return pt.relu(x)
    def backward(self, dy, lr):
        return pt.relu_grad(self.X, dy)

class Sigmoid(LNode):
    def __call__(self, x):
        self.Y = pt.sigmoid(x)
        return self.Y
    def backward(self, dy, lr):
        return pt.sigmoid_grad(self.Y, dy)

class MaxPool2d(LNode):
    def __call__(self, x):
        self.X = x
        return pt.maxpool2d_2x2(x)
    def backward(self, dy, lr):
        return pt.maxpool2d_2x2_grad(self.X, dy)

class Flatten(LNode):
    def __call__(self, x):
        self.shape = x.shape()
        return x.reshape([x.shape()[0], x.stride()[0]])
    def backward(self, dy, lr):
        return dy.reshape(self.shape)

Activation = Sigmoid

cg = [Conv2d(1, 8), MaxPool2d(), Activation(), Conv2d(8, 16), MaxPool2d(),
      Activation(), Flatten(), Linear(7 * 7 * 16, 64), Activation(),
      Linear(64, 10)]
# cg = [Flatten(), Linear(28 * 28, 10)]

def eval(cg: list[LNode], x):
    """
    evaluate computational graph with x: (N, 1, H, W)
    """
    for node in cg:
        # print(x.shape())
        x = node(x)
    return x

def train_step(cg: list[LNode], x: pt.Tensor, y: list[int]):
    """
    SGD step.
    returns the batch average loss
    """
    # print("Eval")
    x = eval(cg, x)
    # print(x.numpy())
    # print(pt.softmax(x).numpy())
    loss = pt.CELoss(x, y)
    grad = pt.CELoss_grad(x, y)
    # print("Backprop")
    for node in reversed(cg):
        # print(grad.shape())
        grad = node.backward(grad, lr)
    return loss

for epoch in range(n_epoch):
    print(f"epoch {epoch + 1}")
    pbar = tqdm.tqdm(total=len(train_loader))
    running_loss = 0.0
    for idx, (input, labels) in enumerate(train_loader):
        input = pt.from_numpy(input.numpy())
        labels = labels.tolist()
        running_loss += train_step(cg, input, labels)
        pbar.update(1)
        pbar.set_postfix_str(f"loss {running_loss / (idx + 1):.4f}")
    del pbar

    eval_correct = 0
    eval_total = 0
    for input, labels in eval_loader:
        eval_total += input.shape[0]
        input = pt.from_numpy(input.numpy())
        labels = labels.tolist()
        y = eval(cg, input)
        y = np.argmax(y.numpy(), axis=1)
        eval_correct += np.sum(y == np.array(labels))

    print(f"acc : {eval_correct / eval_total:.4f}")
