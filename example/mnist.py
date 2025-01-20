import os
import torch
import torchvision
import torchvision.transforms.v2 as v2
import numpy as np
import tqdm
import rivertensor as rt
from rivertensor.optim import SGD

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set the root directory for the MNIST dataset
data_dir = os.path.join(script_dir, 'mnist-data')

batch_size = 32
n_epoch = 10
lr = 1e-2
param_file = "net/mnist_conv_net.pkl"

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

class FCNet(rt.Net):
    def __init__(self):
        self.params = {
            'w1': rt.tensor(rt.base.randn([784, 128]) * (784 ** -1/2), requires_grad=True),
            'w2': rt.tensor(rt.base.randn([128, 10]) * (128 ** -1/2), requires_grad=True)
        }
    
    def forward(self, x: rt.Tensor):
        x = x.reshape([x.shape[0], x.stride[0]])
        x = rt.relu(x @ self.params['w1'])
        x = x @ self.params['w2']
        return x

class ConvNet(rt.Net):
    def __init__(self):
        self.params = {
            'k1': rt.tensor(rt.base.randn([9, 1, 64]) * (1/3), requires_grad=True),
            'k2': rt.tensor(rt.base.randn([9, 64, 64]) * (1/3 * (64 ** -1/2)), requires_grad=True),
            'w1': rt.tensor(rt.base.randn([7 * 7 * 64, 128]) * ((7 * 7 * 64) ** -1/2), requires_grad=True),
            'w2': rt.tensor(rt.base.randn([128, 10]) * (128 ** -1/2), requires_grad=True)
        }
    
    def forward(self, x: rt.Tensor):
        x = rt.conv2d(x, self.params['k1'])
        x = rt.relu(x)
        x = rt.maxpool2d(x)
        x = rt.conv2d(x, self.params['k2'])
        x = rt.relu(x)
        x = rt.maxpool2d(x)
        x = x.reshape([x.shape[0], x.stride[0]])
        x = rt.relu(x @ self.params['w1'])
        x = x @ self.params['w2']
        return x

net = ConvNet()
# with open(param_file, "rb") as f:
#     net.load(f)
optimizer = SGD(lr=lr)

for e in range(n_epoch):
    with tqdm.tqdm(total=len(train_loader), desc=f'Epoch {e+1}/{n_epoch}') as pbar:
        running_loss = 0.0
        niter = 0
        for x, y in train_loader:
            # print(x.shape, y.shape)
            x_rt = rt.tensor(x.numpy())
            logits = net.forward(rt.tensor(x.numpy()))
            loss = rt.cross_entropy(logits, y.tolist())
            loss.backward()
            pbar.update(1)
            running_loss += loss.numpy()
            niter += 1
            pbar.set_postfix_str(f'loss={running_loss / niter:.4f}')
            # print(loss)
            optimizer.step(net.params)
    total = 0
    correct = 0
    for x, y in eval_loader:
        x_rt = rt.tensor(x.numpy())
        logits = net.forward(rt.tensor(x.numpy()))
        logits = logits.numpy()
        y_pred = np.argmax(logits, axis=1)
        correct += np.sum(y_pred == y.numpy())
        total += y.shape[0]
    print(f'Eval acc: {correct / total:.4f}')

# with open(param_file, "wb") as f:
#     net.dump(f)
