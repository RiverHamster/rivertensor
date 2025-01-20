# PKU Programming in AI class project
This library implements basic NN operators, including matrix multiplication,
2D convolution, pooling, activation functions and loss functions in CUDA C++
and exports a Python wrapper, implementing auto differentiation.
It is part of the class project of
[PKU Programming in AI](https://pkuprogramminginai.github.io/Labs-Documentation/#/)
course. The repo also includes PyTorch assignments of the final project.

## Building the project
The project is built with scikit-build-core. It depends on CUDA, numpy and PyBind11.
To run the tests and examples, pytest, scipy and torch are also needed.

First install the conda environment.
```bash
conda env create -f environment.yml -n rivertensor
conda activate rivertensor
```

Build and install the Python package.
```bash
pip install .
```

Run the tests.
```bash
pytest test
```

## Design
The `rivertensor` (abbr. `rt`) package consists of two parts:
- `rt.base`: a tensor library implemented in C++, featuring CPU/GPU memory
   management, arithmetic, matrix, CNN, pooling, softmax and cross-entropy layers.
- `rt.Tensor`: a simple Python class implementing automatic differentiation, by
  building computational graph (CG) with overloaded operators. Currently only a very
  limited subset of operators are supported, and the gradients are directly
  computed with `base.Tensor` instead of building a CG for the gradients, hence
  the second-order derivatives are unavailable.
- `rt.Net`: neural network abstract base class supporting load/dump weights.
- `rt.optim`: package consisting optimizers. Only `SGD` without momentum
  is supported currently.

## Part 1 & 2
This part implements a simple CIFAR10 classifier with PyTorch and ResNet.
CIFAR10 is a tiny image classification dataset consisting of 60,000 images
for training and 10,000 images for evaluation, in 10 classes. The images are
of size 32x32 have 3 channels each.

The model structure is adapted from [a GeeksforGeeks tutorial](https://www.geeksforgeeks.org/resnet18-from-scratch-using-pytorch/).
The Python codes are in `torch-task/`. 

- [`baseline.py`](./torch-task/baseline.py) is a single-GPU model training and
  evaluation code.
- [`ddp.py`](./torch-task/ddp.py) is a multi-GPU SpMD training and evaluation
  code utilizing
  `DistributedDataParallel` module in PyTorch. Gradient accumulation is
  performed implicitly by PyTorch. This code takes all the CPU cores for no
  good reason, and set `OMP_NUM_THREADS=1` will 
- [`modelpar.py`](./torch-task/modelpar.py) is a dual-GPU training and evaluation
  code, manually splitting the model to two GPUs. Currently no pipelining is
  implemented, and no throughput gain is expected therefore.

The three codes are logically equivalent, so similar training losses and
accuracies are anticipated. The results are listed below (5 epochs, on two A100s).

|Code|Eval acc|Wall time|Max train mem per GPU|
|:---:|:---:|:---:|:---:|
|baseline|78.4%|18.8s|1586 MiB|
|ddp|78.6%|27.7s|1394 MiB|
|modelpar|77.2%|23.2s|1198 MiB|

The distributed training codes does not show performance advantages over
single-GPU training, possibly due to the small image size (32x32) and high
communication costs (~100 mini-batches per second).
If larger image size is applied `ddp` may show higher throughput.

## Part 3
[`example/mnist.py`](./example/mnist.py) (which is quite self-explanatory)
implements MNIST training and evaluation using the autodiff
feature of the `rivertensor` package.

The code is shipped to train from scratch. Uncomment the load statement and set
`n_epoch = 0` evaluates the model without training. The code assumes that the
current working directory is the project root.

For the CNN network, the results are listed below (on single 4060Ti):

|Peak GPU mem|eval acc|time per epoch|Typical GPU util|
|:---:|:---:|:---:|:---:|
|~1200 MiB|98.59%|28s|75%|

While meeting correctness criteria for the autodiff library,
the training speed is not comparable to PyTorch, and the GPU utilization is not
quite satisfactory, possibly due to the following reasons:
- Some operations introduces implicit CPU/GPU synchronization, including
  loss display and synchronous thrust calls, which lowers GPU utilization.
- Convolution operation implementation does not utilize TF32 tensor cores, and
  the data movement is not optimally implemented for both forward and backward
  passes, possibly using non-coalesced memory accesses or bank conflicts.

[`mnist/mnist.py`](./mnist/mnist.py) is another code implementing MNIST training
with manual linear computational graph construction.
