from .adtensor import Tensor
from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def step(self, params: dict[str, Tensor]):
        pass

class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr
    
    def step(self, params: dict[str, Tensor]):
        for name, value in params.items():
            # print("SGD: ", name, value)
            if value.requires_grad and value.grad != None:
                value.data -= self.lr * value.grad
