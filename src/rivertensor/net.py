from . import base
from .tensor_init import tensor
from abc import ABC, abstractmethod
import pickle

class Net(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def numpy(self):
        return {name: param.numpy() for name, param in self.params.items()}

    def dump(self, f):
        pickle.dump(self.numpy(), f)
    
    def load(self, f):
        params = pickle.load(f)
        self.params = {name: tensor(param, requires_grad=True) for name, param in params.items()}
