from abc import ABC, abstractmethod

class Op(ABC):
    @abstractmethod
    def forward(self, *args):
        """
        Take some inputs (base.Tensor) and return the output (base.Tensor).
        """
        pass

    def __call__(self, *args):
        return self.forward(*args)

    @abstractmethod
    def backward(self, inputs, grad) -> tuple:
        """
        Takes some inputs (base.Tensor) and the gradient of the output
        (base.Tensor), calculate the gradient of the inputs.
        For simplicity, we will not build the computational graph for the
        gradients, to eliminate the need of implementing some wrapped operators.
        """
        pass
