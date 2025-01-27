import numpy as np
from variable import Variable


class Function:
    input: Variable

    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        return output

    def forward(self, x: np.ndarray):
        raise NotImplementedError()

    def backward(self, gy: np.ndarray):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: np.ndarray):
        return x ** 2

    def backward(self, gy: np.ndarray):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: np.ndarray):
        return np.exp(x)

    def backward(self, gy: np.ndarray):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
