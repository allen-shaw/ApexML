import numpy as np
from typing import Optional
from function import Function


class Variable:
    data: np.ndarray
    grad: Optional[np.ndarray]
    creator: Function

    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = None

    def set_creater(self, func: Function):
        self.creator = func
