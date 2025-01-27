import numpy as np
from typing import Optional


class Variable:
    data: np.ndarray
    grad: Optional[np.ndarray]

    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = None
