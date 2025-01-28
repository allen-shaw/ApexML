import numpy as np
from apex.variable import Variable
import apex.function as F


def test_backward():
    A = F.Square()
    B = F.Exp()
    C = F.Square()
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    print("x.grad:", x.grad)
    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x
