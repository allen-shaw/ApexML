from variable import Variable
from diff import numerical_diff
import function as F
import numpy as np

if __name__ == "__main__":
    f = F.Square()
    x = Variable(np.array(2.0))
    dy = numerical_diff(f, x)
    print(dy)
