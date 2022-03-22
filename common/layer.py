# maskという変数は、True,FalseからなるNumpy行列で、xの要素で0以下の場所をTrue、それ以外をFalseとして保持する
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    # self.maskがTrueの場所(xが0の場所)を0に設定する
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

import numpy as np
    
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

