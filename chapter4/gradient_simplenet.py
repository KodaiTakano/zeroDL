import sys, os 
sys.path.append(os.pardir)
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient
import numpy as np

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # np.random.randn ガウス分布
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss
    
net = simpleNet()
print(net.W) # 重み

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

t = np.array([0, 0, 1]) # 正解ラベル
print(net.loss(x, t)) # 損失

# numerical_gradient(f,x)が内部でf(x)を実行するためのダミー関数
def f(W):
    return net.loss(x, t)

# 勾配
dW = numerical_gradient(f, net.W)
print(dW)
