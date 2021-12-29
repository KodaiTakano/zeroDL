import sys, os
sys.path.append(os.pardir) # os.pardir = ../
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
