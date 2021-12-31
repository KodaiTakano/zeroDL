from pickle import load
import pickle
from posixpath import basename
import sys, os
sys.path.append(os.pardir) # os.pardir = ../
from dataset.mnist import load_mnist
import numpy as np

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = load(f)
    return network

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

x, t = get_data() # x.shape (10000, 784) t.shape (10000, )
network: dict = init_network()

batch_size = 100 # 100個ずつ推論
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i: i + batch_size] # x_batch.shape (100, 784)
    y_batch = predict(network, x_batch) # y_batch.shape (100, 10)
    p = np.argmax(y_batch, axis=1) # p.shape(100, )
    accuracy_cnt += np.sum(p == t[i:i + batch_size])

print("Accurary:" + str(float(accuracy_cnt) / len(x)))