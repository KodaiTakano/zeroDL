import os
from re import A
import re
import sys
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from common.functions import sigmoid, relu
import numpy as np
import matplotlib.pyplot as plt

# サイズ1000*100の標準正規分布(平均0, 分散1)
x = np.random.randn(1000, 100)

node_num = 100 # 隠れ層のノードの数
hidden_layer_size = 5 # 隠れ層が5層
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]
        
    # w = np.random.randn(node_num, node_num) * 1 # 標準偏差1の重み
    # w = np.random.randn(node_num, node_num) * 0.01 # 標準偏差0.01の重み
    # w = np.random.randn(node_num, node_num) / np.sqrt(node_num) # 標準偏差1/√100の重み
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num/2) # 標準偏差√2/√100の重み
    
    z = np.dot(x, w)
    # a = sigmoid(z) # シグモイド関数
    a = relu(z) # ReLU関数
    activations[i] = a
    
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0,1)) # x=0~1の範囲で30分割したヒストグラフ上に平坦化したaを表示
plt.show()