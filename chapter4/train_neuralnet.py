import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_class import TwoLayerNet
import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# ハイパーパラメータ
iters_num = 10000 # 繰り返し回数
train_size = x_train.shape[0] # 60000
batch_size = 50
learning_rate = 0.1

# 損失関数の値を格納
train_loss_list = []

# 訓練データとテストデータに対する認識精度を格納
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1) # 600

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size) # 60000までの整数からランダムで長さ100の配列を作成
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch)

    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 1エポックごとに精度を計算
    if i % iter_per_epoch == 0: #600回ごと
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))