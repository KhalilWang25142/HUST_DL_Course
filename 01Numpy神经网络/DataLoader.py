# DataLoader，读入MNIST图像，并展成784维向量，sklearn，torch.data

from sklearn.datasets import fetch_openml
from torch.utils import data
import numpy as np
from utils import *

# 1. 使用sklearn加载数据集
class sklearnDataLoader():

    def __init__(self, data_home = "./scikit_data", seed=20260318):
        self.data_home = data_home
        self.train_rate = 0.8
        self.test_rate = 0.2
        self.seed = seed

    def Load(self):

        # 1. 读取数据集
        MNIST = fetch_openml('mnist_784', version=1, as_frame=False, cache=True, data_home=self.data_home)
        MNIST.target = MNIST.target.astype(np.int64)

        # 2. 归一化到[0, 1]
        MNIST.data = MNIST.data.astype(np.float32) / 255

        # 3. 随机打乱顺序，对标签做One-hot编码
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(MNIST.data))
        rng.shuffle(indices)

        MNIST.data = MNIST.data[indices]
        MNIST.target = MNIST.target[indices]
        MNIST.target = onehot(MNIST.target, 10)

        # 4. 按照 8:2 分成训练集和测试集
        X_train = MNIST.data[0: int(len(MNIST.data) * self.train_rate)]
        y_train = MNIST.target[0: int(len(MNIST.data) * self.train_rate)]

        X_test = MNIST.data[int(len(MNIST.data) * self.train_rate):]
        y_test = MNIST.target[int(len(MNIST.data) * self.train_rate):]

        return X_train, y_train, X_test, y_test

    def Load_X_y(self):
        X, y = fetch_openml("mnist_784", return_X_y=True, version=1, as_frame=False, cache=True, data_home=self.data_home)
        return X, y

# 2. 使用torch.data加载数据集

if __name__ == "__main__":

    # 1. test sklearn
    DataLoader = sklearnDataLoader()
    X_train, y_train, X_test, y_test = DataLoader.Load()
    print(X_train[0])
    print(y_train[0])
    print(type(y_train[0]))
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
