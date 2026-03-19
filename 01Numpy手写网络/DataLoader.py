from sklearn.datasets import fetch_openml
import numpy as np
from utils import onehot


class sklearnDataLoader:
    def __init__(self, data_home="./scikit_data", train_rate=0.8):
        self.data_home = data_home
        self.train_rate = train_rate
        self.test_rate = 1.0 - train_rate

    def Load(self):
        mnist = fetch_openml(
            "mnist_784",
            version=1,
            as_frame=False,
            cache=True,
            data_home=self.data_home
        )

        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int64)

        np.random.seed(20260317)
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        X = X[indices]
        y = y[indices]

        y = onehot(y, num_classes=10).astype(np.float32)

        split = int(len(X) * self.train_rate)

        X_train = X[:split]
        y_train = y[:split]
        X_test = X[split:]
        y_test = y[split:]

        return X_train, y_train, X_test, y_test

    def Load_X_y(self):
        X, y = fetch_openml(
            "mnist_784",
            return_X_y=True,
            version=1,
            as_frame=False,
            cache=True,
            data_home=self.data_home
        )
        X = X.astype(np.float32) / 255.0
        y = y.astype(np.int64)
        return X, y


if __name__ == "__main__":
    dataloader = sklearnDataLoader()
    X_train, y_train, X_test, y_test = dataloader.Load()
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)