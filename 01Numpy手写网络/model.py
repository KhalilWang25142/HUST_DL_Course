import math
import numpy as np


def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    return (x > 0).astype(x.dtype)


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1.0 - np.tanh(x) ** 2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def d_leaky_relu(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha).astype(x.dtype)


def softmax(x):
    x_shift = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def get_activation_functions(name="relu"):
    name = name.lower()
    if name == "relu":
        return relu, d_relu
    if name == "tanh":
        return tanh, d_tanh
    if name == "sigmoid":
        return sigmoid, d_sigmoid
    if name == "leaky_relu":
        return leaky_relu, d_leaky_relu
    raise ValueError(f"Unsupported activation: {name}")


class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim, activation=None, d_activation=None):
        self.activation = activation
        self.d_activation = d_activation

        if activation in [relu, leaky_relu]:
            scale = math.sqrt(2.0 / input_dim)
        else:
            scale = math.sqrt(1.0 / input_dim)

        self.W = (np.random.randn(input_dim, output_dim) * scale).astype(np.float32)
        self.b = np.zeros((1, output_dim), dtype=np.float32)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b

        if self.activation is None:
            self.A = self.Z
        else:
            self.A = self.activation(self.Z)

        return self.A

    def backward(self, dA):
        batch_size = self.X.shape[0]

        if self.activation is None:
            dZ = dA
        else:
            dZ = dA * self.d_activation(self.Z)

        self.dW = (self.X.T @ dZ) / batch_size
        self.db = np.sum(dZ, axis=0, keepdims=True) / batch_size
        dX = dZ @ self.W.T

        return dX

    def update(self, lr):
        self.W = self.W - lr * self.dW
        self.b = self.b - lr * self.db


class Net:
    def __init__(self, activation=relu, d_activation=d_relu, lr=0.01):
        self.lr = lr

        self.fc1 = FullyConnectedLayer(784, 256, activation=activation, d_activation=d_activation)
        self.fc2 = FullyConnectedLayer(256, 128, activation=activation, d_activation=d_activation)

        # 输出层不加激活函数
        self.fc3 = FullyConnectedLayer(128, 10, activation=None, d_activation=None)

        self.layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)

        y_pred = softmax(X)
        y_pred_label = np.argmax(y_pred, axis=1)
        return y_pred, y_pred_label

    def backward(self, d_logits):
        dA = d_logits
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update(self):
        for layer in self.layers:
            layer.update(self.lr)