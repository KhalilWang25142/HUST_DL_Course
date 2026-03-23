# Input(784) → FC(256) → ReLU → FC(128) → ReLU → Output(10) → Softmax
# activate f -> ReLU, Tanh, Sigmoid
# Softmax
import numpy as np
import math

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

def init_stg(method = "norm", input_dim = 256):
    if method == "norm":
        scale = 0.01
    if method == "Xavier":
        scale = math.sqrt(1.0 / input_dim)
    if method == "He":
        scale = math.sqrt(2.0 / input_dim)
    return scale

class FullyConnectedLayer():
    def __init__(self, input_dim, output_dim, activation=None, d_activation=None, method="norm"):
        scale = init_stg(method = method, input_dim = input_dim)
        self.W = (np.random.randn(input_dim, output_dim) * scale).astype(np.float32)
        self.b = np.zeros((1, output_dim), dtype=np.float32)
        self.activation = activation
        self.d_activation = d_activation

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

        if self.activation == None:
            dZ = dA
        else :
            dZ = dA * self.d_activation(self.Z)

        self.dW = (self.X.T @ dZ) / batch_size
        self.db = np.sum(dZ, axis=0, keepdims=True) / batch_size
        dX = dZ @ self.W.T

        return dX

    def update(self, lr):
        self.W = self.W - lr * self.dW
        self.b = self.b - lr * self.db

class DropOutLayer():
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, X, training = True):
        if (not training):
            return X * (1.0 - self.dropout_rate)
        else:
            self.mask = (np.random.rand(*X.shape) > self.dropout_rate).astype(X.dtype)
            return X * self.mask

    def backward(self, dA):
        return dA * self.mask

class Net():
    def __init__(self, hidden_num1=256, hidden_num2=128, activation=relu, d_activation=d_relu, lr=0.01, method="norm", dropout_rate=0.0):
        self.fc1 = FullyConnectedLayer(784, hidden_num1, activation=activation, d_activation=d_activation, method=method)
        self.fc2 = FullyConnectedLayer(hidden_num1, hidden_num2, activation=activation, d_activation=d_activation, method=method)
        self.fc3 = FullyConnectedLayer(hidden_num2, 10, activation=None, d_activation=None, method=method)
        self.layers = [self.fc1, self.fc2, self.fc3]
        self.lr = lr
        eps = 1e-12
        if dropout_rate > eps:
            print("Dropout rate:", dropout_rate)
            self.drop1 = DropOutLayer(dropout_rate)
            self.drop2 = DropOutLayer(dropout_rate)
            self.layers = [self.fc1, self.drop1, self.fc2, self.drop2, self.fc3]

        # '''
        # group 2 exp
        # '''
        #
        # self.fc1 = FullyConnectedLayer(784, hidden_num1, activation=activation, d_activation=d_activation, method=method)
        # # self.fc2 = FullyConnectedLayer(hidden_num1, hidden_num2, activation=activation, d_activation=d_activation, method=method)
        # self.fc3 = FullyConnectedLayer(hidden_num1, 10, activation=None, d_activation=None, method=method)
        # self.layers = [self.fc1, self.fc3]
        # self.lr = lr
        # eps = 1e-12
        # if dropout_rate > eps:
        #     print("Dropout rate:", dropout_rate)
        #     self.drop1 = DropOutLayer(dropout_rate)
        #     self.drop2 = DropOutLayer(dropout_rate)
        #     self.layers = [self.fc1, self.drop1, self.fc3]

    def forward(self, X, training = True):
        for layer in self.layers:
            if isinstance(layer, DropOutLayer):
                X = layer.forward(X, training=training)
            else:
                X = layer.forward(X)
        y_pred = softmax(X)
        y_pred_label = np.argmax(y_pred, axis=1)
        return y_pred, y_pred_label

    def backward(self, y_true, y_pred):
        dA = y_pred - y_true
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update(self):
        for layer in self.layers:
            if isinstance(layer, FullyConnectedLayer):
                layer.update(self.lr)


