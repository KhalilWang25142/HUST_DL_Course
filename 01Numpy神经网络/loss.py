import numpy as np

from model import FullyConnectedLayer


class CrossEntropyLoss:
    def __init__(self, l2_lambda=0.0, eps=1e-12):
        self.l2_lambda = l2_lambda
        self.eps = eps

    def ce_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.eps, 1.0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def l2_loss(self, model):
        if self.l2_lambda <= 0:
            return 0.0

        reg = 0.0
        for layer in model.layers:
            if isinstance(layer, FullyConnectedLayer):
                reg += np.sum(layer.W ** 2)

        return 0.5 * self.l2_lambda * reg

    def __call__(self, y_true, y_pred, model=None):
        loss = self.ce_loss(y_true, y_pred)
        if model is not None:
            loss += self.l2_loss(model)
        return float(loss)

    def grad(self, y_true, y_pred):

        return y_pred - y_true

    def apply_l2_grad(self, model):

        if self.l2_lambda <= 0:
            return

        for layer in model.layers:
            if isinstance(layer, FullyConnectedLayer):
                layer.dW = layer.dW + self.l2_lambda * layer.W