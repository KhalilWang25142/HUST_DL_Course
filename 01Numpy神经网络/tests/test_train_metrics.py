from argparse import Namespace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train
from train import train_one_epoch


class DummyModel:
    def __init__(self, predictions):
        self._predictions = list(predictions)
        self.backward_calls = 0
        self.update_calls = 0

    def forward(self, X, training=True):
        y_pred = self._predictions.pop(0)
        y_pred_label = np.argmax(y_pred, axis=1)
        return y_pred, y_pred_label

    def backward(self, y_true, y_pred):
        self.backward_calls += 1

    def update(self):
        self.update_calls += 1


class DummyLoss:
    def __call__(self, y_true, y_pred, model):
        return 0.25

    def apply_l2_grad(self, model):
        return None


class FakeRun:
    def __init__(self):
        self.logged = []
        self.summary = {}

    def log(self, payload, step=None):
        self.logged.append((payload, step))


def test_train_one_epoch_returns_loss_and_accuracy(monkeypatch):
    X_train = np.arange(4 * 784, dtype=np.float32).reshape(4, 784)
    y_train_labels = np.array([0, 1, 2, 3])
    y_train = np.eye(10, dtype=np.float32)[y_train_labels]

    batch_1_pred = np.eye(10, dtype=np.float32)[np.array([0, 1])]
    batch_2_pred = np.eye(10, dtype=np.float32)[np.array([9, 3])]

    model = DummyModel(predictions=[batch_1_pred, batch_2_pred])
    loss = DummyLoss()
    monkeypatch.setattr(np.random, "shuffle", lambda _: None)

    train_loss, train_acc = train_one_epoch(
        model=model,
        loss=loss,
        X_train=X_train,
        y_train=y_train,
        batch_size=2,
    )

    assert train_loss == 0.25
    assert train_acc == 0.75
    assert model.backward_calls == 2
    assert model.update_calls == 2


def test_train_epochs_tracks_best_val_accuracy_when_not_final_epoch(monkeypatch):
    args = Namespace(epochs=3, batch_size=2)
    fake_run = FakeRun()

    train_results = iter([(0.4, 0.7), (0.3, 0.8), (0.2, 0.85)])
    eval_results = iter(
        [
            (0.5, 0.72, np.array([0]), np.array([0]), np.zeros((1, 784)), np.array([0]), np.array([0])),
            (0.45, 0.83, np.array([1]), np.array([1]), np.zeros((1, 784)), np.array([1]), np.array([1])),
            (0.48, 0.79, np.array([2]), np.array([2]), np.zeros((1, 784)), np.array([2]), np.array([2])),
        ]
    )

    monkeypatch.setattr(train, "train_one_epoch", lambda *a, **k: next(train_results))
    monkeypatch.setattr(train, "evaluate_one_epoch", lambda *a, **k: next(eval_results))
    monkeypatch.setattr(train, "build_wandb_image_payload", lambda **kwargs: ["image"])

    history, best_y_true, best_y_pred, summary = train.train_epochs(
        args=args,
        model=object(),
        loss=object(),
        X_train=np.zeros((2, 784), dtype=np.float32),
        y_train=np.zeros((2, 10), dtype=np.float32),
        X_test=np.zeros((2, 784), dtype=np.float32),
        y_test=np.zeros((2, 10), dtype=np.float32),
        run=fake_run,
    )

    assert history["val_acc"] == [0.72, 0.83, 0.79]
    assert int(best_y_true[0]) == 1
    assert int(best_y_pred[0]) == 1
    assert summary["best_val_acc"] == 0.83
    assert summary["best_epoch"] == 2
    assert summary["final_val_acc"] == 0.79
    assert fake_run.logged[-1][0]["best_val_acc"] == 0.83
    assert fake_run.logged[-1][0]["best_epoch"] == 2
