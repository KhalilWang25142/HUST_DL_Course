from argparse import Namespace
from datetime import datetime
from pathlib import Path
import sys
import types

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import utils


def test_init_wandb_run_respects_requested_mode(tmp_path, monkeypatch):
    calls = {}
    fake_run = object()

    def fake_init(**kwargs):
        calls.update(kwargs)
        return fake_run

    monkeypatch.setattr(utils, "wandb", types.SimpleNamespace(init=fake_init))

    args = Namespace(
        log_dir=str(tmp_path / "runs"),
        save_dir=str(tmp_path / "outputs"),
        run_suffix="relu/He/dropout_0p0/20260414_000000_000001",
        epochs=2,
        batch_size=4,
        lr=0.02,
        activation="relu",
        hidden_num1=256,
        hidden_num2=128,
        l2_lambda=0.0,
        seed=20260318,
        data_home="./scikit_data",
        method="He",
        dropout_rate=0.0,
        wandb_mode="online",
    )

    run = utils.init_wandb_run(args)

    assert run is fake_run
    assert calls["mode"] == "online"
    assert calls["dir"] == args.log_dir
    assert calls["config"]["epochs"] == 2


def test_finalize_run_paths_creates_unique_dirs(monkeypatch):
    timestamps = iter(
        [
            datetime(2026, 4, 14, 21, 25, 12, 1),
            datetime(2026, 4, 14, 21, 25, 12, 2),
        ]
    )

    class FakeDateTime:
        @classmethod
        def now(cls):
            return next(timestamps)

    monkeypatch.setattr(utils, "datetime", FakeDateTime)

    args1 = Namespace(
        activation="relu",
        method="He",
        dropout_rate=0.0,
        log_dir="./runs/mnist_manual_numpy",
        save_dir="./outputs",
    )
    args2 = Namespace(
        activation="relu",
        method="He",
        dropout_rate=0.0,
        log_dir="./runs/mnist_manual_numpy",
        save_dir="./outputs",
    )

    finalized1 = utils.finalize_run_paths(args1)
    finalized2 = utils.finalize_run_paths(args2)

    assert finalized1.log_dir != finalized2.log_dir
    assert finalized1.save_dir != finalized2.save_dir
    assert finalized1.run_suffix.endswith("20260414_212512_000001")
    assert finalized2.run_suffix.endswith("20260414_212512_000002")


def test_build_wandb_image_payload_uses_first_samples_and_captions(monkeypatch):
    captured = []

    class FakeImage:
        def __init__(self, image, caption):
            self.image = image
            self.caption = caption
            captured.append(self)

    monkeypatch.setattr(utils, "wandb", types.SimpleNamespace(Image=FakeImage))

    X = np.arange(12 * 784, dtype=np.float32).reshape(12, 784)
    y_true = np.eye(10, dtype=np.float32)[np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1])]
    y_pred = np.eye(10, dtype=np.float32)[np.array([0, 9, 2, 3, 4, 5, 0, 7, 8, 1, 0, 1])]

    images = utils.build_wandb_image_payload(X, y_true, y_pred, max_show=3)

    assert len(images) == 3
    assert captured[0].image.shape == (28, 28)
    assert "idx=0" in captured[0].caption
    assert "true=0" in captured[0].caption
    assert "pred=0" in captured[0].caption
    assert "correct=True" in captured[0].caption
    assert "correct=False" in captured[1].caption
