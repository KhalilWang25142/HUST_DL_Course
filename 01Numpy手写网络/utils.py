import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def onehot(y, num_classes=10):
    y = np.asarray(y, dtype=np.int64)
    return np.eye(num_classes, dtype=np.float32)[y]


def accuracy(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)

    return float(np.mean(y_pred == y_true))


def iterate_minibatches(X, y, batch_size=64, shuffle=True):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_seed(seed=20260318):
    np.random.seed(seed)


def build_args():
    p = argparse.ArgumentParser()

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)   # 手写 batch_size 超参数
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--activation", type=str, default="relu",
                   choices=["relu", "tanh", "sigmoid", "leaky_relu"])

    # L2 系数超参数（具体使用统一在 loss.py 中）
    p.add_argument("--l2_lambda", type=float, default=1e-4)

    p.add_argument("--seed", type=int, default=20260318)
    p.add_argument("--data_home", type=str, default="./scikit_data")
    p.add_argument("--log_dir", type=str, default="./runs/mnist_manual_numpy")
    p.add_argument("--save_dir", type=str, default="./outputs")

    return p.parse_args()


def create_summary_writer(log_dir):
    ensure_dir(log_dir)
    if SummaryWriter is None:
        print("Warning: tensorboardX not found, TensorBoard logging skipped.")
        return None
    return SummaryWriter(log_dir=log_dir)


def close_summary_writer(writer):
    if writer is not None:
        writer.flush()
        writer.close()


def plot_training_history(history, save_path):
    ensure_dir(os.path.dirname(save_path))

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_confusion_matrix(y_true, y_pred, save_path, class_names=None):
    ensure_dir(os.path.dirname(save_path))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_misclassified_examples(X, y_true, y_pred, save_path, max_show=25):
    ensure_dir(os.path.dirname(save_path))

    X = np.asarray(X)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim > 1:
        y_true_label = np.argmax(y_true, axis=1)
    else:
        y_true_label = y_true

    if y_pred.ndim > 1:
        y_pred_label = np.argmax(y_pred, axis=1)
    else:
        y_pred_label = y_pred

    wrong_idx = np.where(y_true_label != y_pred_label)[0]

    if len(wrong_idx) == 0:
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "No misclassified examples.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        return

    wrong_idx = wrong_idx[:max_show]

    cols = 5
    rows = int(np.ceil(len(wrong_idx) / cols))

    plt.figure(figsize=(cols * 2.5, rows * 2.5))
    for i, idx in enumerate(wrong_idx):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(X[idx].reshape(28, 28), cmap="gray")
        plt.title(f"T:{y_true_label[idx]} P:{y_pred_label[idx]}", fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()