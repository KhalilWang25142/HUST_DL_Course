import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import *
from datetime import datetime
import json

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

# 按batch_size划分为多个batch并通过yield调用
def iterate_batch(X, y, batch_size=64, shuffle=True):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]

def make_dir(path):
    os.makedirs(path, exist_ok=True)

def build_args():
    p = argparse.ArgumentParser()

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--activation", type=str, default="relu",
                   choices=["relu", "tanh", "sigmoid", "leaky_relu"])
    p.add_argument("--hidden_num1", type=int, default=256)
    p.add_argument("--hidden_num2", type=int, default=128)

    p.add_argument("--l2_lambda", type=float, default=0)

    p.add_argument("--seed", type=int, default=20260318)
    p.add_argument("--data_home", type=str, default="./scikit_data")
    p.add_argument("--log_dir", type=str, default="./runs/mnist_manual_numpy")
    p.add_argument("--save_dir", type=str, default="./outputs")
    p.add_argument("--method", type=str, default="He",
                   choices=["norm", "Xavier", "He"])
    p.add_argument("--dropout_rate", type=float, default=0.0)

    # 调节保存文件地址
    args = p.parse_args()

    # 系统时间
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 避免路径里出现小数点
    dropout_str = str(args.dropout_rate).replace(".", "p")

    # activation / method / dropout_rate / time
    suffix = os.path.join(
        args.activation,
        args.method,
        f"dropout_{dropout_str}",
        time_str
    )

    args.log_dir = os.path.join(args.log_dir, suffix)
    args.save_dir = os.path.join(args.save_dir, suffix)

    return args

def save_args_to_json(args, filename="args.json"):
    os.makedirs(args.log_dir, exist_ok=True)
    save_path = os.path.join(args.log_dir, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

def create_summary_writer(log_dir):
    make_dir(log_dir)
    if SummaryWriter is None:
        print("Warning: tensorboardX not found, TensorBoard logging skipped.")
        return None
    return SummaryWriter(log_dir=log_dir)

def close_summary_writer(writer):
    if writer is not None:
        writer.flush()
        writer.close()

def plot_training_history(history, save_path):
    make_dir(os.path.dirname(save_path))

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
    make_dir(os.path.dirname(save_path))

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
    make_dir(os.path.dirname(save_path))

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