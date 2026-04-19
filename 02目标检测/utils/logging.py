from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from detection_minimal.utils.io import save_csv, save_json, save_jsonl


class ExperimentLogger:
    def __init__(self, run_dirs):
        self.run_dirs = run_dirs
        self.writer = SummaryWriter(log_dir=str(run_dirs.tb_dir))
        self.history: list[dict] = []
        self.start_time = time.time()

    def log_scalars(self, epoch: int, metrics: dict[str, float]) -> None:
        row = {"epoch": epoch}
        row.update(metrics)
        self.history.append(row)
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, epoch)

    def log_images(self, epoch: int, rendered_images: list[np.ndarray]) -> None:
        for index, image in enumerate(rendered_images):
            chw = np.transpose(image, (2, 0, 1))
            self.writer.add_image(f"val_predictions/epoch_{epoch}/sample_{index}", chw, epoch)

    def save_epoch_artifacts(self, summary: dict, confusion_matrix: np.ndarray, pr_curve: dict[str, list[float]]) -> None:
        save_json(summary, self.run_dirs.metrics_dir / "run_summary.json")
        save_jsonl(self.history, self.run_dirs.metrics_dir / "metrics_history.jsonl")
        save_csv(self.history, self.run_dirs.metrics_dir / "metrics_history.csv")
        save_json({"matrix": confusion_matrix.tolist()}, self.run_dirs.metrics_dir / "confusion_matrix.json")
        save_json(pr_curve, self.run_dirs.metrics_dir / "pr_curve.json")
        self._plot_history()
        self._plot_confusion_matrix(confusion_matrix)
        self._plot_pr_curve(pr_curve)

    def _plot_history(self) -> None:
        if not self.history:
            return
        keys = [key for key in self.history[0].keys() if key != "epoch"]
        for key in keys:
            plt.figure(figsize=(6, 4))
            plt.plot([row["epoch"] for row in self.history], [row[key] for row in self.history], marker="o")
            plt.title(key)
            plt.xlabel("epoch")
            plt.ylabel(key)
            plt.tight_layout()
            path = self.run_dirs.plots_dir / f"{key.replace('/', '_')}.png"
            plt.savefig(path, dpi=200)
            plt.close()

    def _plot_confusion_matrix(self, confusion_matrix: np.ndarray) -> None:
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix, cmap="Blues")
        plt.colorbar()
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(self.run_dirs.plots_dir / "confusion_matrix.png", dpi=200)
        plt.close()

    def _plot_pr_curve(self, pr_curve: dict[str, list[float]]) -> None:
        plt.figure(figsize=(6, 4))
        plt.plot(pr_curve["recall"], pr_curve["precision"])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve")
        plt.tight_layout()
        plt.savefig(self.run_dirs.plots_dir / "pr_curve.png", dpi=200)
        plt.close()

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
