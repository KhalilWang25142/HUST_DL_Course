from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.patches import Rectangle


def _tensor_to_numpy_boxes(boxes: torch.Tensor | None) -> np.ndarray:
    if boxes is None:
        return np.zeros((0, 4), dtype=np.float32)
    if isinstance(boxes, torch.Tensor):
        return boxes.detach().cpu().numpy()
    return np.asarray(boxes)


def _tensor_to_numpy_labels(labels: torch.Tensor | None) -> np.ndarray:
    if labels is None:
        return np.zeros((0,), dtype=np.int64)
    if isinstance(labels, torch.Tensor):
        return labels.detach().cpu().numpy()
    return np.asarray(labels)


def draw_detection_overlay(
    image: Image.Image,
    target: dict,
    prediction: dict,
    class_names: list[str],
    save_path: Path,
) -> np.ndarray:
    image_array = np.asarray(image.convert("RGB"))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_array)
    ax.axis("off")

    for box, label in zip(_tensor_to_numpy_boxes(target.get("boxes")), _tensor_to_numpy_labels(target.get("labels"))):
        x1, y1, x2, y2 = box.tolist()
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2))
        ax.text(x1, y1, class_names[int(label)], color="lime", fontsize=8, bbox={"facecolor": "black", "alpha": 0.5})

    pred_scores = prediction.get("scores")
    score_values = pred_scores.detach().cpu().numpy() if isinstance(pred_scores, torch.Tensor) else pred_scores
    if score_values is None:
        score_values = [None] * len(_tensor_to_numpy_boxes(prediction.get("boxes")))

    for box, label, score in zip(
        _tensor_to_numpy_boxes(prediction.get("boxes")),
        _tensor_to_numpy_labels(prediction.get("labels")),
        score_values,
    ):
        x1, y1, x2, y2 = box.tolist()
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2))
        title = class_names[int(label)]
        if score is not None:
            title = f"{title}:{float(score):.2f}"
        ax.text(x1, y2, title, color="red", fontsize=8, bbox={"facecolor": "white", "alpha": 0.6})

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rendered = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]
    plt.close(fig)
    return rendered
