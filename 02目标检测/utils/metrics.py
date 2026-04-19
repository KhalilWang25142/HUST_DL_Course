from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou


@dataclass(slots=True)
class MetricArtifacts:
    metrics: dict[str, float]
    confusion_matrix: np.ndarray
    pr_curve: dict[str, list[float]]


class DetectionEvaluator:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def evaluate(self, predictions: list[dict], targets: list[dict]) -> MetricArtifacts:
        metric = MeanAveragePrecision(box_format="xyxy", class_metrics=False)
        metric.update(predictions, targets)
        metric_values = metric.compute()

        precision, recall, confusion_matrix, pr_curve = self._precision_recall_confusion(predictions, targets)
        metrics = {
            "mAP@0.5": float(metric_values["map_50"].item() if torch.is_tensor(metric_values["map_50"]) else metric_values["map_50"]),
            "mAP@0.5:0.95": float(metric_values["map"].item() if torch.is_tensor(metric_values["map"]) else metric_values["map"]),
            "Precision": precision,
            "Recall": recall,
        }
        return MetricArtifacts(metrics=metrics, confusion_matrix=confusion_matrix, pr_curve=pr_curve)

    def _precision_recall_confusion(self, predictions: list[dict], targets: list[dict]):
        tp = 0
        fp = 0
        fn = 0
        confusion = np.zeros((self.num_classes + 1, self.num_classes + 1), dtype=np.int64)
        score_targets = []
        score_values = []

        for pred, target in zip(predictions, targets):
            pred_boxes = pred["boxes"].detach().cpu()
            pred_labels = pred["labels"].detach().cpu()
            pred_scores = pred.get("scores", torch.ones(len(pred_boxes))).detach().cpu()
            gt_boxes = target["boxes"].detach().cpu()
            gt_labels = target["labels"].detach().cpu()

            matched_pred = set()
            if len(pred_boxes) and len(gt_boxes):
                ious = box_iou(pred_boxes, gt_boxes)
            else:
                ious = torch.zeros((len(pred_boxes), len(gt_boxes)))

            for gt_idx, gt_label in enumerate(gt_labels.tolist()):
                best_iou = 0.0
                best_pred_idx = -1
                for pred_idx in range(len(pred_boxes)):
                    if pred_idx in matched_pred:
                        continue
                    iou = float(ious[pred_idx, gt_idx]) if ious.numel() else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = pred_idx
                if best_pred_idx >= 0 and best_iou >= 0.5:
                    matched_pred.add(best_pred_idx)
                    pred_label = int(pred_labels[best_pred_idx].item())
                    confusion[gt_label, pred_label] += 1
                    score_values.append(float(pred_scores[best_pred_idx].item()))
                    score_targets.append(1 if pred_label == gt_label else 0)
                    if pred_label == gt_label:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                else:
                    confusion[gt_label, 0] += 1
                    fn += 1

            for pred_idx, pred_label in enumerate(pred_labels.tolist()):
                if pred_idx in matched_pred:
                    continue
                confusion[0, pred_label] += 1
                score_values.append(float(pred_scores[pred_idx].item()))
                score_targets.append(0)
                fp += 1

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if score_values:
            pr_precision, pr_recall, _ = precision_recall_curve(score_targets, score_values)
            pr_curve = {"precision": pr_precision.tolist(), "recall": pr_recall.tolist()}
        else:
            pr_curve = {"precision": [1.0], "recall": [0.0]}
        return precision, recall, confusion, pr_curve

