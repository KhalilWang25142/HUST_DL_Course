from __future__ import annotations

import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from detection_minimal.data_provider.factory import get_base_dataset, prepare_data
from detection_minimal.models.adapters import build_adapter
from detection_minimal.utils.experiment import build_run_name, build_run_summary, prepare_run_directories
from detection_minimal.utils.logging import ExperimentLogger
from detection_minimal.utils.metrics import DetectionEvaluator
from detection_minimal.utils.visualization import draw_detection_overlay
from detection_minimal.exp.exp_basic import ExpBasic


def move_prediction_batch_to_cpu(predictions: list[dict]) -> list[dict]:
    converted = []
    for prediction in predictions:
        converted.append(
            {
                key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                for key, value in prediction.items()
            }
        )
    return converted


class ExpDetection(ExpBasic):
    def __init__(self, args):
        super().__init__(args)
        self.device = self._resolve_device(args.device)
        self.args.device = self.device.type if self.device.index is None else f"{self.device.type}:{self.device.index}"
        self._set_seed(args.seed)

    def run(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = build_run_name(self.args.dataset, self.args.model, timestamp)
        run_dirs = prepare_run_directories(Path(self.args.output_root), run_name)
        prepared = prepare_data(self.args, run_dirs.run_dir)
        summary = build_run_summary(
            args=self.args,
            run_name=run_name,
            num_classes=prepared.meta.num_classes,
            split_note=prepared.meta.split_note,
        )
        if prepared.runtime_note:
            summary["runtime_note"] = prepared.runtime_note
        logger = ExperimentLogger(run_dirs)

        try:
            adapter = build_adapter(self.args, prepared.meta.num_classes)
            if adapter.external_trainer:
                adapter.train_external(prepared, run_dirs, logger, summary)
                return

            adapter.model.to(self.device)
            optimizer = torch.optim.Adam(adapter.parameters(), lr=self.args.learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, self.args.train_epochs))

            start_epoch = 1
            best_map = -1.0
            if self.args.resume:
                start_epoch, best_map = self._load_checkpoint(adapter, optimizer, Path(self.args.resume))

            evaluator = DetectionEvaluator(prepared.meta.num_classes)
            for epoch in range(start_epoch, self.args.train_epochs + 1):
                train_metrics = self._train_one_epoch(adapter, optimizer, prepared.train_loader)
                val_loss, predictions, targets = self._validate_one_epoch(adapter, prepared.val_loader)
                metric_artifacts = evaluator.evaluate(predictions, targets)
                peak_memory = self._read_peak_memory()
                metrics = {
                    **train_metrics,
                    "val/loss": val_loss,
                    **metric_artifacts.metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                    "time/epoch_seconds": train_metrics.pop("_epoch_seconds"),
                    "memory/peak_mb": peak_memory,
                }
                logger.log_scalars(epoch, metrics)
                rendered_images = self._render_visuals(adapter, prepared, run_dirs, epoch)
                logger.log_images(epoch, rendered_images)
                logger.save_epoch_artifacts(summary, metric_artifacts.confusion_matrix, metric_artifacts.pr_curve)
                self._save_checkpoint(run_dirs, adapter, optimizer, epoch, best=False, metrics=metrics)
                if metric_artifacts.metrics["mAP@0.5:0.95"] >= best_map:
                    best_map = metric_artifacts.metrics["mAP@0.5:0.95"]
                    self._save_checkpoint(run_dirs, adapter, optimizer, epoch, best=True, metrics=metrics)
                scheduler.step()
        finally:
            logger.close()

    def _train_one_epoch(self, adapter, optimizer, dataloader):
        adapter.train()
        start_time = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        aggregates = {}
        steps = 0
        for images, targets in dataloader:
            images = [image.to(self.device) for image in images]
            targets = [{key: value.to(self.device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            total_loss, loss_dict = adapter.train_step(images, targets)
            total_loss.backward()
            optimizer.step()
            for key, value in loss_dict.items():
                aggregates[key] = aggregates.get(key, 0.0) + float(value)
            steps += 1
        metrics = {key: value / max(steps, 1) for key, value in aggregates.items()}
        metrics["_epoch_seconds"] = time.time() - start_time
        return metrics

    def _validate_one_epoch(self, adapter, dataloader):
        adapter.eval()
        total_loss = 0.0
        steps = 0
        predictions = []
        targets_out = []
        for images, targets in dataloader:
            images = [image.to(self.device) for image in images]
            targets = [{key: value.to(self.device) for key, value in target.items()} for target in targets]
            total_loss += adapter.val_step(images, targets)
            batch_predictions = move_prediction_batch_to_cpu(adapter.predict(images))
            predictions.extend(batch_predictions)
            targets_out.extend([{key: value.detach().cpu() for key, value in target.items()} for target in targets])
            steps += 1
        return total_loss / max(steps, 1), predictions, targets_out

    def _render_visuals(self, adapter, prepared, run_dirs, epoch: int):
        base_val_dataset = get_base_dataset(prepared.val_dataset)
        rendered = []
        for image_index in prepared.val_visual_indices:
            pil_image = base_val_dataset.get_pil_image(image_index)
            image_tensor, target = base_val_dataset[image_index]
            prediction = adapter.predict([image_tensor.to(self.device)])[0]
            rendered_image = draw_detection_overlay(
                image=pil_image,
                target={key: value.cpu() for key, value in target.items()},
                prediction=prediction,
                class_names=prepared.meta.class_names,
                save_path=run_dirs.images_dir / f"epoch_{epoch}" / f"sample_{image_index}.png",
            )
            rendered.append(rendered_image)
        return rendered

    def _save_checkpoint(self, run_dirs, adapter, optimizer, epoch: int, best: bool, metrics: dict):
        name = "best.pth" if best else "last.pth"
        payload = {
            "epoch": epoch,
            "model_state": adapter.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        }
        torch.save(payload, run_dirs.checkpoints_dir / name)

    def _load_checkpoint(self, adapter, optimizer, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        adapter.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        metrics = checkpoint.get("metrics", {})
        return int(checkpoint["epoch"]) + 1, float(metrics.get("mAP@0.5:0.95", -1.0))

    def _resolve_device(self, device_name: str) -> torch.device:
        if device_name.startswith("cuda") and torch.cuda.is_available():
            return torch.device(device_name)
        if device_name == "cpu":
            return torch.device("cpu")
        if device_name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _read_peak_memory(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
