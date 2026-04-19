from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou, nms, roi_align

from detection_minimal.data_provider.factory import get_base_dataset
from detection_minimal.utils.box_ops import clamp_boxes, decode_boxes, encode_boxes
from detection_minimal.utils.metrics import DetectionEvaluator
from detection_minimal.utils.visualization import draw_detection_overlay


def _device_of(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def resolve_local_yolo_weights(args, project_root: Path) -> Path | None:
    explicit = getattr(args, "yolo_weights", "")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    project_root = Path(project_root)
    candidates.extend(
        [
            project_root / "yolov8n.pt",
            project_root / "detection_minimal" / "yolov8n.pt",
            project_root / "detection_minimal" / "weights" / "yolov8n.pt",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


class BaseAdapter:
    external_trainer = False

    def __init__(self, args, num_classes: int):
        self.args = args
        self.num_classes = num_classes
        self.model = None

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def train_step(self, images, targets):
        raise NotImplementedError

    def val_step(self, images, targets):
        raise NotImplementedError

    def predict(self, images):
        raise NotImplementedError


class FasterRCNNAdapter(BaseAdapter):
    def __init__(self, args, num_classes: int):
        super().__init__(args, num_classes)
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if args.pretrained else None
        self.model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None if args.pretrained else None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    def train_step(self, images, targets):
        losses = self.model(images, targets)
        total_loss = sum(losses.values())
        loss_dict = {f"train/{key}": float(value.detach().cpu().item()) for key, value in losses.items()}
        loss_dict["train/loss"] = float(total_loss.detach().cpu().item())
        return total_loss, loss_dict

    def val_step(self, images, targets):
        self.model.train()
        with torch.no_grad():
            losses = self.model(images, targets)
            total_loss = sum(losses.values())
        self.model.eval()
        return float(total_loss.detach().cpu().item())

    def predict(self, images):
        self.model.eval()
        with torch.no_grad():
            return self.model(images)


class DetectionHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes + 1)
        self.bbox_pred = nn.Linear(1024, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.cls_score(x), self.bbox_pred(x)


class SelectiveSearchProposalGenerator:
    def __init__(self, cache_dir: Path, max_proposals: int, smoke_test: bool):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_proposals = max_proposals
        self.smoke_test = smoke_test

    def generate(self, image_tensor: torch.Tensor, image_key: str) -> torch.Tensor:
        cache_path = self.cache_dir / f"{hashlib.md5(image_key.encode('utf-8')).hexdigest()}.pt"
        if cache_path.exists():
            return torch.load(cache_path)

        image = (image_tensor.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        search.setBaseImage(image)
        search.switchToSelectiveSearchFast()
        rects = search.process()

        proposals = []
        max_props = 32 if self.smoke_test else self.max_proposals
        for x, y, w, h in rects[: max_props * 4]:
            if w < 8 or h < 8:
                continue
            proposals.append([float(x), float(y), float(x + w), float(y + h)])
            if len(proposals) >= max_props:
                break
        if not proposals:
            proposals = [[0.0, 0.0, float(image.shape[1] - 1), float(image.shape[0] - 1)]]
        tensor = torch.tensor(proposals, dtype=torch.float32)
        torch.save(tensor, cache_path)
        return tensor


class _ResNetBackbone(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.pool = backbone.avgpool
        self.out_channels = 2048

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        pooled = self.pool(feat)
        return pooled.flatten(1)


class _ClassicalRCNNBase(BaseAdapter):
    def __init__(self, args, num_classes: int, mode: str):
        super().__init__(args, num_classes)
        self.mode = mode
        self.backbone = _ResNetBackbone(pretrained=args.pretrained)
        self.head = DetectionHead(self.backbone.out_channels, num_classes)
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.SmoothL1Loss()
        self.model = nn.ModuleDict({"backbone": self.backbone, "head": self.head})
        self.proposal_generator = SelectiveSearchProposalGenerator(
            cache_dir=Path(args.output_root) / "proposal_cache" / f"{args.dataset}_{mode}",
            max_proposals=128,
            smoke_test=args.smoke_test,
        )

    def _sample_training_rois(self, proposals: torch.Tensor, target: dict):
        gt_boxes = target["boxes"].cpu()
        gt_labels = target["labels"].cpu()
        if len(gt_boxes):
            proposals = torch.cat([proposals, gt_boxes], dim=0)
            ious = box_iou(proposals, gt_boxes)
            max_iou, max_idx = ious.max(dim=1)
            labels = gt_labels[max_idx]
            labels[max_iou < 0.5] = 0
            keep_mask = (max_iou >= 0.5) | (max_iou < 0.3)
            proposals = proposals[keep_mask]
            labels = labels[keep_mask]
            matched_boxes = gt_boxes[max_idx[keep_mask]]
        else:
            labels = torch.zeros((len(proposals),), dtype=torch.int64)
            matched_boxes = proposals.clone()

        if len(proposals) == 0:
            proposals = torch.tensor([[0.0, 0.0, 16.0, 16.0]], dtype=torch.float32)
            labels = torch.zeros((1,), dtype=torch.int64)
            matched_boxes = proposals.clone()

        positive = torch.where(labels > 0)[0]
        negative = torch.where(labels == 0)[0]
        pos_limit = 8 if self.args.smoke_test else 16
        neg_limit = 24 if self.args.smoke_test else 48
        if len(positive) > pos_limit:
            positive = positive[:pos_limit]
        if len(negative) > neg_limit:
            negative = negative[:neg_limit]
        keep = torch.cat([positive, negative]) if len(positive) or len(negative) else torch.arange(len(proposals))
        return proposals[keep], labels[keep], matched_boxes[keep]

    def _extract_rcnn_features(self, image: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
        crops = roi_align(image.unsqueeze(0), [proposals], output_size=(224, 224), spatial_scale=1.0, aligned=True)
        return self.backbone.forward_pooled(crops)

    def _extract_fast_rcnn_features(self, image: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone.forward_map(image.unsqueeze(0))
        spatial_scale = feature_map.shape[-1] / image.shape[-1]
        pooled = roi_align(feature_map, [proposals], output_size=(7, 7), spatial_scale=spatial_scale, aligned=True)
        pooled = F.adaptive_avg_pool2d(pooled, 1).flatten(1)
        return pooled

    def _extract_features(self, image: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
        if self.mode == "rcnn":
            return self._extract_rcnn_features(image, proposals)
        return self._extract_fast_rcnn_features(image, proposals)

    def train_step(self, images, targets):
        device = _device_of(self.model)
        total_loss = torch.tensor(0.0, device=device)
        total_cls = 0.0
        total_reg = 0.0
        processed = 0
        for image, target in zip(images, targets):
            image_key = str(target["image_id"].item())
            proposals = self.proposal_generator.generate(image.detach().cpu(), image_key)
            sampled_props, labels, matched_boxes = self._sample_training_rois(proposals, target)
            sampled_props = sampled_props.to(device)
            labels = labels.to(device)
            matched_boxes = matched_boxes.to(device)
            image = image.to(device)

            features = self._extract_features(image, sampled_props)
            cls_logits, bbox_deltas = self.head(features)
            cls_loss = self.cls_loss(cls_logits, labels)
            positive_mask = labels > 0
            if positive_mask.any():
                reg_targets = encode_boxes(sampled_props[positive_mask], matched_boxes[positive_mask])
                reg_loss = self.reg_loss(bbox_deltas[positive_mask], reg_targets)
            else:
                reg_loss = torch.tensor(0.0, device=device)
            image_loss = cls_loss + reg_loss
            total_loss = total_loss + image_loss
            total_cls += float(cls_loss.detach().cpu().item())
            total_reg += float(reg_loss.detach().cpu().item())
            processed += 1

        total_loss = total_loss / max(processed, 1)
        return total_loss, {
            "train/cls_loss": total_cls / max(processed, 1),
            "train/reg_loss": total_reg / max(processed, 1),
            "train/loss": float(total_loss.detach().cpu().item()),
        }

    def val_step(self, images, targets):
        self.model.eval()
        with torch.no_grad():
            total = 0.0
            count = 0
            for image, target in zip(images, targets):
                image = image.to(_device_of(self.model))
                proposals = self.proposal_generator.generate(image.detach().cpu(), str(target["image_id"].item()))
                sampled_props, labels, matched_boxes = self._sample_training_rois(proposals, target)
                sampled_props = sampled_props.to(_device_of(self.model))
                labels = labels.to(_device_of(self.model))
                matched_boxes = matched_boxes.to(_device_of(self.model))
                features = self._extract_features(image, sampled_props)
                cls_logits, bbox_deltas = self.head(features)
                cls_loss = self.cls_loss(cls_logits, labels)
                positive_mask = labels > 0
                if positive_mask.any():
                    reg_targets = encode_boxes(sampled_props[positive_mask], matched_boxes[positive_mask])
                    reg_loss = self.reg_loss(bbox_deltas[positive_mask], reg_targets)
                else:
                    reg_loss = torch.tensor(0.0, device=_device_of(self.model))
                total += float((cls_loss + reg_loss).detach().cpu().item())
                count += 1
        return total / max(count, 1)

    def predict(self, images):
        device = _device_of(self.model)
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for image in images:
                image = image.to(device)
                proposals = self.proposal_generator.generate(image.detach().cpu(), f"pred_{hash(image.shape)}")
                proposals = proposals.to(device)
                features = self._extract_features(image, proposals)
                cls_logits, bbox_deltas = self.head(features)
                scores = torch.softmax(cls_logits, dim=1)
                pred_scores, pred_labels = scores[:, 1:].max(dim=1)
                pred_labels = pred_labels + 1
                boxes = decode_boxes(proposals, bbox_deltas)
                boxes = clamp_boxes(boxes, image.shape[1], image.shape[2])
                keep = pred_scores >= self.args.score_threshold
                if keep.any():
                    boxes = boxes[keep]
                    pred_labels = pred_labels[keep]
                    pred_scores = pred_scores[keep]
                    keep_nms = nms(boxes, pred_scores, 0.5)
                    boxes = boxes[keep_nms]
                    pred_labels = pred_labels[keep_nms]
                    pred_scores = pred_scores[keep_nms]
                else:
                    boxes = torch.zeros((0, 4), device=device)
                    pred_labels = torch.zeros((0,), dtype=torch.int64, device=device)
                    pred_scores = torch.zeros((0,), device=device)
                outputs.append({"boxes": boxes.cpu(), "labels": pred_labels.cpu(), "scores": pred_scores.cpu()})
        return outputs


class RCNNAdapter(_ClassicalRCNNBase):
    def __init__(self, args, num_classes: int):
        super().__init__(args, num_classes, mode="rcnn")


class FastRCNNAdapter(_ClassicalRCNNBase):
    def __init__(self, args, num_classes: int):
        super().__init__(args, num_classes, mode="fast_rcnn")


class YOLOv8Adapter(BaseAdapter):
    external_trainer = True

    def __init__(self, args, num_classes: int):
        super().__init__(args, num_classes)
        self.runtime_note = ""

    def _resolve_amp(self) -> bool:
        if self.args.amp is not None:
            return bool(self.args.amp)
        return str(self.args.device).startswith("cuda")

    def _resolve_cache_mode(self):
        return False if self.args.yolo_cache == "false" else self.args.yolo_cache

    def _build_yolo_model(self):
        from ultralytics import YOLO

        if not self.args.pretrained:
            return YOLO("yolov8n.yaml")
        project_root = Path(__file__).resolve().parents[2]
        local_weight_path = resolve_local_yolo_weights(self.args, project_root)
        if local_weight_path is not None:
            return YOLO(str(local_weight_path))
        try:
            return YOLO("yolov8n.pt")
        except Exception:
            if not self.args.smoke_test:
                searched = [
                    str(project_root / "yolov8n.pt"),
                    str(project_root / "detection_minimal" / "yolov8n.pt"),
                    str(project_root / "detection_minimal" / "weights" / "yolov8n.pt"),
                ]
                if getattr(self.args, "yolo_weights", ""):
                    searched.insert(0, str(Path(self.args.yolo_weights)))
                raise FileNotFoundError(
                    "Unable to load local YOLOv8 pretrained weights. "
                    f"Searched: {searched}"
                )
            self.runtime_note = "smoke_test used yolov8n.yaml because pretrained yolov8n.pt download was unreachable"
            return YOLO("yolov8n.yaml")

    def train_external(self, prepared_data, run_dirs, logger, summary):
        yolo_model = self._build_yolo_model()
        if self.runtime_note:
            summary["model_init_note"] = self.runtime_note
        if prepared_data.runtime_note:
            summary["runtime_note"] = prepared_data.runtime_note
        fraction = 1.0
        if self.args.smoke_test and not prepared_data.runtime_note:
            fraction = 0.02
        base_val_dataset = get_base_dataset(prepared_data.val_dataset)
        evaluator = DetectionEvaluator(prepared_data.meta.num_classes)

        def on_fit_epoch_end(trainer):
            if not trainer.csv.exists():
                return
            df = pd.read_csv(trainer.csv)
            if df.empty:
                return
            last = df.iloc[-1].to_dict()
            epoch = int(last["epoch"])
            metrics = {
                "train/loss": float(last.get("train/box_loss", 0.0) + last.get("train/cls_loss", 0.0) + last.get("train/dfl_loss", 0.0)),
                "val/loss": float(last.get("val/box_loss", 0.0) + last.get("val/cls_loss", 0.0) + last.get("val/dfl_loss", 0.0)),
                "mAP@0.5": float(last.get("metrics/mAP50(B)", 0.0)),
                "mAP@0.5:0.95": float(last.get("metrics/mAP50-95(B)", 0.0)),
                "Precision": float(last.get("metrics/precision(B)", 0.0)),
                "Recall": float(last.get("metrics/recall(B)", 0.0)),
                "lr": float(last.get("lr/pg0", 0.0)),
            }
            logger.log_scalars(epoch + 1, metrics)
            # Avoid calling predict() on the training model inside Ultralytics fit callbacks.
            # The predictor path wraps the model for inference and disables gradients, which
            # breaks the next training epoch with "tensor does not require grad" errors.

        yolo_model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        yolo_model.train(
            data=str(prepared_data.yolo_yaml_path),
            epochs=self.args.train_epochs,
            batch=self.args.batch_size,
            imgsz=self.args.imgsz,
            workers=self.args.num_workers,
            project=str(run_dirs.run_dir.parent.resolve()),
            name=run_dirs.run_dir.name,
            exist_ok=True,
            device=self.args.device,
            lr0=self.args.learning_rate,
            fraction=fraction,
            save=True,
            verbose=True,
            pretrained=self.args.pretrained and not self.runtime_note,
            resume=self.args.resume or False,
            amp=self._resolve_amp(),
            cache=self._resolve_cache_mode(),
        )

        predictions = []
        targets = []
        for image_index in prepared_data.val_visual_indices:
            image_path = base_val_dataset.get_image_path(image_index)
            result = yolo_model.predict(source=image_path, verbose=False, conf=self.args.score_threshold, save=False)[0]
            prediction = {
                "boxes": result.boxes.xyxy.cpu(),
                "labels": result.boxes.cls.cpu().to(torch.int64) + 1,
                "scores": result.boxes.conf.cpu(),
            }
            _, target = base_val_dataset[image_index]
            predictions.append(prediction)
            targets.append({k: v.cpu() for k, v in target.items()})
        artifacts = evaluator.evaluate(predictions, targets)
        logger.save_epoch_artifacts(summary, artifacts.confusion_matrix, artifacts.pr_curve)
        return artifacts.metrics


def build_adapter(args, num_classes: int) -> BaseAdapter:
    if args.model == "faster_rcnn":
        return FasterRCNNAdapter(args, num_classes)
    if args.model == "rcnn":
        return RCNNAdapter(args, num_classes)
    if args.model == "fast_rcnn":
        return FastRCNNAdapter(args, num_classes)
    if args.model == "yolov8":
        return YOLOv8Adapter(args, num_classes)
    raise ValueError(f"Unsupported model: {args.model}")
