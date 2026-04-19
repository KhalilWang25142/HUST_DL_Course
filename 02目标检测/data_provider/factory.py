from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from detection_minimal.data_provider.common import detection_collate_fn
from detection_minimal.data_provider.datasets import COCO128DetectionDataset, VOC2007DetectionDataset
from detection_minimal.data_provider.downloads import (
    create_coco128_smoke_fallback,
    ensure_coco128_dataset,
    ensure_voc2007_dataset,
    export_voc_to_yolo,
)
from detection_minimal.data_provider.metadata import DatasetMeta, get_dataset_meta
from detection_minimal.data_provider.yolo_config import build_yolo_data_config

DEFAULT_DATA_ROOT = Path("./detection_minimal/dataset")


@dataclass(slots=True)
class PreparedData:
    meta: DatasetMeta
    train_dataset: torch.utils.data.Dataset
    val_dataset: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    yolo_yaml_path: Path
    val_visual_indices: list[int]
    runtime_note: str = ""


def _apply_smoke_subset(dataset, enabled: bool, limit: int):
    if not enabled:
        return dataset
    limit = min(len(dataset), limit)
    return Subset(dataset, list(range(limit)))


def _visual_indices(dataset, count: int) -> list[int]:
    if isinstance(dataset, Subset):
        return dataset.indices[: min(count, len(dataset.indices))]
    return list(range(min(count, len(dataset))))


def _resolve_original_dataset(dataset):
    if isinstance(dataset, Subset):
        return dataset.dataset
    return dataset


def resolve_data_root(configured_root: Path | str, dataset_name: str, project_root: Path | None = None) -> Path:
    project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    configured_root = Path(configured_root)

    if configured_root.is_absolute():
        return configured_root

    project_level_root = project_root
    package_level_root = project_root / "detection_minimal" / "dataset"

    if configured_root == DEFAULT_DATA_ROOT:
        if (project_level_root / dataset_name).exists():
            return project_level_root
        if (package_level_root / dataset_name).exists():
            return package_level_root
        return package_level_root

    if (configured_root / dataset_name).exists():
        return configured_root

    project_relative_root = project_root / configured_root
    if (project_relative_root / dataset_name).exists():
        return project_relative_root

    return project_relative_root


def prepare_data(args, output_root: Path) -> PreparedData:
    data_root = resolve_data_root(args.data_root, args.dataset)
    meta = get_dataset_meta(args.dataset)

    if args.dataset == "voc2007":
        voc_root = ensure_voc2007_dataset(data_root)
        train_dataset = VOC2007DetectionDataset(voc_root, "train", meta.class_to_idx)
        val_dataset = VOC2007DetectionDataset(voc_root, "val", meta.class_to_idx)
        test_dataset = VOC2007DetectionDataset(voc_root, "test", meta.class_to_idx)
        yolo_root = export_voc_to_yolo(voc_root, data_root / "voc2007_yolo")
        yolo_yaml_path = build_yolo_data_config(
            output_dir=output_root / "yolo_data",
            dataset_name="voc2007",
            root_path=yolo_root,
            train_path="images/train",
            val_path="images/val",
            class_names=meta.class_names[1:],
        )
    else:
        runtime_note = ""
        try:
            coco_root = ensure_coco128_dataset(data_root)
        except Exception:
            if not args.smoke_test:
                raise
            voc_root = ensure_voc2007_dataset(data_root)
            coco_root = create_coco128_smoke_fallback(data_root, voc_root, limit=8)
            runtime_note = "smoke_test fallback dataset used because official coco128 download was unreachable"
        train_dataset = COCO128DetectionDataset(coco_root, "train")
        val_dataset = COCO128DetectionDataset(coco_root, "val")
        test_dataset = COCO128DetectionDataset(coco_root, "val")
        yolo_yaml_path = build_yolo_data_config(
            output_dir=output_root / "yolo_data",
            dataset_name="coco128",
            root_path=coco_root,
            train_path="images/train2017",
            val_path="images/train2017",
            class_names=meta.class_names[1:],
        )
    if args.dataset == "voc2007":
        runtime_note = ""

    smoke_limit = 4 if args.smoke_test else None
    train_dataset = _apply_smoke_subset(train_dataset, args.smoke_test, smoke_limit or len(train_dataset))
    val_dataset = _apply_smoke_subset(val_dataset, args.smoke_test, smoke_limit or len(val_dataset))
    test_dataset = _apply_smoke_subset(test_dataset, args.smoke_test, smoke_limit or len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )

    return PreparedData(
        meta=meta,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        yolo_yaml_path=yolo_yaml_path,
        val_visual_indices=_visual_indices(val_dataset, args.save_vis_count),
        runtime_note=runtime_note,
    )


def get_base_dataset(dataset):
    return _resolve_original_dataset(dataset)
