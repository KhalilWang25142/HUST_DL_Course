from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as F

from detection_minimal.data_provider.adapters import convert_voc_annotation, convert_yolo_lines_to_target


class VOC2007DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, split: str, class_to_idx: dict[str, int]):
        self.dataset = VOCDetection(root=str(root), year="2007", image_set=split, download=False)
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, annotation = self.dataset[index]
        target = convert_voc_annotation(annotation, self.class_to_idx, image_id=index)
        return F.to_tensor(image), target

    def get_pil_image(self, index: int) -> Image.Image:
        image, _ = self.dataset[index]
        return image.convert("RGB")

    def get_image_path(self, index: int) -> str:
        return str(self.dataset.images[index])


class COCO128DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, split: str):
        self.root = Path(root)
        self.split = split
        image_dir = self.root / "images" / "train2017"
        label_dir = self.root / "labels" / "train2017"
        self.image_paths = sorted(image_dir.glob("*.jpg"))
        self.label_dir = label_dir

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        label_path = self.label_dir / f"{image_path.stem}.txt"
        lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
        target = convert_yolo_lines_to_target(lines, image_size=(image.height, image.width), image_id=index)
        return F.to_tensor(image), target

    def get_pil_image(self, index: int) -> Image.Image:
        return Image.open(self.image_paths[index]).convert("RGB")

    def get_image_path(self, index: int) -> str:
        return str(self.image_paths[index])

