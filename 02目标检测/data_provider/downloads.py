from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from torchvision.datasets import VOCDetection

from detection_minimal.data_provider.metadata import get_dataset_meta


COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
VOC_TO_COCO_NAME = {
    "aeroplane": "airplane",
    "motorbike": "motorcycle",
    "sofa": "couch",
    "tvmonitor": "tv",
    "diningtable": "dining table",
    "pottedplant": "potted plant",
}


def normalize_voc_name_for_coco(name: str) -> str:
    return VOC_TO_COCO_NAME.get(name, name)


def ensure_voc2007_dataset(root: Path) -> Path:
    root = Path(root)
    voc_root = root / "voc2007"
    VOCDetection(root=str(voc_root), year="2007", image_set="train", download=True)
    VOCDetection(root=str(voc_root), year="2007", image_set="val", download=True)
    VOCDetection(root=str(voc_root), year="2007", image_set="test", download=True)
    return voc_root


def ensure_coco128_dataset(root: Path) -> Path:
    root = Path(root)
    dataset_root = root / "coco128"
    image_dir = dataset_root / "images" / "train2017"
    label_dir = dataset_root / "labels" / "train2017"
    if image_dir.exists() and label_dir.exists():
        return dataset_root

    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "coco128.zip"
    urllib.request.urlretrieve(COCO128_URL, archive_path)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(root)
    return dataset_root


def create_coco128_smoke_fallback(root: Path, voc_root: Path, limit: int = 8) -> Path:
    root = Path(root)
    fallback_root = root / "coco128_smoke_fallback"
    image_dir = fallback_root / "images" / "train2017"
    label_dir = fallback_root / "labels" / "train2017"
    if image_dir.exists() and any(image_dir.glob("*.jpg")):
        return fallback_root

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    jpeg_dir = voc_root / "VOCdevkit" / "VOC2007" / "JPEGImages"
    annotations_dir = voc_root / "VOCdevkit" / "VOC2007" / "Annotations"
    image_ids = [line.strip() for line in (voc_root / "VOCdevkit" / "VOC2007" / "ImageSets" / "Main" / "train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    coco_class_to_idx = get_dataset_meta("coco128").class_to_idx

    for image_id in image_ids[:limit]:
        shutil.copy2(jpeg_dir / f"{image_id}.jpg", image_dir / f"{image_id}.jpg")
        xml_root = ElementTree.parse(annotations_dir / f"{image_id}.xml").getroot()
        width = float(xml_root.findtext("size/width"))
        height = float(xml_root.findtext("size/height"))
        label_lines = []
        for obj in xml_root.findall("object"):
            coco_name = normalize_voc_name_for_coco(obj.findtext("name"))
            if coco_name not in coco_class_to_idx:
                continue
            bbox = obj.find("bndbox")
            xmin = float(bbox.findtext("xmin"))
            ymin = float(bbox.findtext("ymin"))
            xmax = float(bbox.findtext("xmax"))
            ymax = float(bbox.findtext("ymax"))
            x_center = ((xmin + xmax) / 2.0) / width
            y_center = ((ymin + ymax) / 2.0) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height
            label_lines.append(
                f"{coco_class_to_idx[coco_name] - 1} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )
        (label_dir / f"{image_id}.txt").write_text("\n".join(label_lines), encoding="utf-8")
    return fallback_root


def export_voc_to_yolo(voc_root: Path, export_root: Path) -> Path:
    voc_root = Path(voc_root)
    export_root = Path(export_root)
    jpeg_dir = voc_root / "VOCdevkit" / "VOC2007" / "JPEGImages"
    annotations_dir = voc_root / "VOCdevkit" / "VOC2007" / "Annotations"
    sets_dir = voc_root / "VOCdevkit" / "VOC2007" / "ImageSets" / "Main"
    class_to_idx = get_dataset_meta("voc2007").class_to_idx

    for split in ["train", "val", "test"]:
        image_list_path = sets_dir / f"{split}.txt"
        dest_image_dir = export_root / "images" / split
        dest_label_dir = export_root / "labels" / split
        dest_image_dir.mkdir(parents=True, exist_ok=True)
        dest_label_dir.mkdir(parents=True, exist_ok=True)
        if not image_list_path.exists():
            continue
        image_ids = [line.strip() for line in image_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for image_id in image_ids:
            src_image = jpeg_dir / f"{image_id}.jpg"
            dst_image = dest_image_dir / f"{image_id}.jpg"
            if not dst_image.exists():
                shutil.copy2(src_image, dst_image)
            label_path = dest_label_dir / f"{image_id}.txt"
            if label_path.exists():
                continue
            label_lines = []
            xml_root = ElementTree.parse(annotations_dir / f"{image_id}.xml").getroot()
            width = float(xml_root.findtext("size/width"))
            height = float(xml_root.findtext("size/height"))
            for obj in xml_root.findall("object"):
                cls_name = obj.findtext("name")
                if cls_name not in class_to_idx:
                    continue
                bbox = obj.find("bndbox")
                xmin = float(bbox.findtext("xmin"))
                ymin = float(bbox.findtext("ymin"))
                xmax = float(bbox.findtext("xmax"))
                ymax = float(bbox.findtext("ymax"))
                x_center = ((xmin + xmax) / 2.0) / width
                y_center = ((ymin + ymax) / 2.0) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height
                cls_idx = class_to_idx[cls_name] - 1
                label_lines.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
            label_path.write_text("\n".join(label_lines), encoding="utf-8")
    return export_root
