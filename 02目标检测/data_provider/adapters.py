from typing import Iterable

import torch


def _empty_target(image_size: tuple[int, int], image_id: int) -> dict[str, torch.Tensor]:
    height, width = image_size
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "orig_size": torch.tensor([height, width], dtype=torch.int64),
    }


def convert_voc_annotation(
    annotation: dict,
    class_to_idx: dict[str, int],
    image_id: int,
) -> dict[str, torch.Tensor]:
    ann = annotation["annotation"]
    width = int(ann["size"]["width"])
    height = int(ann["size"]["height"])
    objects = ann.get("object", [])
    if isinstance(objects, dict):
        objects = [objects]
    if not objects:
        return _empty_target((height, width), image_id)

    boxes = []
    labels = []
    for item in objects:
        name = item["name"]
        bbox = item["bndbox"]
        boxes.append(
            [
                float(bbox["xmin"]),
                float(bbox["ymin"]),
                float(bbox["xmax"]),
                float(bbox["ymax"]),
            ]
        )
        labels.append(class_to_idx[name])

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "orig_size": torch.tensor([height, width], dtype=torch.int64),
    }


def convert_yolo_lines_to_target(
    lines: Iterable[str],
    image_size: tuple[int, int],
    image_id: int,
) -> dict[str, torch.Tensor]:
    height, width = image_size
    lines = [line.strip() for line in lines if line.strip()]
    if not lines:
        return _empty_target((height, width), image_id)

    boxes = []
    labels = []
    for line in lines:
        cls_id, x_center, y_center, box_width, box_height = map(float, line.split())
        abs_width = box_width * width
        abs_height = box_height * height
        center_x = x_center * width
        center_y = y_center * height
        xmin = center_x - abs_width / 2.0
        ymin = center_y - abs_height / 2.0
        xmax = center_x + abs_width / 2.0
        ymax = center_y + abs_height / 2.0
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(int(cls_id) + 1)

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "orig_size": torch.tensor([height, width], dtype=torch.int64),
    }

