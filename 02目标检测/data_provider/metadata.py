from __future__ import annotations

from dataclasses import dataclass


VOC_CLASSES = [
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

COCO_CLASSES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


@dataclass(slots=True)
class DatasetMeta:
    name: str
    class_names: list[str]
    split_note: str

    @property
    def num_classes(self) -> int:
        return len(self.class_names) - 1

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.class_names) if idx > 0}


def build_split_note(dataset_name: str) -> str:
    if dataset_name == "coco128":
        return "official coco128 train/val share the same images"
    if dataset_name == "voc2007":
        return "official VOC2007 splits"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_dataset_meta(dataset_name: str) -> DatasetMeta:
    if dataset_name == "voc2007":
        return DatasetMeta(name="voc2007", class_names=VOC_CLASSES, split_note=build_split_note("voc2007"))
    if dataset_name == "coco128":
        return DatasetMeta(name="coco128", class_names=COCO_CLASSES, split_note=build_split_note("coco128"))
    raise ValueError(f"Unsupported dataset: {dataset_name}")

