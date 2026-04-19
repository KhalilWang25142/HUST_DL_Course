import argparse
import sys
from pathlib import Path


def ensure_project_root_on_path(file_path: str) -> None:
    project_root = str(Path(file_path).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detection Minimal Benchmark")
    parser.add_argument("--dataset", type=str, choices=["voc2007", "coco128"], required=True)
    parser.add_argument(
        "--model",
        type=str,
        choices=["rcnn", "fast_rcnn", "faster_rcnn", "yolov8"],
        required=True,
    )
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data_root", type=Path, default=Path("./detection_minimal/dataset"))
    parser.add_argument("--output_root", type=Path, default=Path("./detection_minimal/outputs"))
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--yolo_weights", type=str, default="")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--yolo-cache", type=str, choices=["false", "ram", "disk"], default="false")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--smoke_test", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--save_vis_count", type=int, default=10)
    parser.add_argument("--score_threshold", type=float, default=0.25)
    return parser


def main() -> None:
    ensure_project_root_on_path(__file__)
    parser = build_parser()
    args = parser.parse_args()
    from detection_minimal.exp.exp_detection import ExpDetection

    exp = ExpDetection(args)
    exp.run()


if __name__ == "__main__":
    main()
