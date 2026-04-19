import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    command = [
        sys.executable,
        str(project_dir / "run.py"),
        "--dataset",
        "coco128",
        "--model",
        "yolov8",
        "--train_epochs",
        "100",
        "--batch_size",
        "32",
        "--num_workers",
        "8",
        "--data_root",
        str(project_dir / "dataset"),
        "--output_root",
        str(project_dir / "outputs"),
        "--yolo_weights",
        str(project_dir / "yolov8n.pt"),
        "--device",
        "cuda",
        "--amp",
        "--yolo-cache",
        "ram",
        "--imgsz",
        "640",
        *sys.argv[1:],
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
