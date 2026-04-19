import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    command = [
        sys.executable,
        str(project_dir / "run.py"),
        "--dataset",
        "voc2007",
        "--model",
        "rcnn",
        "--train_epochs",
        "10",
        "--batch_size",
        "4",
        "--num_workers",
        "4",
        "--learning_rate",
        "0.0001",
        "--data_root",
        str(project_dir / "dataset"),
        "--output_root",
        str(project_dir / "outputs"),
        "--device",
        "cuda",
        *sys.argv[1:],
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
