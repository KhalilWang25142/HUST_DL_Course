from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunDirectories:
    run_dir: Path
    checkpoints_dir: Path
    tb_dir: Path
    images_dir: Path
    metrics_dir: Path
    plots_dir: Path
    logs_dir: Path


def build_run_name(dataset: str, model: str, timestamp: str) -> str:
    return f"{dataset}_{model}_{timestamp}"


def prepare_run_directories(output_root: Path, run_name: str) -> RunDirectories:
    run_dir = Path(output_root) / run_name
    checkpoints_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tb"
    images_dir = run_dir / "images"
    metrics_dir = run_dir / "metrics"
    plots_dir = run_dir / "plots"
    logs_dir = run_dir / "logs"

    for path in [run_dir, checkpoints_dir, tb_dir, images_dir, metrics_dir, plots_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return RunDirectories(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        tb_dir=tb_dir,
        images_dir=images_dir,
        metrics_dir=metrics_dir,
        plots_dir=plots_dir,
        logs_dir=logs_dir,
    )


def build_run_summary(args, run_name: str, num_classes: int, split_note: str) -> dict:
    timestamp = run_name.rsplit("_", 2)[-2] + "_" + run_name.rsplit("_", 1)[-1] if "_" in run_name else run_name
    return {
        "run_name": run_name,
        "timestamp": timestamp,
        "dataset": args.dataset,
        "model": args.model,
        "epochs": args.train_epochs,
        "device": args.device,
        "pretrained": bool(args.pretrained),
        "classes": num_classes,
        "split_note": split_note,
    }
