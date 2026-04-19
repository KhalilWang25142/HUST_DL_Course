from __future__ import annotations

from pathlib import Path


def build_yolo_data_config(
    output_dir: Path,
    dataset_name: str,
    root_path: Path,
    train_path: str,
    val_path: str,
    class_names: list[str],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / f"{dataset_name}.yaml"
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    content = (
        f"path: {Path(root_path).as_posix()}\n"
        f"train: {train_path}\n"
        f"val: {val_path}\n"
        "names:\n"
        f"{names_block}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path
