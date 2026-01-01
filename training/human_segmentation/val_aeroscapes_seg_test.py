from __future__ import annotations
from pathlib import Path
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO, checks
import torch


def main() -> None:
    # Same check pattern as your training scripts
    checks()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Refusing to run on CPU.")
    print("GPU:", torch.cuda.get_device_name(0))

    # Minimal: set data_yaml once
    data_yaml = Path(r"E:\aeroscapes_person_yolo\aeroscapes_person.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    runs_root = Path(r"E:\XRDrone\runs")

    # Minimal: only weights + name vary per run
    # (ignoring aeroscapes_human_seg_yolo11_960 as requested)
    model_dirs = [
        "aeroscapes_human_seg_yolo11n_960",
        "aeroscapes_human_seg_yolo11s_960",
        "aeroscapes_human_seg_yolo11m_960",
    ]

    for d in model_dirs:
        weights = runs_root / d / "weights" / "best.pt"
        name = f"{d}_test"

        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")

        print(f"\n=== VAL (SEG) {d} ===")
        print(f"Weights: {weights}")
        print(f"Data:    {data_yaml}")
        print(f"Name:    {name}")

        model = YOLO(str(weights))
        metrics = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=960,
            batch=4,
            device="0",
            workers=2,
            conf=0.001,
            iou=0.7,
            max_det=300,
            half=True,
            plots=True,
            save_json=True,
            single_cls=True,
            rect=False,
            compile=False,
            project=str(runs_root),
            name=name,
            verbose=True,
        )

        print("Box mAP50-95:", metrics.box.map)
        if hasattr(metrics, "seg"):
            print("Seg mAP50-95:", metrics.seg.map)

        print(metrics.to_csv())
        print(metrics.to_json())


if __name__ == "__main__":
    main()
