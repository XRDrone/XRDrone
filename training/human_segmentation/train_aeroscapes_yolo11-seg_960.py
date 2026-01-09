from __future__ import annotations
from pathlib import Path

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE") 

from ultralytics import YOLO, checks
import torch


def main() -> None:
    checks()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Refusing to train on CPU.")
    print("Training on GPU:", torch.cuda.get_device_name(0))

    data_yaml = Path(r"E:\aeroscapes_person_yolo\aeroscapes_person.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    model = YOLO(r"E:\XRDrone\runs\aeroscapes_human_seg_yolo11x_960\weights\last.pt")

    train_args = dict(
        data=str(data_yaml),
        epochs=100,
        time=None,
        patience=10,
        batch=4,
        imgsz=960,
        device=0,
        workers=2,
        cache="disk",

        # Saving / run naming 
        save=True,
        save_period=False,
        project=r"E:\XRDrone\runs",
        name="aeroscapes_human_seg_yolo11x_960",
        exist_ok=True,
        plots=True,
        val=True,
        compile=False,

        # Repro / training behavior
        pretrained=False,
        optimizer="AdamW",
        seed=42,
        deterministic=True,
        single_cls=True,
        rect=False,
        multi_scale=False,
        cos_lr=True,
        close_mosaic=10,
        resume=True,
        amp=True,
        fraction=1.0,
        profile=False,
        freeze=0,

        # Hyperparams / losses 
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        dropout=0.0,

        # Segmentation-specific 
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,

        # Augmentations 
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=1.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.5,
        copy_paste_mode="flip",
    )

    print("\n=== Training config ===")
    for k in sorted(train_args):
        print(f"{k}: {train_args[k]}")
    print("=======================\n")

    results = model.train(**train_args)

    # Print where outputs went 
    save_dir = getattr(results, "save_dir", None) or getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir:
        print(f"\nTraining complete. Save dir: {save_dir}")
    else:
        print("\nTraining complete.")

if __name__ == "__main__":
    main()
