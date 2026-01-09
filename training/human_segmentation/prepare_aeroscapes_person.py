from __future__ import annotations

from pathlib import Path
import base64
import io
import json
import random
import re
import shutil
import sys
import zlib

import cv2
import numpy as np
from PIL import Image

# ----------------------------
# PATHS
# ----------------------------
SRC_ROOT = Path(r"E:\aeroscapes-DatasetNinja")
OUT_ROOT = Path(r"E:\aeroscapes_person_yolo")

# ----------------------------
# SETTINGS
# ----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png"}

PERSON_CLASS_TITLE = "person"
PERSON_CLASS_ID = 52174

# DatasetNinja stats 
EXPECTED_PERSON_TRAIN = 2597

# Create test by sampling from TRAIN positives 
SEED = 42
TEST_FRAC_FROM_TRAIN = 0.10  # 10% of train person-positives become test

# polygon simplification (smaller => more points)
EPS_FRAC = 0.002

# ----------------------------

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def progress_bar(done: int, total: int, prefix: str) -> None:
    if total <= 0:
        return
    width = 30
    frac = done / total
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(frac * 100)
    sys.stdout.write(f"\r{prefix}: [{bar}] {pct:3d}% ({done}/{total})")
    sys.stdout.flush()

def ensure_dirs() -> None:
    for split in ["train", "val", "test"]:
        (OUT_ROOT / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / f"labels/{split}").mkdir(parents=True, exist_ok=True)

def write_yaml() -> None:
    yaml_text = f"""path: {OUT_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: person
"""
    (OUT_ROOT / "aeroscapes_person.yaml").write_text(yaml_text, encoding="utf-8")

def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def has_person_object(ann: dict) -> bool:
    for o in ann.get("objects", []):
        if o.get("geometryType") != "bitmap":
            continue
        if o.get("classTitle") == PERSON_CLASS_TITLE or o.get("classId") == PERSON_CLASS_ID:
            return True
    return False

def decode_bitmap_to_mask(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    try:
        png_bytes = zlib.decompress(raw)
    except zlib.error:
        png_bytes = raw
    pil = Image.open(io.BytesIO(png_bytes)).convert("L")
    arr = np.array(pil, dtype=np.uint8)
    return (arr > 0).astype(np.uint8) * 255

def build_full_person_mask(ann: dict) -> tuple[np.ndarray, int, int]:
    w = int(ann["size"]["width"])
    h = int(ann["size"]["height"])
    full = np.zeros((h, w), dtype=np.uint8)

    for o in ann.get("objects", []):
        if o.get("geometryType") != "bitmap":
            continue
        if not (o.get("classTitle") == PERSON_CLASS_TITLE or o.get("classId") == PERSON_CLASS_ID):
            continue
        bm = o.get("bitmap")
        if not bm or "data" not in bm:
            continue

        x0, y0 = map(int, bm.get("origin", [0, 0]))
        local = decode_bitmap_to_mask(bm["data"])
        lh, lw = local.shape[:2]

        x1, y1 = max(0, x0), max(0, y0)
        x2, y2 = min(w, x0 + lw), min(h, y0 + lh)
        if x2 <= x1 or y2 <= y1:
            continue

        lx1, ly1 = x1 - x0, y1 - y0
        lx2, ly2 = lx1 + (x2 - x1), ly1 + (y2 - y1)

        patch = local[ly1:ly2, lx1:lx2]
        full[y1:y2, x1:x2] = np.maximum(full[y1:y2, x1:x2], patch)

    return full, w, h

def mask_to_yolo_polygons(mask: np.ndarray, w: int, h: int) -> list[list[float]]:
    if mask.max() == 0:
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[list[float]] = []

    for cnt in contours:
        if len(cnt) < 3:
            continue
        perim = cv2.arcLength(cnt, True)
        eps = max(1.0, EPS_FRAC * perim)
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = approx.reshape(-1, 2)
        if pts.shape[0] < 3:
            continue

        poly: list[float] = []
        for x, y in pts:
            poly.append(float(x) / float(w))
            poly.append(float(y) / float(h))

        if len(poly) >= 6:
            polys.append(poly)

    return polys

def write_yolo_seg_label(label_path: Path, polys: list[list[float]]) -> None:
    lines = ["0 " + " ".join(f"{v:.6f}" for v in poly) for poly in polys]
    label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

def scan_split_for_person(split: str) -> tuple[list[tuple[Path, Path]], dict]:
    img_dir = SRC_ROOT / split / "img"
    ann_dir = SRC_ROOT / split / "ann"
    if not img_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(f"Missing {split}/img or {split}/ann under {SRC_ROOT}")

    imgs = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    imgs.sort()
    total_imgs = len(imgs)

    positives: list[tuple[Path, Path]] = []
    ann_missing = 0
    ann_parse_fail = 0
    no_person = 0

    progress_bar(0, total_imgs, f"Scanning {split}")
    for i, img_path in enumerate(imgs, start=1):
        ann_path = ann_dir / f"{img_path.name}.json"
        if not ann_path.exists():
            ann_missing += 1
            progress_bar(i, total_imgs, f"Scanning {split}")
            continue

        try:
            ann = load_json(ann_path)
        except Exception:
            ann_parse_fail += 1
            progress_bar(i, total_imgs, f"Scanning {split}")
            continue

        if has_person_object(ann):
            positives.append((img_path, ann_path))
        else:
            no_person += 1

        progress_bar(i, total_imgs, f"Scanning {split}")
    sys.stdout.write("\n")

    stats = dict(
        total_imgs=total_imgs,
        positives=len(positives),
        ann_missing=ann_missing,
        ann_parse_fail=ann_parse_fail,
        no_person=no_person,
    )
    return positives, stats

def export_items(split: str, items: list[tuple[Path, Path]]) -> dict:
    img_out = OUT_ROOT / f"images/{split}"
    lbl_out = OUT_ROOT / f"labels/{split}"

    decode_fail = 0
    poly_empty = 0
    exported = 0

    total = len(items)
    progress_bar(0, total, f"Exporting {split}")
    for i, (img_path, ann_path) in enumerate(items, start=1):
        try:
            ann = load_json(ann_path)
            mask, w, h = build_full_person_mask(ann)
        except Exception:
            decode_fail += 1
            progress_bar(i, total, f"Exporting {split}")
            continue

        polys = mask_to_yolo_polygons(mask, w, h)
        if not polys:
            poly_empty += 1
            progress_bar(i, total, f"Exporting {split}")
            continue

        out_img_name = sanitize(img_path.name)
        out_img = img_out / out_img_name
        out_lbl = lbl_out / f"{Path(out_img_name).stem}.txt"

        shutil.copy2(img_path, out_img)
        write_yolo_seg_label(out_lbl, polys)

        exported += 1
        progress_bar(i, total, f"Exporting {split}")

    sys.stdout.write("\n")
    return dict(exported=exported, decode_fail=decode_fail, poly_empty=poly_empty)

def main() -> None:
    ensure_dirs()

    # 1) Scan provided splits
    train_pos, train_stats = scan_split_for_person("train")
    val_pos, val_stats     = scan_split_for_person("val")

    print(f"TRAIN positives (person): {train_stats['positives']}")
    if EXPECTED_PERSON_TRAIN is not None and train_stats["positives"] != EXPECTED_PERSON_TRAIN:
        print(f"WARNING: expected {EXPECTED_PERSON_TRAIN} person images in TRAIN, found {train_stats['positives']}.")

    print(f"VAL positives (person):   {val_stats['positives']}")

    # 2) Create test from TRAIN positives (deterministic)
    random.seed(SEED)
    shuffled = train_pos[:]
    random.shuffle(shuffled)

    n_train_pos = len(shuffled)
    n_test = int(round(n_train_pos * TEST_FRAC_FROM_TRAIN))
    n_test = max(1, n_test) if n_train_pos >= 10 else min(n_train_pos, n_test)  # avoid silly splits

    test_set = shuffled[:n_test]
    train_set = shuffled[n_test:]

    print(f"Split from TRAIN positives -> train: {len(train_set)}, test: {len(test_set)} (test_frac={TEST_FRAC_FROM_TRAIN})")
    print(f"VAL kept as provided -> val: {len(val_pos)}")

    # 3) Export
    train_export = export_items("train", train_set)
    test_export  = export_items("test",  test_set)
    val_export   = export_items("val",   val_pos)

    write_yaml()

    # 4) Summary
    print("Done.\n")
    print("SCAN STATS")
    print("TRAIN:", train_stats)
    print("VAL:  ", val_stats)
    print("\nEXPORT STATS")
    print("train:", train_export)
    print("test: ", test_export)
    print("val:  ", val_export)
    print("\nYAML:", OUT_ROOT / "aeroscapes_person.yaml")

if __name__ == "__main__":
    main()
