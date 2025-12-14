from pathlib import Path
import random
import shutil
import re
import sys

SRC_ROOT = Path(r"E:\WiSARDv1")
OUT_ROOT = Path(r"E:\wisard_vis_yolo")

SEED = 42
TRAIN_FOLDERS = 29
VAL_FOLDERS = 4
TEST_FOLDERS = 4

IMG_EXTS = {".jpg", ".jpeg"} 

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def clean_label_lines(label_path: Path) -> list[str]:
    """
    Keep only valid YOLO bbox lines: cls xc yc w h
    - 5 tokens
    - cls int
    - xc,yc,w,h floats in [0,1]
    - w,h > 0
    """
    if not label_path.exists():
        return []

    text = label_path.read_text(errors="ignore").strip()
    if not text:
        return []

    out = []
    for raw in text.splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])
        except:
            continue

        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            continue
        if w <= 0.0 or h <= 0.0:
            continue

        out.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return out

def ensure_dirs():
    for split in ["train", "val", "test"]:
        (OUT_ROOT / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / f"labels/{split}").mkdir(parents=True, exist_ok=True)

def write_yaml():
    yaml_text = f"""path: {OUT_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: person
"""
    (OUT_ROOT / "wisard_vis.yaml").write_text(yaml_text, encoding="utf-8")

def progress_bar(done: int, total: int, prefix: str = "Progress"):
    if total <= 0:
        return
    width = 30
    frac = done / total
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(frac * 100)
    sys.stdout.write(f"\r{prefix}: [{bar}] {pct:3d}% ({done}/{total})")
    sys.stdout.flush()

def is_vis_session_folder(p: Path) -> bool:
    return p.is_dir() and "_VIS" in p.name

def count_images_in_sessions(split_map: dict[Path, str]) -> int:
    total = 0
    for session in split_map.keys():
        for img in session.rglob("*"):
            if img.is_file() and img.suffix.lower() in IMG_EXTS:
                total += 1
    return total

def main():
    ensure_dirs()

    # VIS-only session folders
    sessions = [p for p in SRC_ROOT.iterdir() if is_vis_session_folder(p)]
    sessions.sort()

    random.seed(SEED)
    random.shuffle(sessions)

    need = TRAIN_FOLDERS + VAL_FOLDERS + TEST_FOLDERS
    if len(sessions) < need:
        raise RuntimeError(f"Not enough VIS folders ({len(sessions)}) for requested {need} split.")

    train_sessions = sessions[:TRAIN_FOLDERS]
    val_sessions   = sessions[TRAIN_FOLDERS:TRAIN_FOLDERS + VAL_FOLDERS]
    test_sessions  = sessions[TRAIN_FOLDERS + VAL_FOLDERS:need]

    split_map = {s: "train" for s in train_sessions}
    split_map.update({s: "val" for s in val_sessions})
    split_map.update({s: "test" for s in test_sessions})

    # Pre-count for progress bar
    total_to_copy = count_images_in_sessions(split_map)
    print(f"Found {total_to_copy} VIS images to copy across train/val/test folders.")
    progress_bar(0, total_to_copy)

    done = 0
    labels_found = 0
    labels_missing = 0
    labels_nonempty = 0
    bad_label_files = 0

    UPDATE_EVERY = 50  

    for session, split in split_map.items():
        img_out = OUT_ROOT / f"images/{split}"
        lbl_out = OUT_ROOT / f"labels/{split}"

        for img in session.rglob("*"):
            if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                continue

            lbl_src = img.with_suffix(".txt")

            # Clean/validate label lines (or empty if missing)
            if lbl_src.exists():
                labels_found += 1
                raw = lbl_src.read_text(errors="ignore").strip()
                lines = clean_label_lines(lbl_src)
                if raw and not lines:
                    bad_label_files += 1
                if lines:
                    labels_nonempty += 1
            else:
                labels_missing += 1
                lines = []

            out_stem = sanitize(f"{session.name}__{img.stem}")
            out_img = img_out / f"{out_stem}{img.suffix.lower()}"
            out_lbl = lbl_out / f"{out_stem}.txt"

            shutil.copy2(img, out_img)
            out_lbl.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

            done += 1
            if done % UPDATE_EVERY == 0 or done == total_to_copy:
                progress_bar(done, total_to_copy)

    sys.stdout.write("\n")  
    write_yaml()

    print("Done.")
    print("VIS sessions total:", len(sessions))
    print("Split folders (train/val/test):", len(train_sessions), len(val_sessions), len(test_sessions))
    print("Images copied:", done)
    print("Label files found:", labels_found)
    print("Label files missing (created empty):", labels_missing)
    print("Non-empty label files (positives):", labels_nonempty)
    print("Label files that had text but no valid lines (check!):", bad_label_files)
    print("YAML:", OUT_ROOT / "wisard_vis.yaml")

if __name__ == "__main__":
    main()