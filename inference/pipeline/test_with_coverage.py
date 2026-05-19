"""
test_with_coverage.py

Lightweight smoke checks for the pure ArUco runtime packet shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_latest_log(log_root: str = "logs") -> int:
    root = Path(log_root)
    runs = sorted([p for p in root.glob("run_*") if p.is_dir()])
    if not runs:
        raise AssertionError(f"No run folders found under {root}")
    run = runs[-1]
    for name in [
        "run_metadata.json",
        "summary.json",
        "pose_log.jsonl",
        "marker_log.jsonl",
        "detections_log.jsonl",
        "packets_log.jsonl",
        "frames_log.csv",
        "errors_log.jsonl",
    ]:
        path = run / name
        if not path.exists():
            raise AssertionError(f"Missing expected log file: {path}")

    packet_lines = (run / "packets_log.jsonl").read_text(encoding="utf-8").splitlines()
    if not packet_lines:
        raise AssertionError("packets_log.jsonl is empty")
    packet = json.loads(packet_lines[0])
    for key in ["frame_id", "timestamp", "timestamp_video_s", "width", "height", "pose", "detections", "counts"]:
        if key not in packet:
            raise AssertionError(f"Missing packet key: {key}")
    if not isinstance(packet["pose"], dict):
        raise AssertionError("packet.pose must be an object")
    if not isinstance(packet["detections"], list):
        raise AssertionError("packet.detections must be a list")
    print(f"Validated latest log folder: {run}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Validate latest XRDrone ArUco log folder")
    parser.add_argument("--log-root", default="logs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(validate_latest_log(args.log_root))
