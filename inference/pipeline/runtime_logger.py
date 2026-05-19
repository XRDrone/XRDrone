"""
runtime_logger.py

Creates per-run log folders and writes XRDrone ArUco runtime logs.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


class RuntimeLogger:
    def __init__(self, root: str | Path, *, prefix: str = "run") -> None:
        root_path = Path(root)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = root_path / f"{prefix}_{stamp}"
        suffix = 1
        while self.run_dir.exists():
            self.run_dir = root_path / f"{prefix}_{stamp}_{suffix:02d}"
            suffix += 1
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self._pose_f = (self.run_dir / "pose_log.jsonl").open("w", encoding="utf-8")
        self._det_f = (self.run_dir / "detections_log.jsonl").open("w", encoding="utf-8")
        self._packet_f = (self.run_dir / "packets_log.jsonl").open("w", encoding="utf-8")
        self._marker_f = (self.run_dir / "marker_log.jsonl").open("w", encoding="utf-8")
        self._error_f = (self.run_dir / "errors_log.jsonl").open("w", encoding="utf-8")
        self._frame_csv_f = (self.run_dir / "frames_log.csv").open("w", newline="", encoding="utf-8")
        self._frame_csv = csv.DictWriter(
            self._frame_csv_f,
            fieldnames=[
                "frame_id",
                "timestamp",
                "timestamp_video_s",
                "width",
                "height",
                "pose_valid",
                "markers_detected",
                "markers_used",
                "detection_count",
                "processing_ms",
            ],
        )
        self._frame_csv.writeheader()

    @staticmethod
    def _write_jsonl(handle, obj: dict[str, Any]) -> None:
        handle.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")

    def write_metadata(self, obj: dict[str, Any]) -> None:
        (self.run_dir / "run_metadata.json").write_text(
            json.dumps(obj, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_summary(self, obj: dict[str, Any]) -> None:
        (self.run_dir / "summary.json").write_text(
            json.dumps(obj, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def log_pose(self, pose: dict[str, Any]) -> None:
        self._write_jsonl(self._pose_f, pose)

    def log_markers(self, markers: dict[str, Any]) -> None:
        self._write_jsonl(self._marker_f, markers)

    def log_detections(self, frame_record: dict[str, Any]) -> None:
        self._write_jsonl(self._det_f, frame_record)

    def log_packet(self, packet: dict[str, Any]) -> None:
        self._write_jsonl(self._packet_f, packet)

    def log_error(self, obj: dict[str, Any]) -> None:
        self._write_jsonl(self._error_f, obj)

    def log_frame_row(self, row: dict[str, Any]) -> None:
        self._frame_csv.writerow(row)

    def close(self) -> None:
        for handle in (
            self._pose_f,
            self._det_f,
            self._packet_f,
            self._marker_f,
            self._error_f,
            self._frame_csv_f,
        ):
            try:
                handle.flush()
                handle.close()
            except Exception:
                pass
