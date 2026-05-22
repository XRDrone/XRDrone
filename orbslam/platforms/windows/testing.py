#!/usr/bin/env python3
"""
testing.py

Runtime logging helper for rtsp_yolo_orbslam_fusion.py.

This module does not run the pipeline by itself. The fusion script imports
OrbslamFusionRunLogger and calls it once per processed frame when --logs is
enabled.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


FRAME_LOG_FIELDS = [
    "frame_id",
    "timestamp",
    "width",
    "height",
    "processing_time_s",
    "fps_estimate",
    "pose_valid",
    "pose_age_s",
    "orbslam_frame_id",
    "detection_count",
    "world_valid_count",
    "packet_sent",
    "packet_size_bytes",
    "rtsp_reconnect_count",
    "error_count",
]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _append_jsonl(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, separators=(",", ":"), default=_json_default) + "\n")
    handle.flush()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _safe_min(values: list[float]) -> float | None:
    return min(values) if values else None


def _safe_max(values: list[float]) -> float | None:
    return max(values) if values else None


class OrbslamFusionRunLogger:
    """Writes ORB-SLAM fusion run logs in an ArUco-comparable layout."""

    def __init__(
        self,
        *,
        log_root: str | Path = "logs",
        run_mode: str = "live",
        args_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.log_root = Path(log_root)
        self.run_mode = run_mode
        self.args_snapshot = args_snapshot or {}
        self.run_dir: Path | None = None
        self.started = False
        self.start_wall_time: float | None = None
        self.end_wall_time: float | None = None

        self.pose_handle: Any | None = None
        self.detections_handle: Any | None = None
        self.packets_handle: Any | None = None
        self.errors_handle: Any | None = None
        self.frames_handle: Any | None = None
        self.frames_writer: csv.DictWriter[str] | None = None

        self.pending_errors: list[dict[str, Any]] = []
        self.processing_times: list[float] = []
        self.pose_ages: list[float] = []
        self.packet_sizes: list[int] = []
        self.track_ids_seen: set[int] = set()

        self.frame_count = 0
        self.pose_valid_frames = 0
        self.detection_total = 0
        self.frames_with_detections = 0
        self.world_projection_attempts = 0
        self.world_projection_successes = 0
        self.packet_count = 0
        self.udp_error_count = 0
        self.error_count = 0
        self.rtsp_read_failures = 0
        self.rtsp_reconnect_count = 0
        self.yolo_error_count = 0
        self.projection_error_count = 0
        self.malformed_pose_packets = 0
        self.pose_packets_received = 0
        self.pose_packets_parsed = 0

    def start(
        self,
        *,
        first_frame_id: int | None = None,
        first_frame_timestamp: float | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> Path:
        if self.started:
            if self.run_dir is None:
                raise RuntimeError("logger started without a run directory")
            return self.run_dir

        stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = self.log_root / stamp
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.start_wall_time = time.time()
        self.started = True

        self.pose_handle = (self.run_dir / "pose_log.jsonl").open("w", encoding="utf-8")
        self.detections_handle = (self.run_dir / "detections_log.jsonl").open(
            "w", encoding="utf-8"
        )
        self.packets_handle = (self.run_dir / "packets_log.jsonl").open(
            "w", encoding="utf-8"
        )
        self.errors_handle = (self.run_dir / "errors_log.jsonl").open(
            "w", encoding="utf-8"
        )
        self.frames_handle = (self.run_dir / "frames_log.csv").open(
            "w", encoding="utf-8", newline=""
        )
        self.frames_writer = csv.DictWriter(self.frames_handle, fieldnames=FRAME_LOG_FIELDS)
        self.frames_writer.writeheader()
        self.frames_handle.flush()

        metadata = {
            "schema": "xrdrone-orbslam-fusion-logs-v1",
            "created_at": _now_iso(),
            "run_mode": self.run_mode,
            "input_source": "rtsp",
            "first_frame_id": first_frame_id,
            "first_frame_timestamp": first_frame_timestamp,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "args": self.args_snapshot,
            "logs": {
                "run_metadata": "run_metadata.json",
                "summary": "summary.json",
                "pose": "pose_log.jsonl",
                "detections": "detections_log.jsonl",
                "packets": "packets_log.jsonl",
                "frames": "frames_log.csv",
                "errors": "errors_log.jsonl",
            },
            "notes": [
                "ORB-SLAM does not provide ArUco marker IDs, corners, or reprojection error.",
                "marker_log.jsonl is intentionally not written for the ORB-SLAM path.",
            ],
        }
        _write_json(self.run_dir / "run_metadata.json", metadata)

        for error in self.pending_errors:
            _append_jsonl(self.errors_handle, error)
        self.pending_errors.clear()

        return self.run_dir

    def log_error(
        self,
        *,
        stage: str,
        message: str,
        frame_id: int | None = None,
        exception: BaseException | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "timestamp": time.time(),
            "datetime": _now_iso(),
            "frame_id": frame_id,
            "stage": stage,
            "message": message,
            "exception_type": type(exception).__name__ if exception else None,
            "exception": str(exception) if exception else None,
            "extra": extra or {},
        }

        self.error_count += 1
        if stage == "udp_send":
            self.udp_error_count += 1
        elif stage in {"rtsp_open", "rtsp_read"}:
            self.rtsp_read_failures += 1
        elif stage == "rtsp_reconnect":
            self.rtsp_reconnect_count += 1
        elif stage == "yolo_inference":
            self.yolo_error_count += 1
        elif stage == "projection":
            self.projection_error_count += 1
        elif stage == "pose_parse":
            self.malformed_pose_packets += 1

        if self.started and self.errors_handle is not None:
            _append_jsonl(self.errors_handle, payload)
        else:
            self.pending_errors.append(payload)

    def update_pose_receiver_stats(self, stats: dict[str, Any] | None) -> None:
        if not stats:
            return
        self.pose_packets_received = int(stats.get("received_packets", 0))
        self.pose_packets_parsed = int(stats.get("parsed_packets", 0))
        self.malformed_pose_packets = int(stats.get("malformed_packets", 0))

    def log_frame(
        self,
        *,
        frame_id: int,
        timestamp: float,
        width: int,
        height: int,
        processing_time_s: float,
        detections: list[dict[str, Any]],
        pose: Any,
        raw_pose: dict[str, Any] | None,
        packet: dict[str, Any],
        packet_sent: bool,
        packet_size_bytes: int,
        rtsp_reconnect_count: int = 0,
        pose_receiver_stats: dict[str, Any] | None = None,
    ) -> None:
        if not self.started:
            self.start(
                first_frame_id=frame_id,
                first_frame_timestamp=timestamp,
                frame_width=width,
                frame_height=height,
            )

        if self.frames_writer is None or self.frames_handle is None:
            raise RuntimeError("frame logger is not initialized")
        if (
            self.pose_handle is None
            or self.detections_handle is None
            or self.packets_handle is None
        ):
            raise RuntimeError("JSONL loggers are not initialized")

        self.update_pose_receiver_stats(pose_receiver_stats)

        pose_valid = bool(_get_attr(pose, "pose_valid", False))
        pose_received_wall_time = _get_attr(pose, "received_wall_time")
        pose_age_s = None
        if pose_received_wall_time is not None:
            pose_age_s = max(0.0, timestamp - float(pose_received_wall_time))

        if pose_valid:
            self.pose_valid_frames += 1
        if pose_age_s is not None:
            self.pose_ages.append(pose_age_s)

        detection_count = len(detections)
        world_valid_count = sum(1 for det in detections if det.get("world_valid"))
        world_attempt_count = detection_count if pose_valid else 0

        self.frame_count += 1
        self.processing_times.append(processing_time_s)
        self.packet_sizes.append(packet_size_bytes)
        self.detection_total += detection_count
        self.world_projection_attempts += world_attempt_count
        self.world_projection_successes += world_valid_count
        self.rtsp_reconnect_count = max(self.rtsp_reconnect_count, rtsp_reconnect_count)
        if detection_count > 0:
            self.frames_with_detections += 1
        if packet_sent:
            self.packet_count += 1

        for det in detections:
            track_id = det.get("id")
            if isinstance(track_id, int):
                self.track_ids_seen.add(track_id)

        pose_record = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "pose_source": "orbslam",
            "pose_valid": pose_valid,
            "pose_age_s": pose_age_s,
            "orbslam_frame_id": _get_attr(pose, "frame_id"),
            "orbslam_timestamp": _get_attr(pose, "timestamp"),
            "camera_position": {
                "x": _get_attr(pose, "x"),
                "y": _get_attr(pose, "y"),
                "z": _get_attr(pose, "z"),
            },
            "quaternion": {
                "qx": _get_attr(pose, "qx"),
                "qy": _get_attr(pose, "qy"),
                "qz": _get_attr(pose, "qz"),
                "qw": _get_attr(pose, "qw"),
            },
            "euler_rad": {
                "yaw": _get_attr(pose, "yaw"),
                "pitch": _get_attr(pose, "pitch"),
                "roll": _get_attr(pose, "roll"),
            },
            "hfov": _get_attr(pose, "hfov"),
            "markers_used": 0,
            "rvec": None,
            "tvec": None,
            "reprojection_error": None,
            "raw_pose": raw_pose,
        }
        _append_jsonl(self.pose_handle, pose_record)

        detections_record = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "width": width,
            "height": height,
            "pose_valid": pose_valid,
            "detection_count": detection_count,
            "world_valid_count": world_valid_count,
            "detections": detections,
        }
        _append_jsonl(self.detections_handle, detections_record)

        _append_jsonl(
            self.packets_handle,
            {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "packet_sent": packet_sent,
                "packet_size_bytes": packet_size_bytes,
                "packet": packet,
            },
        )

        fps_estimate = 1.0 / processing_time_s if processing_time_s > 0 else None
        frame_row = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "width": width,
            "height": height,
            "processing_time_s": processing_time_s,
            "fps_estimate": fps_estimate,
            "pose_valid": int(pose_valid),
            "pose_age_s": pose_age_s,
            "orbslam_frame_id": _get_attr(pose, "frame_id"),
            "detection_count": detection_count,
            "world_valid_count": world_valid_count,
            "packet_sent": int(packet_sent),
            "packet_size_bytes": packet_size_bytes,
            "rtsp_reconnect_count": self.rtsp_reconnect_count,
            "error_count": self.error_count,
        }
        self.frames_writer.writerow(frame_row)
        self.frames_handle.flush()

    def finalize(self, pose_receiver_stats: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.started:
            return {}
        if self.run_dir is None:
            raise RuntimeError("logger finalized without a run directory")

        self.update_pose_receiver_stats(pose_receiver_stats)
        self.end_wall_time = time.time()
        duration = max(0.0, self.end_wall_time - (self.start_wall_time or self.end_wall_time))
        processing_sorted = sorted(self.processing_times)
        packet_sorted = sorted(self.packet_sizes)

        summary = {
            "schema": "xrdrone-orbslam-fusion-summary-v1",
            "completed_at": _now_iso(),
            "run_mode": self.run_mode,
            "run_dir": str(self.run_dir),
            "duration_s": duration,
            "processed_frames": self.frame_count,
            "avg_runtime_fps": self.frame_count / duration if duration > 0 else None,
            "processing_time_s": {
                "avg": _safe_mean(self.processing_times),
                "min": processing_sorted[0] if processing_sorted else None,
                "median": _safe_median(processing_sorted),
                "max": processing_sorted[-1] if processing_sorted else None,
            },
            "pose": {
                "pose_valid_frames": self.pose_valid_frames,
                "pose_valid_ratio": self.pose_valid_frames / self.frame_count
                if self.frame_count
                else 0.0,
                "pose_packets_received": self.pose_packets_received,
                "pose_packets_parsed": self.pose_packets_parsed,
                "malformed_pose_packets": self.malformed_pose_packets,
                "avg_pose_age_s": _safe_mean(self.pose_ages),
                "max_pose_age_s": _safe_max(self.pose_ages),
            },
            "detections": {
                "total_detections": self.detection_total,
                "frames_with_detections": self.frames_with_detections,
                "avg_detections_per_frame": self.detection_total / self.frame_count
                if self.frame_count
                else 0.0,
                "unique_track_ids": len(self.track_ids_seen),
            },
            "world_projection": {
                "attempts": self.world_projection_attempts,
                "successes": self.world_projection_successes,
                "success_ratio": self.world_projection_successes / self.world_projection_attempts
                if self.world_projection_attempts
                else 0.0,
            },
            "udp": {
                "packets_sent": self.packet_count,
                "udp_errors": self.udp_error_count,
                "packet_size_bytes": {
                    "avg": _safe_mean([float(v) for v in packet_sorted]),
                    "min": _safe_min([float(v) for v in packet_sorted]),
                    "median": _safe_median([float(v) for v in packet_sorted]),
                    "max": _safe_max([float(v) for v in packet_sorted]),
                },
            },
            "rtsp": {
                "read_failures": self.rtsp_read_failures,
                "reconnect_count": self.rtsp_reconnect_count,
            },
            "errors": {
                "total_errors": self.error_count,
                "yolo_errors": self.yolo_error_count,
                "projection_errors": self.projection_error_count,
            },
        }
        _write_json(self.run_dir / "summary.json", summary)
        self.close()
        return summary

    def close(self) -> None:
        for handle_name in (
            "pose_handle",
            "detections_handle",
            "packets_handle",
            "errors_handle",
            "frames_handle",
        ):
            handle = getattr(self, handle_name)
            if handle is not None:
                handle.close()
                setattr(self, handle_name, None)
