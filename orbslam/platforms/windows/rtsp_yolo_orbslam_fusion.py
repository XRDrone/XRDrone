#!/usr/bin/env python3
"""
rtsp_yolo_orbslam_fusion.py

Windows-side XRDrone fusion script:

1. Reads drone video from an RTSP stream.
2. Runs YOLO human detection/tracking.
3. Receives ORB-SLAM3 camera pose over UDP.
4. Estimates each detected person's 3D ground-plane position.
5. Sends fused JSON over UDP to Unity.
6. Optionally writes ArUco-comparable runtime logs with --logs.

Expected ORB-SLAM text pose format:
    frame timestamp x y z qx qy qz qw

Latency notes:
- OPENCV_FFMPEG_CAPTURE_OPTIONS is set before opening RTSP.
- CAP_PROP_BUFFERSIZE is set to 1.
- DROP_STALE_GRABS skips stale frames before decoding the next frame.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from orbslam_fusion_testing import OrbslamFusionRunLogger
except ImportError:  # Allows --help/basic runs before the testing file is copied beside this file.
    OrbslamFusionRunLogger = None  # type: ignore[assignment]


# =============================================================================
# USER CONFIGURATION
# Default run command:
#     python rtsp_yolo_orbslam_fusion.py
# =============================================================================
RTSP_URL = "rtsp://127.0.0.1:8554/dji"
YOLO_MODEL_PATH = "yolov5nu.pt"

# Use 0.0.0.0 so Windows can listen even if its Wi-Fi IP changes.
POSE_LISTEN_IP = "0.0.0.0"
POSE_PORT = 9000
POSE_FORMAT = "orbslam-text"

UNITY_OUTPUT_HOST = "127.0.0.1"
UNITY_OUTPUT_PORT = 9002

SHOW_WINDOW = True
DEVICE = 0  # Use 0 for CUDA GPU 0, or "cpu" for CPU-only.
IMG_SIZE = 640
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.70
PERSON_CLASS_ID = 0
USE_YOLO_TRACKING = True

MAX_POSE_AGE_SECONDS = 1.0
POSE_SCALE = 1.0

SWAP_YZ = False
INVERT_X = False
INVERT_Y = False
INVERT_Z = False

DEFAULT_HFOV_DEG = 70.0
DEFAULT_CAMERA_HEIGHT_M = 1.5
GROUND_Y = 0.0
POSE_ANGLES_IN_DEGREES = False

YAW_OFFSET_DEG = 0.0
PITCH_OFFSET_DEG = 0.0
ROLL_OFFSET_DEG = 0.0

RECONNECT_DELAY_SECONDS = 2.0
DROP_STALE_GRABS = 3
RTSP_CAPTURE_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
LOG_ROOT = "logs"
WAIT_TIMEOUT_SECONDS = 0.0  # 0 means wait forever.


@dataclass
class Pose:
    received_wall_time: float
    timestamp: float
    frame_id: int | None
    x: float
    y: float
    z: float
    qx: float | None
    qy: float | None
    qz: float | None
    qw: float | None
    yaw: float
    pitch: float
    roll: float
    hfov: float | None
    pose_valid: bool


class LatestPoseReceiver:
    """Receives ORB-SLAM UDP pose packets and stores only the newest pose."""

    def __init__(
        self,
        listen_ip: str,
        listen_port: int,
        default_camera_height_m: float,
        pose_scale: float,
        pose_format: str,
        swap_yz: bool,
        invert_x: bool,
        invert_y: bool,
        invert_z: bool,
        error_callback: Callable[..., None] | None = None,
    ) -> None:
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.default_camera_height_m = default_camera_height_m
        self.pose_scale = pose_scale
        self.pose_format = pose_format
        self.swap_yz = swap_yz
        self.invert_x = invert_x
        self.invert_y = invert_y
        self.invert_z = invert_z
        self.error_callback = error_callback
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.latest_pose: Pose | None = None
        self.latest_raw: dict[str, Any] | None = None
        self.received_packets = 0
        self.parsed_packets = 0
        self.malformed_packets = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def get_latest(self, max_age_s: float) -> tuple[Pose | None, dict[str, Any] | None]:
        with self._lock:
            pose = self.latest_pose
            raw = self.latest_raw

        if pose is None:
            return None, None
        if time.time() - pose.received_wall_time > max_age_s:
            return None, raw
        return pose, raw

    def get_stats(self) -> dict[str, int]:
        with self._stats_lock:
            return {
                "received_packets": self.received_packets,
                "parsed_packets": self.parsed_packets,
                "malformed_packets": self.malformed_packets,
            }

    def _increment_stat(self, name: str) -> None:
        with self._stats_lock:
            setattr(self, name, getattr(self, name) + 1)

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.listen_ip, self.listen_port))
        sock.settimeout(0.5)

        print(f"[pose] Listening for ORB-SLAM pose UDP on {self.listen_ip}:{self.listen_port}")
        print("[pose] Expected text format: frame timestamp x y z qx qy qz qw")

        while not self._stop.is_set():
            try:
                data, _addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break

            self._increment_stat("received_packets")
            msg = data.decode("utf-8", errors="replace").strip()
            try:
                pose, raw = self._parse_packet(msg)
            except Exception as exc:
                self._increment_stat("malformed_packets")
                print(f"[pose] Ignored malformed pose packet: {exc}; packet={msg!r}")
                if self.error_callback is not None:
                    self.error_callback(
                        stage="pose_parse",
                        message="Ignored malformed pose packet",
                        exception=exc,
                        extra={"packet": msg},
                    )
                continue

            self._increment_stat("parsed_packets")
            with self._lock:
                self.latest_pose = pose
                self.latest_raw = raw

        sock.close()

    def _parse_packet(self, msg: str) -> tuple[Pose, dict[str, Any]]:
        if self.pose_format == "json":
            raw = json.loads(msg)
            return self._parse_json_pose(raw), raw

        if self.pose_format == "orbslam-text":
            return self._parse_orbslam_text_pose(msg)

        if msg.startswith("{"):
            raw = json.loads(msg)
            return self._parse_json_pose(raw), raw

        return self._parse_orbslam_text_pose(msg)

    def _apply_position_mapping(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        if self.swap_yz:
            y, z = z, y
        if self.invert_x:
            x = -x
        if self.invert_y:
            y = -y
        if self.invert_z:
            z = -z
        return x * self.pose_scale, y * self.pose_scale, z * self.pose_scale

    def _parse_orbslam_text_pose(self, msg: str) -> tuple[Pose, dict[str, Any]]:
        parts = msg.split()
        if len(parts) != 9:
            raise ValueError(
                f"expected 9 space-separated values, got {len(parts)}; "
                "expected: frame timestamp x y z qx qy qz qw"
            )

        frame_id = int(parts[0])
        timestamp = float(parts[1])
        raw_x, raw_y, raw_z = float(parts[2]), float(parts[3]), float(parts[4])
        qx, qy, qz, qw = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])

        x, y, z = self._apply_position_mapping(raw_x, raw_y, raw_z)
        yaw, pitch, roll = quaternion_to_yaw_pitch_roll(qx, qy, qz, qw)

        raw = {
            "format": "orbslam-text",
            "frame_id": frame_id,
            "timestamp": timestamp,
            "raw_position": {"x": raw_x, "y": raw_y, "z": raw_z},
            "position": {"x": x, "y": y, "z": z},
            "quaternion": {"qx": qx, "qy": qy, "qz": qz, "qw": qw},
            "euler_rad": {"yaw": yaw, "pitch": pitch, "roll": roll},
        }

        return (
            Pose(
                time.time(),
                timestamp,
                frame_id,
                x,
                y,
                z,
                qx,
                qy,
                qz,
                qw,
                yaw,
                pitch,
                roll,
                None,
                True,
            ),
            raw,
        )

    def _parse_json_pose(self, raw: dict[str, Any]) -> Pose:
        pose_block = raw.get("pose", raw)

        def read_float(*keys: str, default: float = 0.0) -> float:
            for key in keys:
                value = pose_block.get(key, raw.get(key))
                if value is not None:
                    return float(value)
            return default

        def read_bool(*keys: str, default: bool = True) -> bool:
            for key in keys:
                value = pose_block.get(key, raw.get(key))
                if value is not None:
                    return bool(value)
            return default

        frame_id_raw = pose_block.get("frame_id", raw.get("frame_id"))
        frame_id = int(frame_id_raw) if frame_id_raw is not None else None

        raw_x = read_float("x")
        raw_y = read_float("y", "altitude", "height", default=self.default_camera_height_m)
        raw_z = read_float("z")
        x, y, z = self._apply_position_mapping(raw_x, raw_y, raw_z)

        timestamp = read_float("timestamp", default=time.time())
        hfov_value = pose_block.get("hfov", raw.get("hfov"))
        hfov = float(hfov_value) if hfov_value is not None else None

        qx_value = pose_block.get("qx", raw.get("qx"))
        qy_value = pose_block.get("qy", raw.get("qy"))
        qz_value = pose_block.get("qz", raw.get("qz"))
        qw_value = pose_block.get("qw", raw.get("qw"))

        if None not in (qx_value, qy_value, qz_value, qw_value):
            qx, qy, qz, qw = float(qx_value), float(qy_value), float(qz_value), float(qw_value)
            yaw, pitch, roll = quaternion_to_yaw_pitch_roll(qx, qy, qz, qw)
        else:
            qx = qy = qz = qw = None
            yaw, pitch, roll = read_float("yaw"), read_float("pitch"), read_float("roll")

        return Pose(
            received_wall_time=time.time(),
            timestamp=timestamp,
            frame_id=frame_id,
            x=x,
            y=y,
            z=z,
            qx=qx,
            qy=qy,
            qz=qz,
            qw=qw,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            hfov=hfov,
            pose_valid=read_bool("pose_valid", "valid", default=True),
        )


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)

    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def quaternion_to_yaw_pitch_roll(
    qx: float, qy: float, qz: float, qw: float
) -> tuple[float, float, float]:
    rot = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    pitch = math.asin(max(-1.0, min(1.0, -rot[1, 2])))
    cp = math.cos(pitch)
    if abs(cp) > 1e-6:
        yaw = math.atan2(rot[0, 2], rot[2, 2])
        roll = math.atan2(rot[1, 0], rot[1, 1])
    else:
        yaw = math.atan2(-rot[2, 0], rot[0, 0])
        roll = 0.0
    return yaw, pitch, roll


def rotation_matrix_yaw_pitch_roll(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    r_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    r_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    r_roll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return r_yaw @ r_pitch @ r_roll


def deg_to_rad_if_needed(value: float, angles_in_degrees: bool) -> float:
    return math.radians(value) if angles_in_degrees else value


def project_pixel_to_ground(
    *,
    u: float,
    v: float,
    image_width: int,
    image_height: int,
    pose: Pose,
    default_hfov_deg: float,
    ground_y: float,
    angles_in_degrees: bool,
    yaw_offset_deg: float,
    pitch_offset_deg: float,
    roll_offset_deg: float,
) -> tuple[bool, float | None, float | None, float | None]:
    hfov_deg = pose.hfov if pose.hfov is not None else default_hfov_deg
    hfov_rad = math.radians(hfov_deg)
    fx = image_width / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx
    cx, cy = image_width / 2.0, image_height / 2.0

    ray_camera = np.array([(u - cx) / fx, -(v - cy) / fy, 1.0], dtype=np.float64)
    ray_camera /= np.linalg.norm(ray_camera)

    yaw = deg_to_rad_if_needed(pose.yaw, angles_in_degrees) + math.radians(yaw_offset_deg)
    pitch = deg_to_rad_if_needed(pose.pitch, angles_in_degrees) + math.radians(pitch_offset_deg)
    roll = deg_to_rad_if_needed(pose.roll, angles_in_degrees) + math.radians(roll_offset_deg)

    ray_world = rotation_matrix_yaw_pitch_roll(yaw, pitch, roll) @ ray_camera
    ray_world /= np.linalg.norm(ray_world)
    origin = np.array([pose.x, pose.y, pose.z], dtype=np.float64)

    if abs(ray_world[1]) < 1e-6:
        return False, None, None, None

    t = (ground_y - origin[1]) / ray_world[1]
    if t <= 0.0:
        return False, None, None, None

    point = origin + t * ray_world
    return True, float(point[0]), float(point[1]), float(point[2])


def open_rtsp_capture(rtsp_url: str) -> cv2.VideoCapture:
    # Must be set before opening the capture.
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", RTSP_CAPTURE_OPTIONS)

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open RTSP stream: {rtsp_url}")

    return cap


def open_rtsp_capture_with_wait(
    *,
    rtsp_url: str,
    reconnect_delay_s: float,
    wait_timeout_s: float,
    logger: Any | None,
) -> cv2.VideoCapture:
    wait_start = time.time()
    attempt = 0
    while True:
        attempt += 1
        try:
            cap = open_rtsp_capture(rtsp_url)
            print(f"[video] RTSP stream opened after {attempt} attempt(s): {rtsp_url}")
            return cap
        except Exception as exc:
            elapsed = time.time() - wait_start
            if logger is not None:
                logger.log_error(
                    stage="rtsp_open",
                    message="Waiting for RTSP stream to become available",
                    exception=exc,
                    extra={"attempt": attempt, "elapsed_s": elapsed, "rtsp_url": rtsp_url},
                )
            if wait_timeout_s > 0 and elapsed >= wait_timeout_s:
                raise TimeoutError(
                    f"Timed out after {elapsed:.2f}s waiting for RTSP stream: {rtsp_url}"
                ) from exc
            print(f"[video] Waiting for RTSP stream ({attempt})... {exc}")
            time.sleep(reconnect_delay_s)


def draw_overlay(frame: np.ndarray, detections: list[dict[str, Any]], pose_ok: bool) -> np.ndarray:
    for det in detections:
        x1 = int(det["cx"] - det["w"] / 2)
        y1 = int(det["cy"] - det["h"] / 2)
        x2 = int(det["cx"] + det["w"] / 2)
        y2 = int(det["cy"] + det["h"] / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"id={det['id']} conf={det['conf']:.2f}"
        if det["world_valid"]:
            label += f" xyz=({det['world_x']:.2f},{det['world_y']:.2f},{det['world_z']:.2f})"
        else:
            label += " xyz=invalid"

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )

    pose_text = "pose: OK" if pose_ok else "pose: missing/stale"
    cv2.putText(
        frame,
        pose_text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if pose_ok else (0, 0, 255),
        2,
    )
    return frame


def extract_raw_pose_value(raw_pose: dict[str, Any] | None, *keys: str) -> Any:
    if raw_pose is None:
        return None
    pose_block = raw_pose.get("pose", raw_pose)
    position = raw_pose.get("position", {})
    quaternion = raw_pose.get("quaternion", {})
    for key in keys:
        if key in pose_block:
            return pose_block[key]
        if key in raw_pose:
            return raw_pose[key]
        if key in position:
            return position[key]
        if key in quaternion:
            return quaternion[key]
    return None


def build_slam_block(
    *,
    pose: Pose | None,
    raw_pose: dict[str, Any] | None,
    pose_ok: bool,
    now: float,
) -> dict[str, Any]:
    pose_age_s = max(0.0, now - pose.received_wall_time) if pose is not None else None
    if pose_ok and pose is not None:
        return {
            "tracking_state": "ok",
            "match_mode": "latest",
            "pose_valid": True,
            "frame_id": pose.frame_id,
            "timestamp": pose.timestamp,
            "pose_age_s": pose_age_s,
            "x": pose.x,
            "y": pose.y,
            "z": pose.z,
            "qx": pose.qx,
            "qy": pose.qy,
            "qz": pose.qz,
            "qw": pose.qw,
        }

    tracking_state = "stale" if raw_pose is not None else "missing"
    return {
        "tracking_state": tracking_state,
        "match_mode": "none",
        "pose_valid": False,
        "frame_id": extract_raw_pose_value(raw_pose, "frame_id"),
        "timestamp": extract_raw_pose_value(raw_pose, "timestamp"),
        "pose_age_s": None,
        "x": extract_raw_pose_value(raw_pose, "x"),
        "y": extract_raw_pose_value(raw_pose, "y"),
        "z": extract_raw_pose_value(raw_pose, "z"),
        "qx": extract_raw_pose_value(raw_pose, "qx"),
        "qy": extract_raw_pose_value(raw_pose, "qy"),
        "qz": extract_raw_pose_value(raw_pose, "qz"),
        "qw": extract_raw_pose_value(raw_pose, "qw"),
    }


def build_pose_packet(
    *,
    pose: Pose | None,
    pose_ok: bool,
    default_hfov_deg: float,
) -> dict[str, Any]:
    return {
        "x": pose.x if pose else None,
        "altitude": pose.y if pose else None,
        "z": pose.z if pose else None,
        "yaw": pose.yaw if pose else None,
        "pitch": pose.pitch if pose else None,
        "roll": pose.roll if pose else None,
        "hfov": pose.hfov if pose and pose.hfov is not None else default_hfov_deg,
        "markers_used": 0,
        "pose_valid": bool(pose_ok),
    }


def build_fusion_status(
    *,
    pose_ok: bool,
    raw_pose: dict[str, Any] | None,
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    attempted = len(detections) if pose_ok else 0
    projected = sum(1 for det in detections if det.get("world_valid"))

    if not pose_ok:
        slam_tracking = "stale" if raw_pose is not None else "missing"
        projection_state = "unavailable"
        reason = "no fresh ORB-SLAM pose; waiting for ORB-SLAM UDP packets"
    elif attempted == 0:
        slam_tracking = "ok"
        projection_state = "idle"
        reason = "no eligible detections to project"
    elif projected == attempted:
        slam_tracking = "ok"
        projection_state = "ok"
        reason = ""
    elif projected > 0:
        slam_tracking = "ok"
        projection_state = "partial"
        reason = "some detections could not be projected to the ground plane"
    else:
        slam_tracking = "ok"
        projection_state = "unavailable"
        reason = "detections did not intersect the configured ground plane"

    return {
        "source": "orbslam",
        "slam_tracking": slam_tracking,
        "match_mode": "latest" if pose_ok else "none",
        "projection_state": projection_state,
        "pose_valid": bool(pose_ok),
        "projection_attempted": attempted,
        "projection_projected": projected,
        "reason": reason,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO human detection on RTSP video and fuse with ORB-SLAM pose UDP."
    )
    parser.add_argument("--rtsp-url", default=RTSP_URL)
    parser.add_argument("--model", default=YOLO_MODEL_PATH)
    parser.add_argument("--device", default=DEVICE, help='CUDA device like "0", or "cpu".')
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--person-class", type=int, default=PERSON_CLASS_ID)
    parser.add_argument("--track", action=argparse.BooleanOptionalAction, default=USE_YOLO_TRACKING)
    parser.add_argument("--pose-listen-ip", default=POSE_LISTEN_IP)
    parser.add_argument("--pose-port", type=int, default=POSE_PORT)
    parser.add_argument(
        "--pose-format", choices=("auto", "orbslam-text", "json"), default=POSE_FORMAT
    )
    parser.add_argument("--max-pose-age-s", type=float, default=MAX_POSE_AGE_SECONDS)
    parser.add_argument("--pose-scale", type=float, default=POSE_SCALE)
    parser.add_argument("--swap-yz", action="store_true", default=SWAP_YZ)
    parser.add_argument("--invert-x", action="store_true", default=INVERT_X)
    parser.add_argument("--invert-y", action="store_true", default=INVERT_Y)
    parser.add_argument("--invert-z", action="store_true", default=INVERT_Z)
    parser.add_argument("--output-host", default=UNITY_OUTPUT_HOST)
    parser.add_argument("--output-port", type=int, default=UNITY_OUTPUT_PORT)
    parser.add_argument("--default-hfov-deg", type=float, default=DEFAULT_HFOV_DEG)
    parser.add_argument("--default-camera-height-m", type=float, default=DEFAULT_CAMERA_HEIGHT_M)
    parser.add_argument("--ground-y", type=float, default=GROUND_Y)
    parser.add_argument(
        "--pose-angles-in-degrees", action="store_true", default=POSE_ANGLES_IN_DEGREES
    )
    parser.add_argument("--yaw-offset-deg", type=float, default=YAW_OFFSET_DEG)
    parser.add_argument("--pitch-offset-deg", type=float, default=PITCH_OFFSET_DEG)
    parser.add_argument("--roll-offset-deg", type=float, default=ROLL_OFFSET_DEG)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=SHOW_WINDOW)
    parser.add_argument("--reconnect-delay-s", type=float, default=RECONNECT_DELAY_SECONDS)
    parser.add_argument("--drop-stale-grabs", type=int, default=DROP_STALE_GRABS)
    parser.add_argument("--logs", action="store_true", help="Write runtime logs under --log-root.")
    parser.add_argument(
        "--log-root",
        default=LOG_ROOT,
        help="Directory where logs/run_* folders are written.",
    )
    parser.add_argument(
        "--wait-for-video",
        action="store_true",
        help="Wait for the Linux RTSP publisher before starting normal processing/logging.",
    )
    parser.add_argument(
        "--wait-timeout-s",
        type=float,
        default=WAIT_TIMEOUT_SECONDS,
        help="Maximum seconds to wait for RTSP when --wait-for-video is set. 0 waits forever.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logger = None
    if args.logs:
        if OrbslamFusionRunLogger is None:
            raise RuntimeError(
                "--logs requires orbslam_fusion_testing.py to be in the same folder "
                "as rtsp_yolo_orbslam_fusion.py"
            )
        logger = OrbslamFusionRunLogger(
            log_root=args.log_root,
            run_mode="wait_for_video" if args.wait_for_video else "live",
            args_snapshot=vars(args),
        )
        print(f"[logs] Runtime logging enabled. Root: {args.log_root}")

    pose_receiver = LatestPoseReceiver(
        listen_ip=args.pose_listen_ip,
        listen_port=args.pose_port,
        default_camera_height_m=args.default_camera_height_m,
        pose_scale=args.pose_scale,
        pose_format=args.pose_format,
        swap_yz=args.swap_yz,
        invert_x=args.invert_x,
        invert_y=args.invert_y,
        invert_z=args.invert_z,
        error_callback=logger.log_error if logger is not None else None,
    )
    pose_receiver.start()

    out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap: cv2.VideoCapture | None = None
    rtsp_reconnect_count = 0

    print(f"[video] Opening RTSP stream: {args.rtsp_url}")
    print(f"[video] Low-latency OpenCV/FFmpeg options: {RTSP_CAPTURE_OPTIONS}")
    print(f"[video] Dropping {args.drop_stale_grabs} stale grab(s) before each read")

    try:
        if args.wait_for_video:
            cap = open_rtsp_capture_with_wait(
                rtsp_url=args.rtsp_url,
                reconnect_delay_s=args.reconnect_delay_s,
                wait_timeout_s=args.wait_timeout_s,
                logger=logger,
            )
        else:
            cap = open_rtsp_capture(args.rtsp_url)

        print(f"[yolo] Loading model: {args.model}")
        model = YOLO(args.model)

        frame_id = 0
        print(f"[udp] Sending fused packets to {args.output_host}:{args.output_port}")

        while True:
            frame_start_perf = time.perf_counter()
            for _ in range(max(0, args.drop_stale_grabs)):
                cap.grab()

            ret, frame = cap.read()
            if not ret:
                print("\n[video] RTSP read failed. Reconnecting...")
                if logger is not None:
                    logger.log_error(
                        stage="rtsp_read",
                        message="RTSP read failed",
                        frame_id=frame_id,
                    )
                cap.release()
                time.sleep(args.reconnect_delay_s)
                rtsp_reconnect_count += 1
                if logger is not None:
                    logger.log_error(
                        stage="rtsp_reconnect",
                        message="Reconnecting RTSP capture",
                        frame_id=frame_id,
                        extra={"rtsp_reconnect_count": rtsp_reconnect_count},
                    )
                if args.wait_for_video:
                    cap = open_rtsp_capture_with_wait(
                        rtsp_url=args.rtsp_url,
                        reconnect_delay_s=args.reconnect_delay_s,
                        wait_timeout_s=args.wait_timeout_s,
                        logger=logger,
                    )
                else:
                    cap = open_rtsp_capture(args.rtsp_url)
                continue

            frame_id += 1
            height, width = frame.shape[:2]
            now = time.time()

            pose, raw_pose = pose_receiver.get_latest(args.max_pose_age_s)
            pose_ok = pose is not None and pose.pose_valid

            infer = model.track if args.track else model.predict
            infer_kwargs = {
                "imgsz": args.imgsz,
                "conf": args.conf,
                "iou": args.iou,
                "classes": [args.person_class],
                "device": args.device,
                "verbose": False,
            }
            if args.track:
                infer_kwargs["persist"] = True

            try:
                results = infer(frame, **infer_kwargs)
            except Exception as exc:
                if logger is not None:
                    logger.log_error(
                        stage="yolo_inference",
                        message="YOLO inference failed",
                        frame_id=frame_id,
                        exception=exc,
                    )
                raise

            detections: list[dict[str, Any]] = []
            boxes = results[0].boxes if results else None

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int).tolist()
                else:
                    track_ids = [None] * len(xyxy)

                for i, (box, conf, cls_id, track_id) in enumerate(
                    zip(xyxy, confs, clss, track_ids)
                ):
                    x1, y1, x2, y2 = [float(v) for v in box]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    bw = x2 - x1
                    bh = y2 - y1
                    foot_x = cx
                    foot_y = y2

                    world_valid = False
                    world_x = world_y = world_z = None
                    if pose_ok and pose is not None:
                        try:
                            world_valid, world_x, world_y, world_z = project_pixel_to_ground(
                                u=foot_x,
                                v=foot_y,
                                image_width=width,
                                image_height=height,
                                pose=pose,
                                default_hfov_deg=args.default_hfov_deg,
                                ground_y=args.ground_y,
                                angles_in_degrees=args.pose_angles_in_degrees,
                                yaw_offset_deg=args.yaw_offset_deg,
                                pitch_offset_deg=args.pitch_offset_deg,
                                roll_offset_deg=args.roll_offset_deg,
                            )
                        except Exception as exc:
                            if logger is not None:
                                logger.log_error(
                                    stage="projection",
                                    message="Ground-plane projection failed",
                                    frame_id=frame_id,
                                    exception=exc,
                                    extra={"detection_index": i},
                                )

                    detections.append(
                        {
                            "id": (
                                int(track_id)
                                if track_id is not None
                                else int(frame_id * 1000 + i)
                            ),
                            "cls": int(cls_id),
                            "conf": float(conf),
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "cx": float(cx),
                            "cy": float(cy),
                            "w": float(bw),
                            "h": float(bh),
                            "foot_x": float(foot_x),
                            "foot_y": float(foot_y),
                            "world_valid": bool(world_valid),
                            "world_x": world_x,
                            "world_y": world_y,
                            "world_z": world_z,
                        }
                    )

            pose_packet = build_pose_packet(
                pose=pose,
                pose_ok=bool(pose_ok),
                default_hfov_deg=args.default_hfov_deg,
            )
            slam_packet = build_slam_block(
                pose=pose,
                raw_pose=raw_pose,
                pose_ok=bool(pose_ok),
                now=now,
            )
            fusion_status = build_fusion_status(
                pose_ok=bool(pose_ok),
                raw_pose=raw_pose,
                detections=detections,
            )

            packet = {
                "frame_id": frame_id,
                "timestamp": now,
                "width": width,
                "height": height,
                "detections": detections,
                "pose": pose_packet,
                "slam": slam_packet,
                "fusion_status": fusion_status,
            }

            payload = json.dumps(packet, separators=(",", ":")).encode("utf-8")
            packet_sent = False
            try:
                out_sock.sendto(payload, (args.output_host, args.output_port))
                packet_sent = True
            except OSError as exc:
                if logger is not None:
                    logger.log_error(
                        stage="udp_send",
                        message="Failed to send UDP packet to Unity",
                        frame_id=frame_id,
                        exception=exc,
                        extra={"output_host": args.output_host, "output_port": args.output_port},
                    )

            processing_time_s = time.perf_counter() - frame_start_perf
            if logger is not None:
                logger.log_frame(
                    frame_id=frame_id,
                    timestamp=now,
                    width=width,
                    height=height,
                    processing_time_s=processing_time_s,
                    detections=detections,
                    pose=pose,
                    raw_pose=raw_pose,
                    packet=packet,
                    packet_sent=packet_sent,
                    packet_size_bytes=len(payload),
                    rtsp_reconnect_count=rtsp_reconnect_count,
                    pose_receiver_stats=pose_receiver.get_stats(),
                )

            print(
                f"\rframe={frame_id} humans={len(detections)} pose_ok={pose_ok} "
                f"pose_frame={pose.frame_id if pose else None} "
                f"sent={args.output_host}:{args.output_port}",
                end="",
                flush=True,
            )

            if args.show:
                annotated = draw_overlay(frame.copy(), detections, bool(pose_ok))
                cv2.imshow("XRDrone YOLO + ORB-SLAM fusion", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[exit] Stopped by user.")
    finally:
        if logger is not None:
            summary = logger.finalize(pose_receiver.get_stats())
            if summary:
                print(f"\n[logs] Wrote logs to: {summary.get('run_dir')}")
        pose_receiver.stop()
        if cap is not None:
            cap.release()
        out_sock.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
