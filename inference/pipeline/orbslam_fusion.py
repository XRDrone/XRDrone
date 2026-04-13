"""
orbslam_fusion.py

Middle-man helpers for fusing detector output with ORB-SLAM pose packets.

This module is responsible for:
  - listening for ORB-SLAM UDP packets from an external sender
  - aligning poses to detector frames by frame_id first, then timestamp
  - projecting detection foot points onto a configured ground plane
  - producing transport-friendly status/pose dictionaries for UDP output
"""

from __future__ import annotations

import json
import math
import socket
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class OrbSlamPose:
    frame_id: int
    timestamp: float
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    pose_valid: bool = True
    tracking_state: str = "ok"


class OrbSlamPoseReceiver:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        max_entries: int = 4096,
        stale_timeout_s: float = 0.50,
        max_packet_bytes: int = 65535,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.max_entries = max(32, int(max_entries))
        self.stale_timeout_s = max(0.01, float(stale_timeout_s))
        self.max_packet_bytes = max(1024, int(max_packet_bytes))
        self._sock: socket.socket | None = None
        self._frame_map: OrderedDict[int, OrbSlamPose] = OrderedDict()
        self._latest_pose: OrbSlamPose | None = None
        self._last_error: str | None = None
        self._last_receive_monotonic: float | None = None
        self._waiting_since = time.monotonic()

    def _bind_if_needed(self) -> None:
        if self._sock is not None:
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.setblocking(False)
        self._sock = sock

    def _store_pose(self, pose: OrbSlamPose) -> None:
        self._frame_map[pose.frame_id] = pose
        self._frame_map.move_to_end(pose.frame_id)
        while len(self._frame_map) > self.max_entries:
            self._frame_map.popitem(last=False)
        self._latest_pose = pose
        self._last_receive_monotonic = time.monotonic()

    def poll(self) -> None:
        try:
            self._bind_if_needed()
        except Exception as exc:
            self._last_error = f"ORB-SLAM UDP bind failed: {exc}"
            return

        assert self._sock is not None
        received_any = False
        parse_error: str | None = None

        while True:
            try:
                payload, _addr = self._sock.recvfrom(self.max_packet_bytes)
            except BlockingIOError:
                break
            except Exception as exc:
                self._last_error = f"ORB-SLAM UDP receive failed: {exc}"
                return

            received_any = True
            pose = parse_orbslam_udp_payload(payload)
            if pose is None:
                parse_error = "invalid ORB-SLAM UDP packet"
                continue
            self._store_pose(pose)

        if received_any and self._latest_pose is not None:
            self._last_error = None
            return

        if parse_error and self._latest_pose is None:
            self._last_error = parse_error
            return

        if self._latest_pose is None:
            self._last_error = "waiting for ORB-SLAM UDP packets"
            return

        if self.is_stale():
            self._last_error = "no recent ORB-SLAM UDP packets"
        else:
            self._last_error = None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def latest_pose(self) -> OrbSlamPose | None:
        return self._latest_pose

    def is_stale(self) -> bool:
        if self._last_receive_monotonic is None:
            return True
        return (time.monotonic() - self._last_receive_monotonic) > self.stale_timeout_s

    def match(
        self, *, frame_id: int, timestamp: float, time_tolerance_s: float
    ) -> tuple[OrbSlamPose | None, str]:
        self.poll()

        pose = self._frame_map.get(int(frame_id))
        if pose is not None:
            return pose, "frame_id"

        best_pose: OrbSlamPose | None = None
        best_dt = float("inf")
        ts = float(timestamp)
        for candidate in self._frame_map.values():
            dt = abs(candidate.timestamp - ts)
            if dt < best_dt:
                best_dt = dt
                best_pose = candidate
        if best_pose is not None and best_dt <= float(time_tolerance_s):
            return best_pose, "timestamp"

        if self._latest_pose is not None:
            latest_dt = abs(self._latest_pose.timestamp - ts)
            if latest_dt <= float(time_tolerance_s):
                return self._latest_pose, "latest"

        return None, "none"


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def parse_orbslam_udp_packet(packet: dict[str, Any]) -> OrbSlamPose | None:
    if not isinstance(packet, dict):
        return None

    slam_obj: dict[str, Any]
    if isinstance(packet.get("slam"), dict):
        slam_obj = dict(packet["slam"])
        if "frame_id" not in slam_obj and "frame_id" in packet:
            slam_obj["frame_id"] = packet["frame_id"]
        if "timestamp" not in slam_obj and "timestamp" in packet:
            slam_obj["timestamp"] = packet["timestamp"]
    else:
        slam_obj = packet

    frame_id = _safe_int(slam_obj.get("frame_id"))
    timestamp = _safe_float(slam_obj.get("timestamp"))
    x = _safe_float(slam_obj.get("x"))
    y = _safe_float(slam_obj.get("y"))
    z = _safe_float(slam_obj.get("z"))
    qx = _safe_float(slam_obj.get("qx"))
    qy = _safe_float(slam_obj.get("qy"))
    qz = _safe_float(slam_obj.get("qz"))
    qw = _safe_float(slam_obj.get("qw"))
    if None in (frame_id, timestamp, x, y, z, qx, qy, qz, qw):
        return None

    pose_valid = bool(slam_obj.get("pose_valid", True))
    tracking_state = str(slam_obj.get("tracking_state", "ok")).strip().lower() or "ok"

    return OrbSlamPose(
        frame_id=frame_id,
        timestamp=timestamp,
        x=x,
        y=y,
        z=z,
        qx=qx,
        qy=qy,
        qz=qz,
        qw=qw,
        pose_valid=pose_valid,
        tracking_state=tracking_state,
    )


def parse_orbslam_udp_payload(payload: bytes | str | dict[str, Any]) -> OrbSlamPose | None:
    if isinstance(payload, dict):
        return parse_orbslam_udp_packet(payload)

    try:
        if isinstance(payload, bytes):
            text = payload.decode("utf-8")
        else:
            text = str(payload)
        obj = json.loads(text)
    except Exception:
        return None

    return parse_orbslam_udp_packet(obj)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> list[list[float]]:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    x = qx / norm
    y = qy / norm
    z = qz / norm
    w = qw / norm

    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _rotation_matrix_to_euler_xyz_deg(r: list[list[float]]) -> tuple[float, float, float]:
    sy = math.sqrt(r[0][0] * r[0][0] + r[1][0] * r[1][0])
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(r[2][1], r[2][2])
        pitch = math.atan2(-r[2][0], sy)
        yaw = math.atan2(r[1][0], r[0][0])
    else:
        roll = math.atan2(-r[1][2], r[1][1])
        pitch = math.atan2(-r[2][0], sy)
        yaw = 0.0

    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _mat_vec_mul(mat: list[list[float]], vec: list[float]) -> list[float]:
    return [
        mat[0][0] * vec[0] + mat[0][1] * vec[1] + mat[0][2] * vec[2],
        mat[1][0] * vec[0] + mat[1][1] * vec[1] + mat[1][2] * vec[2],
        mat[2][0] * vec[0] + mat[2][1] * vec[1] + mat[2][2] * vec[2],
    ]


def _derive_intrinsics(
    width: int, height: int, hfov_deg: float
) -> tuple[float, float, float, float]:
    width_f = max(1.0, float(width))
    height_f = max(1.0, float(height))
    hfov_rad = math.radians(max(1.0, float(hfov_deg)))
    fx = width_f / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx
    cx = width_f / 2.0
    cy = height_f / 2.0
    return fx, fy, cx, cy


def _compute_foot_point(det: dict[str, Any], width: int, height: int) -> tuple[float, float]:
    bbox = det.get("bbox_xyxy")
    if isinstance(bbox, list | tuple) and len(bbox) == 4:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        foot_x = ((x1 + x2) * 0.5) / max(1.0, float(width))
        foot_y = y2 / max(1.0, float(height))
        return _clamp01(foot_x), _clamp01(foot_y)

    cx = float(det.get("cx", 0.0))
    cy = float(det.get("cy", 0.0))
    h = float(det.get("h", 0.0))
    return _clamp01(cx), _clamp01(cy + 0.5 * h)


def _passes_projection_filter(
    det: dict[str, Any],
    *,
    projection_classes: tuple[str, ...] | None,
    projection_min_conf: float | None,
) -> bool:
    cls_name = str(det.get("class", "")).lower()
    conf = float(det.get("udp_confidence", det.get("confidence", 0.0)))
    if projection_classes and cls_name not in {c.lower() for c in projection_classes}:
        return False
    if projection_min_conf is not None and conf < float(projection_min_conf):
        return False
    return True


def attach_foot_and_world_from_orbslam(
    detections: list[dict[str, Any]],
    *,
    pose: OrbSlamPose | None,
    width: int,
    height: int,
    hfov_deg: float,
    projection_classes: tuple[str, ...] | None = None,
    projection_min_conf: float | None = None,
    ground_plane_y: float = 0.0,
) -> dict[str, int]:
    projected = 0
    attempted = 0

    fx, fy, cx, cy = _derive_intrinsics(width, height, hfov_deg)
    rotation_cw = None
    camera_world = None
    if pose is not None and pose.pose_valid:
        rotation_cw = _quaternion_to_rotation_matrix(pose.qx, pose.qy, pose.qz, pose.qw)
        camera_world = [float(pose.x), float(pose.y), float(pose.z)]

    for det in detections:
        foot_x, foot_y = _compute_foot_point(det, width, height)
        det["foot_x"] = foot_x
        det["foot_y"] = foot_y
        det["world_valid"] = False
        det["world_x"] = 0.0
        det["world_y"] = 0.0
        det["world_z"] = 0.0

        if pose is None or not pose.pose_valid or rotation_cw is None or camera_world is None:
            continue
        if not _passes_projection_filter(
            det,
            projection_classes=projection_classes,
            projection_min_conf=projection_min_conf,
        ):
            continue

        attempted += 1
        u = foot_x * float(width)
        v = foot_y * float(height)
        ray_camera = [(u - cx) / fx, (v - cy) / fy, 1.0]
        ray_world = _mat_vec_mul(rotation_cw, ray_camera)
        denom = ray_world[1]
        if abs(denom) < 1e-9:
            continue

        t = (float(ground_plane_y) - camera_world[1]) / denom
        if t <= 0.0:
            continue

        world = [
            camera_world[0] + t * ray_world[0],
            camera_world[1] + t * ray_world[1],
            camera_world[2] + t * ray_world[2],
        ]
        det["world_valid"] = True
        det["world_x"] = float(world[0])
        det["world_y"] = float(world[1])
        det["world_z"] = float(world[2])
        projected += 1

    return {"attempted": attempted, "projected": projected}


def build_pose_packet(pose: OrbSlamPose | None, *, hfov_deg: float) -> dict[str, Any]:
    if pose is None or not pose.pose_valid:
        return {
            "x": 0.0,
            "altitude": 0.0,
            "z": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "hfov": float(hfov_deg),
            "markers_used": 0,
            "pose_valid": False,
        }

    yaw, pitch, roll = _rotation_matrix_to_euler_xyz_deg(
        _quaternion_to_rotation_matrix(pose.qx, pose.qy, pose.qz, pose.qw)
    )
    return {
        "x": float(pose.x),
        "altitude": float(pose.y),
        "z": float(pose.z),
        "yaw": float(yaw),
        "pitch": float(pitch),
        "roll": float(roll),
        "hfov": float(hfov_deg),
        "markers_used": 0,
        "pose_valid": True,
    }


def build_slam_packet(
    pose: OrbSlamPose | None, *, tracking_state: str | None = None, match_mode: str
) -> dict[str, Any]:
    if pose is None:
        return {
            "tracking_state": str(tracking_state or "missing"),
            "match_mode": str(match_mode),
            "pose_valid": False,
            "frame_id": None,
            "timestamp": None,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }

    resolved_tracking_state = str(tracking_state or pose.tracking_state or "ok").lower()
    return {
        "tracking_state": resolved_tracking_state,
        "match_mode": str(match_mode),
        "pose_valid": bool(pose.pose_valid),
        "frame_id": int(pose.frame_id),
        "timestamp": float(pose.timestamp),
        "x": float(pose.x),
        "y": float(pose.y),
        "z": float(pose.z),
        "qx": float(pose.qx),
        "qy": float(pose.qy),
        "qz": float(pose.qz),
        "qw": float(pose.qw),
    }


def build_fusion_status(
    *,
    source: str,
    pose: OrbSlamPose | None,
    match_mode: str,
    receiver_error: str | None,
    projection_attempted: int,
    projection_projected: int,
) -> dict[str, Any]:
    error_text = str(receiver_error or "").strip().lower()
    if pose is None:
        if "recent" in error_text or "stale" in error_text:
            slam_tracking = "stale"
        else:
            slam_tracking = "missing"
        projection_state = "unavailable"
    elif not pose.pose_valid:
        slam_tracking = str(pose.tracking_state or "invalid").lower()
        projection_state = "unavailable"
    elif projection_attempted <= 0:
        slam_tracking = str(pose.tracking_state or "ok").lower()
        projection_state = "idle"
    elif projection_projected < projection_attempted:
        slam_tracking = str(pose.tracking_state or "ok").lower()
        projection_state = "partial"
    else:
        slam_tracking = str(pose.tracking_state or "ok").lower()
        projection_state = "ok"

    reasons: list[str] = []
    if pose is None:
        reasons.append("no aligned ORB-SLAM pose")
    elif not pose.pose_valid:
        reasons.append("received ORB-SLAM packet with pose_valid=false")
    if receiver_error:
        reasons.append(str(receiver_error))
    if (
        projection_attempted > 0
        and projection_projected == 0
        and pose is not None
        and pose.pose_valid
    ):
        reasons.append("projection rays missed the ground plane")

    return {
        "source": str(source),
        "slam_tracking": slam_tracking,
        "match_mode": str(match_mode),
        "projection_state": projection_state,
        "pose_valid": bool(pose is not None and pose.pose_valid),
        "projection_attempted": int(projection_attempted),
        "projection_projected": int(projection_projected),
        "reason": "; ".join(reasons),
    }


def build_failure_overlay_lines(status: dict[str, Any]) -> list[str]:
    lines = [
        f"SLAM: {str(status.get('slam_tracking', 'unknown')).upper()}",
        f"Match: {str(status.get('match_mode', 'none')).upper()}",
        (
            "Projection: "
            f"{str(status.get('projection_state', 'unknown')).upper()} "
            f"({int(status.get('projection_projected', 0))}/"
            f"{int(status.get('projection_attempted', 0))})"
        ),
    ]
    reason = str(status.get("reason", "")).strip()
    if reason:
        lines.append(reason)
    return lines


__all__ = [
    "OrbSlamPose",
    "OrbSlamPoseReceiver",
    "attach_foot_and_world_from_orbslam",
    "build_failure_overlay_lines",
    "build_fusion_status",
    "build_pose_packet",
    "build_slam_packet",
    "parse_orbslam_udp_packet",
    "parse_orbslam_udp_payload",
]
