"""
aruco_pose.py

Pure ArUco camera-pose estimation for XRDrone prerecorded-video runs.
This module is dedicated to ArUco marker pose estimation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class ArucoPoseResult:
    pose_packet: dict[str, Any]
    marker_packet: dict[str, Any]
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


def _safe_aruco_dict(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is unavailable. Install opencv-contrib-python.")
    if not hasattr(cv2.aruco, name):
        valid = sorted(k for k in dir(cv2.aruco) if k.startswith("DICT_"))
        raise ValueError(f"Unknown ArUco dictionary {name!r}. Valid examples: {valid[:8]}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _make_detector(dictionary, corner_refinement: str):
    params = cv2.aruco.DetectorParameters()
    refinement = str(corner_refinement or "none").strip().lower()
    refine_map = {
        "none": getattr(cv2.aruco, "CORNER_REFINE_NONE", 0),
        "subpix": getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 1),
        "contour": getattr(cv2.aruco, "CORNER_REFINE_CONTOUR", 2),
        "apriltag": getattr(cv2.aruco, "CORNER_REFINE_APRILTAG", 3),
    }
    params.cornerRefinementMethod = refine_map.get(refinement, refine_map["none"])
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params)
    return None, dictionary, params


def _detect_markers(detector, frame_bgr: np.ndarray):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(detector, "detectMarkers"):
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        _, dictionary, params = detector
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    return corners, ids, rejected


def camera_matrix_from_hfov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    hfov_rad = math.radians(float(hfov_deg))
    if hfov_rad <= 0.0 or hfov_rad >= math.pi:
        raise ValueError(f"Invalid horizontal FOV: {hfov_deg}")
    fx = float(width) / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def marker_corners_world(
    center_xyz: tuple[float, float, float], marker_size_m: float
) -> np.ndarray:
    """
    Return the four world-space corners for one marker.

    The XRDrone ArUco layout used here assumes each marker lies flat on the Y=0
    plane. Marker centers are given as (x, y, z), and the square extends along
    the X/Z axes.
    """
    x, y, z = (float(v) for v in center_xyz)
    half = float(marker_size_m) * 0.5
    return np.array(
        [
            [x - half, y, z + half],
            [x + half, y, z + half],
            [x + half, y, z - half],
            [x - half, y, z - half],
        ],
        dtype=np.float64,
    )


def _rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    return qx / norm, qy / norm, qz / norm, qw / norm


def _solve_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, str]:
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)

    preferred_flags: list[tuple[str, int]] = []
    if hasattr(cv2, "SOLVEPNP_SQPNP") and len(object_points) >= 6:
        preferred_flags.append(("sqpnp", cv2.SOLVEPNP_SQPNP))
    if hasattr(cv2, "SOLVEPNP_IPPE") and len(object_points) >= 4:
        preferred_flags.append(("ippe", cv2.SOLVEPNP_IPPE))
    preferred_flags.append(("iterative", cv2.SOLVEPNP_ITERATIVE))

    last_error: Exception | None = None
    for solver_name, flag in preferred_flags:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=flag,
            )
            if ok:
                try:
                    if hasattr(cv2, "solvePnPRefineLM") and len(object_points) >= 4:
                        rvec, tvec = cv2.solvePnPRefineLM(
                            object_points,
                            image_points,
                            camera_matrix,
                            dist_coeffs,
                            rvec,
                            tvec,
                        )
                except Exception:
                    pass
                return True, rvec, tvec, solver_name
        except Exception as exc:  # try the next solver
            last_error = exc

    if last_error is not None:
        return False, None, None, f"failed:{last_error}"
    return False, None, None, "failed"


def _reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float | None:
    try:
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        image_points = image_points.reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))
    except Exception:
        return None


class ArucoPoseEstimator:
    def __init__(
        self,
        *,
        marker_world_positions: dict[int, tuple[float, float, float]],
        marker_size_m: float,
        aruco_dict_name: str,
        hfov_deg: float,
        corner_refinement: str = "subpix",
    ) -> None:
        self.marker_world_positions = {
            int(k): tuple(float(v) for v in xyz) for k, xyz in marker_world_positions.items()
        }
        self.marker_size_m = float(marker_size_m)
        self.aruco_dict_name = str(aruco_dict_name)
        self.hfov_deg = float(hfov_deg)
        dictionary = _safe_aruco_dict(self.aruco_dict_name)
        self.detector = _make_detector(dictionary, corner_refinement)

    def estimate(
        self,
        frame_bgr: np.ndarray,
        *,
        frame_id: int,
        timestamp: float,
        timestamp_video_s: float | None,
        draw: bool = False,
    ) -> ArucoPoseResult:
        height, width = frame_bgr.shape[:2]
        camera_matrix = camera_matrix_from_hfov(width, height, self.hfov_deg)
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        corners, ids, rejected = _detect_markers(self.detector, frame_bgr)
        marker_ids = [] if ids is None else [int(v) for v in ids.flatten().tolist()]
        known_ids: list[int] = []
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        marker_entries: list[dict[str, Any]] = []

        if ids is not None and len(ids) > 0:
            if draw:
                cv2.aruco.drawDetectedMarkers(frame_bgr, corners, ids)
            for marker_id, marker_corners in zip(marker_ids, corners, strict=False):
                pts2 = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
                known = marker_id in self.marker_world_positions
                marker_entries.append(
                    {
                        "id": marker_id,
                        "known": bool(known),
                        "image_corners": pts2.round(3).tolist(),
                        "world_center": list(self.marker_world_positions.get(marker_id, ())),
                    }
                )
                if not known:
                    continue
                known_ids.append(marker_id)
                object_points.append(
                    marker_corners_world(
                        self.marker_world_positions[marker_id],
                        self.marker_size_m,
                    )
                )
                image_points.append(pts2)

        base = {
            "frame_id": int(frame_id),
            "timestamp": float(timestamp),
            "timestamp_video_s": None if timestamp_video_s is None else float(timestamp_video_s),
            "hfov_deg": float(self.hfov_deg),
            "marker_size_m": float(self.marker_size_m),
            "aruco_dict": self.aruco_dict_name,
            "pose_valid": False,
            "markers_detected": int(len(marker_ids)),
            "known_markers_detected": int(len(known_ids)),
            "markers_used": 0,
            "marker_ids_detected": marker_ids,
            "marker_ids_used": [],
            "solver": "none",
            "reprojection_error_px": None,
            "x": None,
            "y": None,
            "z": None,
            "qx": None,
            "qy": None,
            "qz": None,
            "qw": None,
            "rvec_world_to_camera": None,
            "tvec_world_to_camera": None,
        }

        marker_packet = {
            "frame_id": int(frame_id),
            "timestamp": float(timestamp),
            "timestamp_video_s": None if timestamp_video_s is None else float(timestamp_video_s),
            "markers": marker_entries,
            "rejected_count": int(0 if rejected is None else len(rejected)),
        }

        if not object_points:
            return ArucoPoseResult(base, marker_packet, None, None, camera_matrix, dist_coeffs)

        obj = np.concatenate(object_points, axis=0).astype(np.float64)
        img = np.concatenate(image_points, axis=0).astype(np.float64)
        ok, rvec, tvec, solver_name = _solve_pnp(obj, img, camera_matrix, dist_coeffs)
        if not ok or rvec is None or tvec is None:
            base["solver"] = solver_name
            return ArucoPoseResult(base, marker_packet, None, None, camera_matrix, dist_coeffs)

        rotation_world_to_camera, _ = cv2.Rodrigues(rvec)
        rotation_camera_to_world = rotation_world_to_camera.T
        camera_world = -rotation_camera_to_world @ tvec.reshape(3, 1)
        qx, qy, qz, qw = _rotation_matrix_to_quaternion_xyzw(rotation_camera_to_world)
        reproj = _reprojection_error(obj, img, rvec, tvec, camera_matrix, dist_coeffs)

        base.update(
            {
                "pose_valid": True,
                "markers_used": int(len(known_ids)),
                "marker_ids_used": known_ids,
                "solver": solver_name,
                "reprojection_error_px": reproj,
                "x": float(camera_world[0, 0]),
                "y": float(camera_world[1, 0]),
                "z": float(camera_world[2, 0]),
                "qx": float(qx),
                "qy": float(qy),
                "qz": float(qz),
                "qw": float(qw),
                "rvec_world_to_camera": rvec.reshape(-1).astype(float).tolist(),
                "tvec_world_to_camera": tvec.reshape(-1).astype(float).tolist(),
            }
        )
        return ArucoPoseResult(base, marker_packet, rvec, tvec, camera_matrix, dist_coeffs)
