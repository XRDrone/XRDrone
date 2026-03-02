"""
pose_estimator.py

Estimates a simple camera pose from ArUco markers for inclusion in the UDP JSON packet.

Produces a pose dict with:
  - x: float (camera world X)
  - altitude: float (camera world Y)
  - z: float (camera world Z)
  - yaw: float (degrees)
  - pitch: float (degrees)
  - roll: float (degrees)
  - hfov: float (degrees, horizontal field of view used to approximate intrinsics)
  - markers_used: int (# of markers contributing to pose)
  - pose_valid: bool (True if pose was computed this frame)

Notes
-----
- Uses an HFOV-based approximate camera matrix (demo-quality).
- Requires OpenCV ArUco support (typically provided by opencv-contrib-python).
  If ArUco is unavailable, pose_valid will always be False.
- Marker world points assume each marker lies on the world plane Y=0.

This module also exposes an optional PoseSolution (K, R_wc, C_w) which can be
used to project image pixels to the ground plane (Y=0) via ray-plane intersection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def _as_float3(v) -> np.ndarray:
    a = np.array(v, dtype=np.float64).reshape(-1)
    if a.size < 3:
        out = np.zeros(3, dtype=np.float64)
        out[: a.size] = a
        return out
    return a[:3].astype(np.float64)


def _hfov_camera_matrix(width: int, height: int, hfov_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate camera intrinsics from HFOV. Returns (K, dist_coeffs)."""
    w = max(1, int(width))
    h = max(1, int(height))
    hfov_rad = np.deg2rad(float(hfov_deg))

    fx = (w / 2.0) / np.tan(hfov_rad / 2.0)
    fy = fx
    cx = w / 2.0
    cy = h / 2.0

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)  # assume no distortion
    return K, dist


def _marker_object_points(size_m: float) -> np.ndarray:
    """Marker corners in marker-local/world plane coordinates (Y=0 plane)."""
    h = float(size_m) / 2.0
    return np.array(
        [
            [-h, 0.0, h],
            [h, 0.0, h],
            [h, 0.0, -h],
            [-h, 0.0, -h],
        ],
        dtype=np.float64,
    )


def _ypr_from_R_wc(R_wc: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix (world->camera) into (yaw, pitch, roll) degrees.

    Convention:
      R_cw = R_wc^T
      yaw   = atan2(R_cw[0,2], R_cw[2,2])
      pitch = -asin(-R_cw[1,2])
      roll  = atan2(R_cw[1,0], R_cw[1,1])
    """
    R_cw = R_wc.T
    yaw = float(np.degrees(np.arctan2(R_cw[0, 2], R_cw[2, 2])))
    pitch = float(-np.degrees(np.arcsin(np.clip(-R_cw[1, 2], -1.0, 1.0))))
    roll = float(np.degrees(np.arctan2(R_cw[1, 0], R_cw[1, 1])))
    return yaw, pitch, roll


@dataclass(frozen=True)
class PoseSolution:
    """Extra per-frame camera pose data needed for 3D registration."""

    C_w: np.ndarray  # (3,) camera position in world
    R_wc: np.ndarray  # (3,3) rotation from world -> camera
    K: np.ndarray  # (3,3) intrinsics

    def pixel_ray_in_world(self, u_px: float, v_px: float) -> Optional[np.ndarray]:
        """Back-project a pixel into a unit direction vector in world coordinates."""
        try:
            Kinv = np.linalg.inv(self.K)
        except Exception:
            return None

        d_c = Kinv @ np.array([float(u_px), float(v_px), 1.0], dtype=np.float64)
        norm = float(np.linalg.norm(d_c))
        if norm <= 0.0:
            return None
        d_c = d_c / norm

        # camera->world rotation is R_cw = R_wc^T
        d_w = self.R_wc.T @ d_c
        norm_w = float(np.linalg.norm(d_w))
        if norm_w <= 0.0:
            return None
        return d_w / norm_w

    def intersect_plane_y0(self, u_px: float, v_px: float, *, eps: float = 1e-8) -> Optional[np.ndarray]:
        """Intersect pixel ray with plane Y=0; returns world point or None."""
        d_w = self.pixel_ray_in_world(u_px, v_px)
        if d_w is None:
            return None

        denom = float(d_w[1])
        if abs(denom) < float(eps):
            return None

        # Ray: P(t) = C_w + t * d_w. Plane: Y=0 -> C_w.y + t*d_w.y = 0
        t = (0.0 - float(self.C_w[1])) / denom
        if t <= 0.0:
            return None

        return self.C_w + t * d_w


@dataclass
class ArucoPoseEstimator:
    enabled: bool = True
    hfov_deg: float = 84.0
    marker_size_m: float = 0.1645
    marker_world_positions: Optional[Mapping[int, Any]] = None
    aruco_dict_name: str = "DICT_4X4_50"

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.hfov_deg = float(self.hfov_deg)
        self.marker_size_m = float(self.marker_size_m)
        self.aruco_dict_name = str(self.aruco_dict_name or "DICT_4X4_50")

        self._last_pose_numbers = {
            "x": 0.0,
            "altitude": 0.0,
            "z": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
        }

        self._aruco = None
        self._dict = None
        self._params = None
        self._detector = None

        if cv2 is None:
            return

        aruco = getattr(cv2, "aruco", None)
        if aruco is None:
            return

        self._aruco = aruco

        dict_id = getattr(aruco, self.aruco_dict_name, None)
        if dict_id is None:
            dict_id = getattr(aruco, "DICT_4X4_50", None)

        try:
            self._dict = aruco.getPredefinedDictionary(dict_id)
        except Exception:
            self._dict = None

        try:
            self._params = aruco.DetectorParameters()
        except Exception:
            self._params = None

        try:
            if self._dict is not None and self._params is not None:
                self._detector = aruco.ArucoDetector(self._dict, self._params)
        except Exception:
            self._detector = None

    def default_pose(self) -> Dict[str, Any]:
        """
        Return a pose dict in the required UDP schema.
        Always returns pose_valid=False and markers_used=0.
        """
        return {
            "x": float(self._last_pose_numbers["x"]),
            "altitude": float(self._last_pose_numbers["altitude"]),
            "z": float(self._last_pose_numbers["z"]),
            "yaw": float(self._last_pose_numbers["yaw"]),
            "pitch": float(self._last_pose_numbers["pitch"]),
            "roll": float(self._last_pose_numbers["roll"]),
            "hfov": float(self.hfov_deg),
            "markers_used": 0,
            "pose_valid": False,
        }

    def estimate(self, frame_bgr: np.ndarray, *, draw: bool = False) -> Dict[str, Any]:
        """Backwards-compatible API: returns pose dict only."""
        pose, _sol = self.estimate_with_solution(frame_bgr, draw=draw)
        return pose

    def estimate_with_solution(
        self, frame_bgr: np.ndarray, *, draw: bool = False
    ) -> Tuple[Dict[str, Any], Optional[PoseSolution]]:
        """
        Estimate pose for the provided frame.

        Returns:
          (pose_dict, pose_solution)

        pose_dict matches the UDP schema.
        pose_solution is optional and contains (K, R_wc, C_w) for 3D registration.
        """
        if not self.enabled:
            return self.default_pose(), None

        if cv2 is None or self._aruco is None or self._dict is None:
            return self.default_pose(), None

        if frame_bgr is None:
            return self.default_pose(), None

        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return self.default_pose(), None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        try:
            if self._detector is not None:
                corners, ids, _rej = self._detector.detectMarkers(gray)
            else:
                corners, ids, _rej = self._aruco.detectMarkers(gray, self._dict, parameters=self._params)
        except Exception:
            return self.default_pose(), None

        if ids is None or len(ids) == 0:
            return self.default_pose(), None

        if draw:
            try:
                self._aruco.drawDetectedMarkers(frame_bgr, corners, ids)
            except Exception:
                pass

        out = self._estimate_from_markers(corners, ids, width=w, height=h)
        if out is None:
            return self.default_pose(), None

        pose, sol = out

        self._last_pose_numbers.update(
            {
                "x": float(pose["x"]),
                "altitude": float(pose["altitude"]),
                "z": float(pose["z"]),
                "yaw": float(pose["yaw"]),
                "pitch": float(pose["pitch"]),
                "roll": float(pose["roll"]),
            }
        )

        return pose, sol

    def _estimate_from_markers(
        self, corners, ids, *, width: int, height: int
    ) -> Optional[Tuple[Dict[str, Any], PoseSolution]]:
        """Multi-marker solvePnP pose. Returns (pose_dict, PoseSolution) or None."""
        if cv2 is None:
            return None

        marker_world = self.marker_world_positions or {0: (0.0, 0.0, 0.0)}
        base_pts = _marker_object_points(self.marker_size_m)

        obj_pts_list = []
        img_pts_list = []

        try:
            flat_ids = ids.flatten()
        except Exception:
            return None

        for i, mid in enumerate(flat_ids):
            try:
                mid_i = int(mid)
            except Exception:
                continue
            if mid_i not in marker_world:
                continue

            world_offset = _as_float3(marker_world[mid_i])
            world_pts = base_pts + world_offset  # (4,3)
            obj_pts_list.append(world_pts)
            img_pts_list.append(np.asarray(corners[i][0], dtype=np.float64))  # (4,2)

        n_markers = len(obj_pts_list)
        if n_markers == 0:
            return None

        obj_pts = np.concatenate(obj_pts_list).astype(np.float64)
        img_pts = np.concatenate(img_pts_list).astype(np.float64)

        K, dist = _hfov_camera_matrix(width, height, self.hfov_deg)

        try:
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        except Exception:
            return None

        if not bool(ok):
            return None

        try:
            R_wc, _ = cv2.Rodrigues(rvec)  # world->camera
        except Exception:
            return None

        try:
            C_w = (-R_wc.T @ tvec).reshape(-1)  # camera position in world
        except Exception:
            return None

        if C_w.size < 3:
            C_w = np.pad(C_w, (0, 3 - C_w.size))

        yaw, pitch, roll = _ypr_from_R_wc(R_wc)

        pose = {
            "x": float(C_w[0]),
            "altitude": float(C_w[1]),
            "z": float(C_w[2]),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "hfov": float(self.hfov_deg),
            "markers_used": int(n_markers),
            "pose_valid": True,
        }

        sol = PoseSolution(C_w=C_w.astype(np.float64), R_wc=R_wc.astype(np.float64), K=K.astype(np.float64))
        return pose, sol