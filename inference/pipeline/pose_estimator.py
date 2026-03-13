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
- UDP pose output stays unchanged even when different solver/refinement paths are used.

This module also exposes an optional PoseSolution (K, R_wc, C_w) which can be
used to project image pixels to the ground plane (Y=0) via ray-plane intersection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


VALID_POSE_USE_CASES = {"auto", "single_marker", "multi_marker_board"}
VALID_INIT_SOLVERS = {"iterative", "ippe_square", "ransac", "sqpnp"}
VALID_REFINERS = {"none", "lm", "vvs"}


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


def _polygon_area_2d(points_xy: np.ndarray) -> float:
    """Return absolute polygon area for a 2D quadrilateral."""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


@dataclass(frozen=True)
class PoseSolution:
    """Extra per-frame camera pose data needed for 3D registration."""

    C_w: np.ndarray  # (3,) camera position in world
    R_wc: np.ndarray  # (3,3) rotation from world -> camera
    K: np.ndarray  # (3,3) intrinsics
    K_inv: Optional[np.ndarray] = None  # (3,3) cached intrinsics inverse
    R_cw: Optional[np.ndarray] = None  # (3,3) rotation from camera -> world

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "K_inv", np.linalg.inv(np.asarray(self.K, dtype=np.float64)))
        except Exception:
            object.__setattr__(self, "K_inv", None)

        try:
            object.__setattr__(self, "R_cw", np.asarray(self.R_wc, dtype=np.float64).T)
        except Exception:
            object.__setattr__(self, "R_cw", None)

    def pixel_ray_in_world(self, u_px: float, v_px: float) -> Optional[np.ndarray]:
        """Back-project a pixel into a unit direction vector in world coordinates."""
        if self.K_inv is None or self.R_cw is None:
            return None

        d_c = self.K_inv @ np.array([float(u_px), float(v_px), 1.0], dtype=np.float64)
        norm = float(np.linalg.norm(d_c))
        if norm <= 0.0:
            return None
        d_c = d_c / norm

        d_w = self.R_cw @ d_c
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


@dataclass(frozen=True)
class MarkerObservation:
    """One detected marker with known world placement."""

    marker_id: int
    object_points: np.ndarray  # (4,3)
    image_points: np.ndarray  # (4,2)
    image_area: float


@dataclass
class ArucoPoseEstimator:
    enabled: bool = True
    hfov_deg: float = 84.0
    marker_size_m: float = 0.1645
    marker_world_positions: Optional[Mapping[int, Any]] = None
    aruco_dict_name: str = "DICT_4X4_50"
    use_case: str = "auto"
    single_init_solver: str = "ippe_square"
    multi_init_solver: str = "sqpnp"
    refiner: str = "vvs"
    enable_refinement: bool = True
    min_markers_for_multi: int = 2
    corner_refinement: str = "none"
    ransac_reproj_threshold_px: float = 4.0
    ransac_confidence: float = 0.99
    ransac_iterations: int = 100

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.hfov_deg = float(self.hfov_deg)
        self.marker_size_m = float(self.marker_size_m)
        self.aruco_dict_name = str(self.aruco_dict_name or "DICT_4X4_50")
        self.use_case = str(self.use_case or "auto").strip().lower()
        self.single_init_solver = str(self.single_init_solver or "ippe_square").strip().lower()
        self.multi_init_solver = str(self.multi_init_solver or "sqpnp").strip().lower()
        self.refiner = str(self.refiner or "vvs").strip().lower()
        self.enable_refinement = bool(self.enable_refinement)
        self.min_markers_for_multi = max(2, int(self.min_markers_for_multi))
        self.corner_refinement = str(self.corner_refinement or "none").strip().lower()
        self.ransac_reproj_threshold_px = max(0.1, float(self.ransac_reproj_threshold_px))
        self.ransac_confidence = min(max(float(self.ransac_confidence), 0.0), 1.0)
        self.ransac_iterations = max(1, int(self.ransac_iterations))

        if self.use_case not in VALID_POSE_USE_CASES:
            self.use_case = "auto"
        if self.single_init_solver not in VALID_INIT_SOLVERS:
            self.single_init_solver = "ippe_square"
        if self.multi_init_solver not in VALID_INIT_SOLVERS:
            self.multi_init_solver = "sqpnp"
        if self.refiner not in VALID_REFINERS:
            self.refiner = "vvs"

        self._last_pose_numbers = {
            "x": 0.0,
            "altitude": 0.0,
            "z": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
        }
        self._last_visible_known_markers = 0
        self._last_selected_mode = "none"
        self._last_overlay_text = "ArUco: no known markers"

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
            self._configure_corner_refinement(self._params)
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

    def get_visible_known_marker_count(self) -> int:
        """Return the number of known-layout ArUco markers visible in the last frame."""
        return int(self._last_visible_known_markers)

    def get_last_selected_mode(self) -> str:
        """Return the pose path used on the last processed frame."""
        return str(self._last_selected_mode)

    def get_pose_mode_overlay_text(self) -> str:
        """Return a user-facing label for the ArUco visibility / pose mode overlay."""
        return str(self._last_overlay_text)

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
            self._update_marker_status(0, "none")
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

    def _configure_corner_refinement(self, params) -> None:
        """Set optional ArUco corner refinement mode when supported by OpenCV."""
        if params is None or self._aruco is None:
            return

        mode_name = {
            "none": "CORNER_REFINE_NONE",
            "subpix": "CORNER_REFINE_SUBPIX",
            "contour": "CORNER_REFINE_CONTOUR",
            "apriltag": "CORNER_REFINE_APRILTAG",
        }.get(self.corner_refinement, "CORNER_REFINE_NONE")

        try:
            mode_value = getattr(self._aruco, mode_name)
        except Exception:
            return

        try:
            params.cornerRefinementMethod = mode_value
        except Exception:
            return

    def _estimate_from_markers(
        self, corners, ids, *, width: int, height: int
    ) -> Optional[Tuple[Dict[str, Any], PoseSolution]]:
        """Dispatch to the configured single- or multi-marker pose path."""
        if cv2 is None:
            return None

        observations = self._collect_observations(corners, ids)
        if not observations:
            self._update_marker_status(0, "none")
            return None

        K, dist = _hfov_camera_matrix(width, height, self.hfov_deg)

        mode = self._choose_mode(len(observations))
        self._update_marker_status(len(observations), mode)

        if mode == "single_marker":
            best = self._select_best_single_marker(observations)
            if best is None:
                return None
            return self._solve_single_marker(best, K, dist)

        result = self._solve_multi_marker_board(observations, K, dist)
        if result is not None:
            return result

        if self.use_case == "auto":
            best = self._select_best_single_marker(observations)
            if best is not None:
                self._update_marker_status(len(observations), "single_marker")
                return self._solve_single_marker(best, K, dist)

        return None

    def _update_marker_status(self, marker_count: int, selected_mode: str) -> None:
        """Track the latest visible-marker count and overlay label without changing UDP schema."""
        count = max(0, int(marker_count))
        mode = str(selected_mode or "none")

        self._last_visible_known_markers = count
        self._last_selected_mode = mode

        if count <= 0:
            self._last_overlay_text = "ArUco: no known markers"
        elif count == 1:
            self._last_overlay_text = "ArUco: single marker"
        else:
            self._last_overlay_text = f"ArUco: multiple markers ({count})"

        if mode == "single_marker" and count > 0:
            self._last_overlay_text += " | mode: single"
        elif mode == "multi_marker_board" and count > 0:
            self._last_overlay_text += " | mode: multi"

    def _choose_mode(self, marker_count: int) -> str:
        """Choose which pose path to use for this frame."""
        if marker_count <= 0:
            return "single_marker"

        if self.use_case == "single_marker":
            return "single_marker"
        if self.use_case == "multi_marker_board":
            return "multi_marker_board"

        if marker_count >= self.min_markers_for_multi:
            return "multi_marker_board"
        return "single_marker"

    def _collect_observations(self, corners, ids) -> List[MarkerObservation]:
        """Convert known detected markers into 3D/2D correspondence groups."""
        marker_world = self.marker_world_positions or {0: (0.0, 0.0, 0.0)}
        base_pts = _marker_object_points(self.marker_size_m)
        out: List[MarkerObservation] = []

        try:
            flat_ids = ids.flatten()
        except Exception:
            return out

        for i, mid in enumerate(flat_ids):
            try:
                marker_id = int(mid)
            except Exception:
                continue
            if marker_id not in marker_world:
                continue

            try:
                image_pts = np.asarray(corners[i][0], dtype=np.float64).reshape(4, 2)
            except Exception:
                continue

            world_offset = _as_float3(marker_world[marker_id])
            object_pts = (base_pts + world_offset).astype(np.float64)
            image_area = _polygon_area_2d(image_pts)

            out.append(
                MarkerObservation(
                    marker_id=marker_id,
                    object_points=object_pts,
                    image_points=image_pts,
                    image_area=image_area,
                )
            )

        return out

    def _select_best_single_marker(self, observations: List[MarkerObservation]) -> Optional[MarkerObservation]:
        """Choose the strongest single visible marker for single-marker pose solving."""
        if not observations:
            return None
        return max(observations, key=lambda obs: (float(obs.image_area), -int(obs.marker_id)))

    def _solve_single_marker(
        self, observation: MarkerObservation, K: np.ndarray, dist: np.ndarray
    ) -> Optional[Tuple[Dict[str, Any], PoseSolution]]:
        """Solve pose from one square marker using the configured single-marker initializer."""
        solver = self.single_init_solver
        if solver == "ransac":
            ok, rvec, tvec = self._solve_pnp_ransac(observation.object_points, observation.image_points, K, dist)
        else:
            flag_name = "SOLVEPNP_IPPE_SQUARE" if solver == "ippe_square" else "SOLVEPNP_ITERATIVE"
            ok, rvec, tvec = self._solve_pnp(observation.object_points, observation.image_points, K, dist, flag_name)

        if not ok or rvec is None or tvec is None:
            if solver == "ippe_square":
                ok, rvec, tvec = self._solve_pnp(
                    observation.object_points,
                    observation.image_points,
                    K,
                    dist,
                    "SOLVEPNP_ITERATIVE",
                )
            if not ok or rvec is None or tvec is None:
                return None

        rvec, tvec = self._refine_pose(observation.object_points, observation.image_points, K, dist, rvec, tvec)
        return self._build_pose_result(rvec, tvec, K, markers_used=1)

    def _solve_multi_marker_board(
        self, observations: List[MarkerObservation], K: np.ndarray, dist: np.ndarray
    ) -> Optional[Tuple[Dict[str, Any], PoseSolution]]:
        """Solve pose from all visible markers together as one known-layout board."""
        if len(observations) < self.min_markers_for_multi:
            return None

        object_points = np.concatenate([obs.object_points for obs in observations], axis=0).astype(np.float64)
        image_points = np.concatenate([obs.image_points for obs in observations], axis=0).astype(np.float64)

        solver = self.multi_init_solver
        if solver == "ransac":
            ok, rvec, tvec = self._solve_pnp_ransac(object_points, image_points, K, dist)
        else:
            if solver == "ippe_square":
                flag_name = "SOLVEPNP_IPPE_SQUARE"
            elif solver == "sqpnp":
                flag_name = "SOLVEPNP_SQPNP"
            else:
                flag_name = "SOLVEPNP_ITERATIVE"

            ok, rvec, tvec = self._solve_pnp(object_points, image_points, K, dist, flag_name)

            if not ok or rvec is None or tvec is None:
                if solver == "ippe_square":
                    ok, rvec, tvec = self._solve_pnp(
                        object_points, image_points, K, dist, "SOLVEPNP_ITERATIVE"
                    )
                elif solver == "sqpnp":
                    ok, rvec, tvec = self._solve_pnp(
                        object_points, image_points, K, dist, "SOLVEPNP_ITERATIVE"
                    )

        if not ok or rvec is None or tvec is None:
            return None

        rvec, tvec = self._refine_pose(object_points, image_points, K, dist, rvec, tvec)
        return self._build_pose_result(rvec, tvec, K, markers_used=len(observations))

    def _solve_pnp(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
        flag_name: str,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Thin wrapper around cv2.solvePnP with a named OpenCV solver flag."""
        if cv2 is None:
            return False, None, None

        try:
            flag_value = getattr(cv2, flag_name)
        except Exception:
            flag_value = getattr(cv2, "SOLVEPNP_ITERATIVE", None)

        if flag_value is None:
            return False, None, None

        try:
            ok, rvec, tvec = cv2.solvePnP(
                np.ascontiguousarray(object_points, dtype=np.float64),
                np.ascontiguousarray(image_points, dtype=np.float64),
                K,
                dist,
                flags=flag_value,
            )
        except Exception:
            return False, None, None

        return bool(ok), rvec, tvec

    def _solve_pnp_ransac(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Thin wrapper around cv2.solvePnPRansac for outlier-robust initialization."""
        if cv2 is None or not hasattr(cv2, "solvePnPRansac"):
            return False, None, None

        try:
            result = cv2.solvePnPRansac(
                np.ascontiguousarray(object_points, dtype=np.float64),
                np.ascontiguousarray(image_points, dtype=np.float64),
                K,
                dist,
                reprojectionError=float(self.ransac_reproj_threshold_px),
                confidence=float(self.ransac_confidence),
                iterationsCount=int(self.ransac_iterations),
            )
        except Exception:
            return False, None, None

        if not isinstance(result, tuple) or len(result) < 3:
            return False, None, None

        ok, rvec, tvec = result[:3]
        return bool(ok), rvec, tvec

    def _refine_pose(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optionally refine pose while preserving the UDP output schema."""
        if cv2 is None or not self.enable_refinement or self.refiner == "none":
            return rvec, tvec

        try:
            if self.refiner == "lm" and hasattr(cv2, "solvePnPRefineLM"):
                rvec, tvec = cv2.solvePnPRefineLM(
                    np.ascontiguousarray(object_points, dtype=np.float64),
                    np.ascontiguousarray(image_points, dtype=np.float64),
                    K,
                    dist,
                    rvec,
                    tvec,
                )
            elif self.refiner == "vvs" and hasattr(cv2, "solvePnPRefineVVS"):
                rvec, tvec = cv2.solvePnPRefineVVS(
                    np.ascontiguousarray(object_points, dtype=np.float64),
                    np.ascontiguousarray(image_points, dtype=np.float64),
                    K,
                    dist,
                    rvec,
                    tvec,
                )
        except Exception:
            return rvec, tvec

        return rvec, tvec

    def _build_pose_result(
        self, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, *, markers_used: int
    ) -> Optional[Tuple[Dict[str, Any], PoseSolution]]:
        """Convert OpenCV pose vectors into the existing UDP pose schema."""
        if cv2 is None:
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
            "markers_used": int(markers_used),
            "pose_valid": True,
        }

        sol = PoseSolution(C_w=C_w.astype(np.float64), R_wc=R_wc.astype(np.float64), K=K.astype(np.float64))
        return pose, sol