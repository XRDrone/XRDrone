"""
motion_smoothing.py

Temporal smoothing utilities for ArUco-based world registration.

Provides:
  - One Euro filtering for vector signals
  - Quaternion SLERP-based adaptive smoothing for camera orientation
  - PoseMotionSmoother for filtering PoseSolution before world projection
  - WorldTrackSmoother for filtering per-track world positions after registration

The UDP schema remains unchanged. Smoothing is applied entirely upstream of
packet formatting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pose_estimator import PoseSolution


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _alpha(cutoff_hz: float, dt: float) -> float:
    cutoff = max(1e-6, float(cutoff_hz))
    dt = max(1e-6, float(dt))
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def _ypr_from_R_wc(R_wc: np.ndarray) -> tuple[float, float, float]:
    """Match the existing pose_estimator yaw/pitch/roll convention."""
    R_cw = np.asarray(R_wc, dtype=np.float64).T
    yaw = float(np.degrees(np.arctan2(R_cw[0, 2], R_cw[2, 2])))
    pitch = float(-np.degrees(np.arcsin(np.clip(-R_cw[1, 2], -1.0, 1.0))))
    roll = float(np.degrees(np.arctan2(R_cw[1, 0], R_cw[1, 1])))
    return yaw, pitch, roll


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_from_rotmat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion [w, x, y, z]."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return _quat_normalize(np.array([w, x, y, z], dtype=np.float64))


def _rotmat_from_quat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = _quat_normalize(q)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    t = _clamp01(t)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        return _quat_normalize(q0 + t * (q1 - q0))

    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    if sin_theta_0 <= 1e-8:
        return q0.copy()

    theta = theta_0 * t
    sin_theta = float(np.sin(theta))

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _quat_normalize(s0 * q0 + s1 * q1)


def _quat_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = abs(float(np.dot(q0, q1)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def _pose_position_params_from_smoothness(smoothness: float) -> tuple[float, float]:
    s = _clamp01(smoothness)
    min_cutoff_hz = 3.5 - 3.1 * s
    beta = 0.03 + 0.32 * s
    return float(min_cutoff_hz), float(beta)


def _pose_rotation_params_from_smoothness(smoothness: float) -> tuple[float, float]:
    s = _clamp01(smoothness)
    min_cutoff_hz = 4.5 - 3.8 * s
    beta = 0.04 + 0.42 * s
    return float(min_cutoff_hz), float(beta)


def _world_params_from_smoothness(smoothness: float) -> tuple[float, float]:
    s = _clamp01(smoothness)
    min_cutoff_hz = 3.0 - 2.6 * s
    beta = 0.02 + 0.24 * s
    return float(min_cutoff_hz), float(beta)


class OneEuroVectorFilter:
    """One Euro filter for scalar or vector numpy signals."""

    def __init__(self, *, min_cutoff_hz: float, beta: float, d_cutoff_hz: float = 1.0) -> None:
        self.min_cutoff_hz = float(min_cutoff_hz)
        self.beta = float(beta)
        self.d_cutoff_hz = float(d_cutoff_hz)
        self.reset()

    def reset(self) -> None:
        self._x_prev: np.ndarray | None = None
        self._dx_hat: np.ndarray | None = None

    def set_params(
        self, *, min_cutoff_hz: float, beta: float, d_cutoff_hz: float | None = None
    ) -> None:
        self.min_cutoff_hz = float(min_cutoff_hz)
        self.beta = float(beta)
        if d_cutoff_hz is not None:
            self.d_cutoff_hz = float(d_cutoff_hz)

    def filter(self, x, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        dt = max(1e-6, float(dt))

        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_hat = np.zeros_like(x, dtype=np.float64)
            return x.copy()

        dx = (x - self._x_prev) / dt
        alpha_d = _alpha(self.d_cutoff_hz, dt)
        assert self._dx_hat is not None
        self._dx_hat = self._dx_hat + alpha_d * (dx - self._dx_hat)

        speed = float(np.linalg.norm(self._dx_hat.reshape(-1)))
        cutoff = self.min_cutoff_hz + self.beta * speed
        alpha_x = _alpha(cutoff, dt)
        x_hat = self._x_prev + alpha_x * (x - self._x_prev)
        self._x_prev = x_hat.copy()
        return x_hat


class OneEuroQuaternionFilter:
    """Adaptive quaternion smoothing using scalar One Euro speed + SLERP."""

    def __init__(self, *, min_cutoff_hz: float, beta: float, d_cutoff_hz: float = 1.0) -> None:
        self.min_cutoff_hz = float(min_cutoff_hz)
        self.beta = float(beta)
        self.d_cutoff_hz = float(d_cutoff_hz)
        self.reset()

    def reset(self) -> None:
        self._q_prev: np.ndarray | None = None
        self._speed_hat: float = 0.0

    def set_params(
        self, *, min_cutoff_hz: float, beta: float, d_cutoff_hz: float | None = None
    ) -> None:
        self.min_cutoff_hz = float(min_cutoff_hz)
        self.beta = float(beta)
        if d_cutoff_hz is not None:
            self.d_cutoff_hz = float(d_cutoff_hz)

    def filter(self, q, dt: float) -> np.ndarray:
        q = _quat_normalize(np.asarray(q, dtype=np.float64).reshape(4))
        dt = max(1e-6, float(dt))

        if self._q_prev is None:
            self._q_prev = q.copy()
            self._speed_hat = 0.0
            return q.copy()

        speed = _quat_angle(self._q_prev, q) / dt
        alpha_d = _alpha(self.d_cutoff_hz, dt)
        self._speed_hat = self._speed_hat + alpha_d * (speed - self._speed_hat)

        cutoff = self.min_cutoff_hz + self.beta * abs(self._speed_hat)
        alpha_q = _alpha(cutoff, dt)
        q_hat = _quat_slerp(self._q_prev, q, alpha_q)
        self._q_prev = q_hat.copy()
        return q_hat


@dataclass
class PoseMotionSmoother:
    """Smooth PoseSolution before pixel-to-ground projection."""

    enabled: bool = True
    smoothness: float = 0.5
    derivative_cutoff_hz: float = 1.0
    reset_timeout_s: float = 0.75
    default_fps: float = 30.0

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.smoothness = _clamp01(self.smoothness)
        self.derivative_cutoff_hz = max(1e-3, float(self.derivative_cutoff_hz))
        self.reset_timeout_s = max(1e-3, float(self.reset_timeout_s))
        self.default_fps = max(1.0, float(self.default_fps))
        self._last_timestamp: float | None = None

        pos_cutoff, pos_beta = _pose_position_params_from_smoothness(self.smoothness)
        rot_cutoff, rot_beta = _pose_rotation_params_from_smoothness(self.smoothness)
        self._position_filter = OneEuroVectorFilter(
            min_cutoff_hz=pos_cutoff,
            beta=pos_beta,
            d_cutoff_hz=self.derivative_cutoff_hz,
        )
        self._rotation_filter = OneEuroQuaternionFilter(
            min_cutoff_hz=rot_cutoff,
            beta=rot_beta,
            d_cutoff_hz=self.derivative_cutoff_hz,
        )

    def reset(self) -> None:
        self._last_timestamp = None
        self._position_filter.reset()
        self._rotation_filter.reset()

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self.enabled != enabled:
            self.enabled = enabled
            self.reset()

    def set_smoothness(self, smoothness: float) -> None:
        self.smoothness = _clamp01(smoothness)
        pos_cutoff, pos_beta = _pose_position_params_from_smoothness(self.smoothness)
        rot_cutoff, rot_beta = _pose_rotation_params_from_smoothness(self.smoothness)
        self._position_filter.set_params(
            min_cutoff_hz=pos_cutoff,
            beta=pos_beta,
            d_cutoff_hz=self.derivative_cutoff_hz,
        )
        self._rotation_filter.set_params(
            min_cutoff_hz=rot_cutoff,
            beta=rot_beta,
            d_cutoff_hz=self.derivative_cutoff_hz,
        )
        self.reset()

    def _resolve_dt(self, timestamp: float | None) -> float:
        default_dt = 1.0 / self.default_fps
        if timestamp is None:
            self._last_timestamp = None
            return default_dt

        ts = float(timestamp)
        if self._last_timestamp is None:
            self._last_timestamp = ts
            return default_dt

        dt = ts - self._last_timestamp
        self._last_timestamp = ts
        if dt <= 0.0 or dt > self.reset_timeout_s:
            self.reset()
            self._last_timestamp = ts
            return default_dt
        return dt

    def smooth(
        self,
        pose_data: dict[str, object],
        pose_solution: PoseSolution | None,
        *,
        timestamp: float | None = None,
    ) -> tuple[dict[str, object], PoseSolution | None]:
        if not self.enabled or self.smoothness <= 0.0:
            return pose_data, pose_solution

        if not bool(pose_data.get("pose_valid", False)) or pose_solution is None:
            self.reset()
            return pose_data, pose_solution

        dt = self._resolve_dt(timestamp)

        C_w = np.asarray(pose_solution.C_w, dtype=np.float64).reshape(3)
        R_wc = np.asarray(pose_solution.R_wc, dtype=np.float64).reshape(3, 3)
        K = np.asarray(pose_solution.K, dtype=np.float64).reshape(3, 3)

        C_w_smooth = self._position_filter.filter(C_w, dt)
        q_wc = _quat_from_rotmat(R_wc)
        q_wc_smooth = self._rotation_filter.filter(q_wc, dt)
        R_wc_smooth = _rotmat_from_quat(q_wc_smooth)

        yaw, pitch, roll = _ypr_from_R_wc(R_wc_smooth)

        smoothed_pose = dict(pose_data)
        smoothed_pose["x"] = float(C_w_smooth[0])
        smoothed_pose["altitude"] = float(C_w_smooth[1])
        smoothed_pose["z"] = float(C_w_smooth[2])
        smoothed_pose["yaw"] = float(yaw)
        smoothed_pose["pitch"] = float(pitch)
        smoothed_pose["roll"] = float(roll)

        smoothed_solution = PoseSolution(
            C_w=C_w_smooth.astype(np.float64),
            R_wc=R_wc_smooth.astype(np.float64),
            K=K.astype(np.float64),
        )
        return smoothed_pose, smoothed_solution


@dataclass
class _WorldTrackState:
    filt: OneEuroVectorFilter
    last_timestamp: float


class WorldTrackSmoother:
    """Smooth per-track world-space X/Z after registration."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        smoothness: float = 0.5,
        derivative_cutoff_hz: float = 1.0,
        reset_timeout_s: float = 0.75,
        max_track_age_s: float = 1.5,
        default_fps: float = 30.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.smoothness = _clamp01(smoothness)
        self.derivative_cutoff_hz = max(1e-3, float(derivative_cutoff_hz))
        self.reset_timeout_s = max(1e-3, float(reset_timeout_s))
        self.max_track_age_s = max(1e-3, float(max_track_age_s))
        self.default_fps = max(1.0, float(default_fps))
        self._states: dict[int, _WorldTrackState] = {}
        self._min_cutoff_hz, self._beta = _world_params_from_smoothness(self.smoothness)

    def reset(self) -> None:
        self._states.clear()

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self.enabled != enabled:
            self.enabled = enabled
            self.reset()

    def set_smoothness(self, smoothness: float) -> None:
        self.smoothness = _clamp01(smoothness)
        self._min_cutoff_hz, self._beta = _world_params_from_smoothness(self.smoothness)
        self.reset()

    def _new_filter(self) -> OneEuroVectorFilter:
        return OneEuroVectorFilter(
            min_cutoff_hz=self._min_cutoff_hz,
            beta=self._beta,
            d_cutoff_hz=self.derivative_cutoff_hz,
        )

    def _prune(self, timestamp: float | None) -> None:
        if timestamp is None:
            return
        ts = float(timestamp)
        stale_keys = [
            track_id
            for track_id, state in self._states.items()
            if ts - float(state.last_timestamp) > self.max_track_age_s
        ]
        for track_id in stale_keys:
            self._states.pop(track_id, None)

    def update_inplace(self, detections, *, timestamp: float | None = None) -> None:
        if not self.enabled or self.smoothness <= 0.0:
            return

        default_dt = 1.0 / self.default_fps
        ts = None if timestamp is None else float(timestamp)

        for det in detections:
            if not bool(det.get("world_valid", False)):
                continue

            track_id = det.get("track_id")
            if track_id is None:
                continue

            try:
                track_key = int(track_id)
                xz = np.array(
                    [float(det.get("world_x", 0.0)), float(det.get("world_z", 0.0))],
                    dtype=np.float64,
                )
            except Exception:
                continue

            state = self._states.get(track_key)
            if state is None:
                filt = self._new_filter()
                xz_smooth = filt.filter(xz, default_dt)
                self._states[track_key] = _WorldTrackState(
                    filt=filt,
                    last_timestamp=ts if ts is not None else 0.0,
                )
            else:
                if ts is None or state.last_timestamp <= 0.0:
                    dt = default_dt
                else:
                    dt = ts - float(state.last_timestamp)
                    if dt <= 0.0 or dt > self.reset_timeout_s:
                        state.filt.reset()
                        dt = default_dt
                xz_smooth = state.filt.filter(xz, dt)
                state.last_timestamp = ts if ts is not None else state.last_timestamp

            det["world_x"] = float(xz_smooth[0])
            det["world_z"] = float(xz_smooth[1])

        self._prune(ts)
