"""
motion_smoothing.py

Rust-backed temporal smoothing for pose registration and world tracking.
The Python layer preserves the existing public API and reconstructs
PoseSolution objects from the native outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pose_estimator import PoseSolution

try:
    from xrdrone_native import (
        OneEuroQuaternionFilter,
        OneEuroVectorFilter,
        PoseMotionSmootherCore,
        WorldTrackSmootherCore,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "xrdrone_native is required for motion_smoothing.py. "
        "Build it first with ./build_native.sh or maturin develop --release."
    ) from exc


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class PoseMotionSmoother:
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
        self._core = PoseMotionSmootherCore(
            enabled=self.enabled,
            smoothness=self.smoothness,
            derivative_cutoff_hz=self.derivative_cutoff_hz,
            reset_timeout_s=self.reset_timeout_s,
            default_fps=self.default_fps,
        )

    def reset(self) -> None:
        self._core.reset()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._core.set_enabled(self.enabled)

    def set_smoothness(self, smoothness: float) -> None:
        self.smoothness = _clamp01(smoothness)
        self._core.set_smoothness(self.smoothness)

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

        smoothed_pose, c_w, r_wc, k = self._core.smooth_pose(
            pose_data,
            [float(v) for v in np.asarray(pose_solution.C_w, dtype=np.float64).reshape(3)],
            [float(v) for v in np.asarray(pose_solution.R_wc, dtype=np.float64).reshape(9)],
            [float(v) for v in np.asarray(pose_solution.K, dtype=np.float64).reshape(9)],
            timestamp=timestamp,
        )

        smoothed_solution = PoseSolution(
            C_w=np.asarray(c_w, dtype=np.float64).reshape(3),
            R_wc=np.asarray(r_wc, dtype=np.float64).reshape(3, 3),
            K=np.asarray(k, dtype=np.float64).reshape(3, 3),
        )
        return dict(smoothed_pose), smoothed_solution


class WorldTrackSmoother:
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
        self._core = WorldTrackSmootherCore(
            enabled=self.enabled,
            smoothness=self.smoothness,
            derivative_cutoff_hz=self.derivative_cutoff_hz,
            reset_timeout_s=self.reset_timeout_s,
            max_track_age_s=self.max_track_age_s,
            default_fps=self.default_fps,
        )

    def reset(self) -> None:
        self._core.reset()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._core.set_enabled(self.enabled)

    def set_smoothness(self, smoothness: float) -> None:
        self.smoothness = _clamp01(smoothness)
        self._core.set_smoothness(self.smoothness)

    def update_inplace(self, detections, *, timestamp: float | None = None) -> None:
        updated = self._core.update_detections(detections, timestamp=timestamp)
        detections[:] = list(updated)


__all__ = [
    "OneEuroQuaternionFilter",
    "OneEuroVectorFilter",
    "PoseMotionSmoother",
    "WorldTrackSmoother",
    "PoseSolution",
]
