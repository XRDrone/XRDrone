"""
adaptive_tuning.py

Bounded runtime adaptation for smoothing and flicker-mitigation policy.

This module does not change physical scene configuration, the UDP schema,
or the structural ArUco solver policy. Instead, it watches rolling runtime
metrics and nudges a small set of stabilizing controls within safe bounds:

  - motion smoothing level
  - ID flicker tau_on / tau_off hysteresis
  - ID flicker coast_frames hold duration

The controller is intentionally conservative:
  - bounded min/max ranges
  - rolling-window metrics
  - fixed update interval + cooldown
  - small step sizes only
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not isfinite(out):
        return float(default)
    return out


def _wrap_angle_deg(angle_deg: float) -> float:
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return float(wrapped)


def _bbox_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    try:
        ax1, ay1, ax2, ay2 = (float(v) for v in a)
        bx1, by1, bx2, by2 = (float(v) for v in b)
    except Exception:
        return 0.0

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return float(inter_area / denom)


@dataclass(frozen=True)
class AdaptiveTuningMetrics:
    pose_valid_ratio: float
    avg_markers_used: float
    pose_position_jitter_m: float
    pose_rotation_jitter_deg: float
    coast_ratio: float
    id_switch_rate: float
    avg_fps: float
    avg_drop_frames: float


class AdaptiveRuntimeTuner:
    """Rolling-window controller for bounded runtime adaptation."""

    _LOW_TRUST_POSE_VALID_RATIO = 0.35
    _LOW_TRUST_MARKERS_USED = 0.75
    _LOW_TRUST_DROP_FRAMES = 3.0

    _STABLE_POSE_VALID_RATIO = 0.85
    _STABLE_POSE_POS_JITTER_M = 0.05
    _STABLE_POSE_ROT_JITTER_DEG = 2.5
    _STABLE_COAST_RATIO = 0.12
    _STABLE_ID_SWITCH_RATE = 0.05
    _STABLE_DROP_FRAMES = 1.0

    _JITTER_POSE_POS_JITTER_M = 0.08
    _JITTER_POSE_ROT_JITTER_DEG = 4.0
    _JITTER_COAST_RATIO = 0.18
    _JITTER_ID_SWITCH_RATE = 0.08

    def __init__(
        self,
        *,
        enabled: bool,
        target_classes: Sequence[str],
        window_frames: int,
        update_interval_frames: int,
        cooldown_frames: int,
        iou_match_threshold: float,
        motion_smoothing_min: float,
        motion_smoothing_max: float,
        motion_smoothing_step: float,
        id_tau_on_min: float,
        id_tau_on_max: float,
        id_tau_off_min: float,
        id_tau_off_max: float,
        id_tau_step: float,
        id_coast_frames_min: int,
        id_coast_frames_max: int,
        id_coast_step: int,
        base_motion_smoothing: float,
        base_tau_on: float,
        base_tau_off: float,
        base_coast_frames: int,
    ) -> None:
        self.enabled = bool(enabled)
        self.target_classes = {str(c).lower() for c in target_classes}
        self.window_frames = max(10, int(window_frames))
        self.update_interval_frames = max(1, int(update_interval_frames))
        self.cooldown_frames = max(0, int(cooldown_frames))
        self.iou_match_threshold = _clamp(float(iou_match_threshold), 0.05, 0.95)

        self.motion_smoothing_min = _clamp(motion_smoothing_min, 0.0, 1.0)
        self.motion_smoothing_max = _clamp(
            max(motion_smoothing_max, self.motion_smoothing_min),
            self.motion_smoothing_min,
            1.0,
        )
        self.motion_smoothing_step = _clamp(motion_smoothing_step, 0.01, 0.25)

        self.id_tau_on_min = _clamp(id_tau_on_min, 0.0, 1.0)
        self.id_tau_on_max = _clamp(max(id_tau_on_max, self.id_tau_on_min), 0.0, 1.0)
        self.id_tau_off_min = _clamp(id_tau_off_min, 0.0, 1.0)
        self.id_tau_off_max = _clamp(max(id_tau_off_max, self.id_tau_off_min), 0.0, 1.0)
        self.id_tau_step = _clamp(id_tau_step, 0.005, 0.2)

        self.id_coast_frames_min = max(0, int(id_coast_frames_min))
        self.id_coast_frames_max = max(self.id_coast_frames_min, int(id_coast_frames_max))
        self.id_coast_step = max(1, int(id_coast_step))

        self.base_motion_smoothing = _clamp(
            base_motion_smoothing,
            self.motion_smoothing_min,
            self.motion_smoothing_max,
        )
        self.base_tau_on = _clamp(base_tau_on, self.id_tau_on_min, self.id_tau_on_max)
        self.base_tau_off = _clamp(base_tau_off, self.id_tau_off_min, self.id_tau_off_max)
        self.base_coast_frames = max(
            self.id_coast_frames_min,
            min(self.id_coast_frames_max, int(base_coast_frames)),
        )

        self.reset()

    def reset(self) -> None:
        self._frame_count = 0
        self._last_applied_frame = -(10**9)
        self._pose_valid_hist: deque[float] = deque(maxlen=self.window_frames)
        self._markers_used_hist: deque[float] = deque(maxlen=self.window_frames)
        self._pose_position_jitter_hist: deque[float] = deque(maxlen=self.window_frames)
        self._pose_rotation_jitter_hist: deque[float] = deque(maxlen=self.window_frames)
        self._coast_ratio_hist: deque[float] = deque(maxlen=self.window_frames)
        self._id_switch_rate_hist: deque[float] = deque(maxlen=self.window_frames)
        self._prev_valid_pose_vec: np.ndarray | None = None
        self._prev_track_entries: list[dict[str, Any]] = []

    def _is_target_class(self, det: dict[str, Any]) -> bool:
        cls_name = str(det.get("class") or det.get("class_name") or "").lower()
        return cls_name in self.target_classes

    def _record_pose_metrics(self, pose_data: dict[str, Any]) -> None:
        pose_valid = bool(pose_data.get("pose_valid", False))
        self._pose_valid_hist.append(1.0 if pose_valid else 0.0)
        self._markers_used_hist.append(max(0.0, _safe_float(pose_data.get("markers_used", 0), 0.0)))

        if not pose_valid:
            self._prev_valid_pose_vec = None
            return

        cur_vec = np.array(
            [
                _safe_float(pose_data.get("x", 0.0), 0.0),
                _safe_float(pose_data.get("altitude", 0.0), 0.0),
                _safe_float(pose_data.get("z", 0.0), 0.0),
                _safe_float(pose_data.get("yaw", 0.0), 0.0),
                _safe_float(pose_data.get("pitch", 0.0), 0.0),
                _safe_float(pose_data.get("roll", 0.0), 0.0),
            ],
            dtype=np.float64,
        )

        if self._prev_valid_pose_vec is not None:
            d_pos = cur_vec[:3] - self._prev_valid_pose_vec[:3]
            d_rot = np.array(
                [
                    _wrap_angle_deg(cur_vec[3] - self._prev_valid_pose_vec[3]),
                    _wrap_angle_deg(cur_vec[4] - self._prev_valid_pose_vec[4]),
                    _wrap_angle_deg(cur_vec[5] - self._prev_valid_pose_vec[5]),
                ],
                dtype=np.float64,
            )
            self._pose_position_jitter_hist.append(float(np.linalg.norm(d_pos)))
            self._pose_rotation_jitter_hist.append(float(np.linalg.norm(d_rot)))

        self._prev_valid_pose_vec = cur_vec

    def _extract_track_entries(self, detections: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for det in detections:
            if not self._is_target_class(det):
                continue
            bbox = det.get("bbox_xyxy")
            track_id = det.get("track_id")
            if bbox is None or track_id is None:
                continue
            try:
                entries.append(
                    {
                        "class": str(det.get("class") or det.get("class_name") or "").lower(),
                        "track_id": int(track_id),
                        "bbox_xyxy": [float(v) for v in bbox],
                    }
                )
            except Exception:
                continue
        return entries

    def _estimate_id_switch_rate(self, raw_detections: Sequence[dict[str, Any]]) -> float:
        cur_entries = self._extract_track_entries(raw_detections)
        if not cur_entries:
            self._prev_track_entries = []
            return 0.0

        switches = 0
        matched = 0
        used_prev: set[int] = set()

        for cur in cur_entries:
            best_idx = -1
            best_iou = 0.0
            for prev_idx, prev in enumerate(self._prev_track_entries):
                if prev_idx in used_prev:
                    continue
                if prev["class"] != cur["class"]:
                    continue
                iou = _bbox_iou_xyxy(cur["bbox_xyxy"], prev["bbox_xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = prev_idx

            if best_idx >= 0 and best_iou >= self.iou_match_threshold:
                used_prev.add(best_idx)
                matched += 1
                if int(self._prev_track_entries[best_idx]["track_id"]) != int(cur["track_id"]):
                    switches += 1

        self._prev_track_entries = cur_entries
        if matched <= 0:
            return 0.0
        return float(switches / matched)

    def _record_continuity_metrics(self, udp_ready_detections: Sequence[dict[str, Any]]) -> None:
        target_dets = [det for det in udp_ready_detections if self._is_target_class(det)]
        if not target_dets:
            self._coast_ratio_hist.append(0.0)
            return

        coasted = 0
        for det in target_dets:
            if str(det.get("continuity_state", "")).lower() == "coasted":
                coasted += 1
        self._coast_ratio_hist.append(float(coasted / len(target_dets)))

    def record_frame(
        self,
        *,
        pose_data: dict[str, Any],
        raw_detections: Sequence[dict[str, Any]],
        udp_ready_detections: Sequence[dict[str, Any]],
    ) -> None:
        if not self.enabled:
            return

        self._frame_count += 1
        self._record_pose_metrics(pose_data)
        self._record_continuity_metrics(udp_ready_detections)
        self._id_switch_rate_hist.append(self._estimate_id_switch_rate(raw_detections))

    def _has_enough_history(self) -> bool:
        if len(self._pose_valid_hist) < min(self.window_frames, self.update_interval_frames):
            return False
        return True

    def _current_metrics(self, *, avg_fps: float, avg_drop_frames: float) -> AdaptiveTuningMetrics:
        return AdaptiveTuningMetrics(
            pose_valid_ratio=_mean(self._pose_valid_hist),
            avg_markers_used=_mean(self._markers_used_hist),
            pose_position_jitter_m=_mean(self._pose_position_jitter_hist),
            pose_rotation_jitter_deg=_mean(self._pose_rotation_jitter_hist),
            coast_ratio=_mean(self._coast_ratio_hist),
            id_switch_rate=_mean(self._id_switch_rate_hist),
            avg_fps=max(0.0, float(avg_fps)),
            avg_drop_frames=max(0.0, float(avg_drop_frames)),
        )

    def _choose_mode(self, metrics: AdaptiveTuningMetrics) -> str:
        if (
            metrics.pose_valid_ratio < self._LOW_TRUST_POSE_VALID_RATIO
            or metrics.avg_markers_used < self._LOW_TRUST_MARKERS_USED
            or metrics.avg_drop_frames > self._LOW_TRUST_DROP_FRAMES
        ):
            return "low_trust"

        stable_pose = metrics.pose_valid_ratio >= self._STABLE_POSE_VALID_RATIO
        stable_jitter = metrics.pose_position_jitter_m <= self._STABLE_POSE_POS_JITTER_M
        stable_rot = metrics.pose_rotation_jitter_deg <= self._STABLE_POSE_ROT_JITTER_DEG
        stable_coast = metrics.coast_ratio <= self._STABLE_COAST_RATIO
        stable_ids = metrics.id_switch_rate <= self._STABLE_ID_SWITCH_RATE
        stable_drops = metrics.avg_drop_frames <= self._STABLE_DROP_FRAMES

        if (
            stable_pose
            and stable_jitter
            and stable_rot
            and stable_coast
            and stable_ids
            and stable_drops
        ):
            return "stable"

        if metrics.pose_valid_ratio >= self._LOW_TRUST_POSE_VALID_RATIO and (
            metrics.pose_position_jitter_m >= self._JITTER_POSE_POS_JITTER_M
            or metrics.pose_rotation_jitter_deg >= self._JITTER_POSE_ROT_JITTER_DEG
            or metrics.coast_ratio >= self._JITTER_COAST_RATIO
            or metrics.id_switch_rate >= self._JITTER_ID_SWITCH_RATE
        ):
            return "jittery_visible"

        return "recovering"

    def _step_toward(self, current: float, target: float, step: float) -> float:
        current = float(current)
        target = float(target)
        step = abs(float(step))
        if abs(target - current) <= step:
            return target
        if target > current:
            return current + step
        return current - step

    def _step_toward_int(self, current: int, target: int, step: int) -> int:
        current_i = int(current)
        target_i = int(target)
        step_i = max(1, int(step))
        if abs(target_i - current_i) <= step_i:
            return target_i
        if target_i > current_i:
            return current_i + step_i
        return current_i - step_i

    def propose_adjustment(
        self,
        *,
        current_motion_smoothing: float,
        current_tau_on: float,
        current_tau_off: float,
        current_coast_frames: int,
        avg_fps: float,
        avg_drop_frames: float,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if not self._has_enough_history():
            return None
        if self._frame_count % self.update_interval_frames != 0:
            return None
        if (self._frame_count - self._last_applied_frame) < self.cooldown_frames:
            return None

        metrics = self._current_metrics(avg_fps=avg_fps, avg_drop_frames=avg_drop_frames)
        mode = self._choose_mode(metrics)

        cur_smooth = _clamp(
            current_motion_smoothing,
            self.motion_smoothing_min,
            self.motion_smoothing_max,
        )
        cur_tau_on = _clamp(current_tau_on, self.id_tau_on_min, self.id_tau_on_max)
        cur_tau_off = _clamp(current_tau_off, self.id_tau_off_min, self.id_tau_off_max)
        cur_coast = max(
            self.id_coast_frames_min, min(self.id_coast_frames_max, int(current_coast_frames))
        )

        stable_target_smooth = max(self.motion_smoothing_min, self.base_motion_smoothing - 0.10)
        stable_target_coast = max(self.id_coast_frames_min, self.base_coast_frames - 1)
        jitter_target_smooth = min(self.motion_smoothing_max, self.base_motion_smoothing + 0.15)
        jitter_target_coast = min(self.id_coast_frames_max, self.base_coast_frames + 2)
        low_trust_target_smooth = min(self.motion_smoothing_max, self.base_motion_smoothing + 0.10)
        low_trust_target_coast = max(self.id_coast_frames_min, self.base_coast_frames - 1)

        if mode == "stable":
            new_smooth = self._step_toward(
                cur_smooth, stable_target_smooth, self.motion_smoothing_step
            )
            new_tau_on = self._step_toward(cur_tau_on, self.base_tau_on, self.id_tau_step)
            new_tau_off = self._step_toward(
                cur_tau_off,
                min(self.id_tau_off_max, self.base_tau_off + 0.02),
                self.id_tau_step,
            )
            new_coast = self._step_toward_int(cur_coast, stable_target_coast, self.id_coast_step)
        elif mode == "jittery_visible":
            new_smooth = self._step_toward(
                cur_smooth, jitter_target_smooth, self.motion_smoothing_step
            )
            new_tau_on = self._step_toward(cur_tau_on, self.base_tau_on, self.id_tau_step)
            new_tau_off = self._step_toward(
                cur_tau_off,
                max(self.id_tau_off_min, self.base_tau_off - 0.06),
                self.id_tau_step,
            )
            new_coast = self._step_toward_int(cur_coast, jitter_target_coast, self.id_coast_step)
        elif mode == "low_trust":
            new_smooth = self._step_toward(
                cur_smooth, low_trust_target_smooth, self.motion_smoothing_step
            )
            new_tau_on = self._step_toward(
                cur_tau_on,
                min(self.id_tau_on_max, self.base_tau_on + 0.04),
                self.id_tau_step,
            )
            new_tau_off = self._step_toward(
                cur_tau_off,
                min(self.id_tau_off_max, self.base_tau_off + 0.04),
                self.id_tau_step,
            )
            new_coast = self._step_toward_int(cur_coast, low_trust_target_coast, self.id_coast_step)
        else:
            new_smooth = self._step_toward(
                cur_smooth, self.base_motion_smoothing, self.motion_smoothing_step
            )
            new_tau_on = self._step_toward(cur_tau_on, self.base_tau_on, self.id_tau_step)
            new_tau_off = self._step_toward(cur_tau_off, self.base_tau_off, self.id_tau_step)
            new_coast = self._step_toward_int(cur_coast, self.base_coast_frames, self.id_coast_step)

        new_smooth = _clamp(new_smooth, self.motion_smoothing_min, self.motion_smoothing_max)
        new_tau_on = _clamp(new_tau_on, self.id_tau_on_min, self.id_tau_on_max)
        new_tau_off = _clamp(min(new_tau_off, new_tau_on), self.id_tau_off_min, self.id_tau_off_max)
        new_coast = max(self.id_coast_frames_min, min(self.id_coast_frames_max, int(new_coast)))

        changed = (
            abs(new_smooth - cur_smooth) > 1e-9
            or abs(new_tau_on - cur_tau_on) > 1e-9
            or abs(new_tau_off - cur_tau_off) > 1e-9
            or int(new_coast) != int(cur_coast)
        )

        if changed:
            self._last_applied_frame = self._frame_count

        return {
            "mode": mode,
            "changed": changed,
            "motion_smoothing": float(new_smooth),
            "tau_on": float(new_tau_on),
            "tau_off": float(new_tau_off),
            "coast_frames": int(new_coast),
            "metrics": metrics,
        }
