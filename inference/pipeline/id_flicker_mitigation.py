"""
id_flicker_mitigation.py

Sender-side mitigation for intermittent object-ID flicker in UDP JSON streams.

Implements a practical low-latency continuity layer:
  - confidence hysteresis (tau_on / tau_off)
  - per-track confidence EMA
  - bounded hold-and-forward ("coasting") keyed by track_id
  - stale-state pruning

The external UDP schema stays unchanged. This module only annotates internal
detection dicts with optional helper fields consumed upstream by the formatter:
  - udp_confidence: float
  - force_udp_emit: bool
  - continuity_state: "observed" | "coasted"
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clone_detection(det: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in det.items():
        if isinstance(value, list):
            out[key] = list(value)
        elif isinstance(value, dict):
            out[key] = dict(value)
        else:
            out[key] = value
    return out


@dataclass
class _TrackContinuityState:
    cls_name: str
    conf_ema: float = 0.0
    emitted: bool = False
    miss_count: int = 0
    last_seen_step: int = 0
    last_gate_conf: float = 0.0
    last_raw_conf: float = 0.0
    last_observed_det: dict[str, Any] | None = None
    last_stable_det: dict[str, Any] | None = None


class RobustIDFlickerMitigator:
    """
    Hysteresis + coasting continuity filter for tracked detections.

    Typical usage:
        mitigator = RobustIDFlickerMitigator(...)
        udp_ready_detections = mitigator.apply(tracked_detections)

    Expected input:
      - detection dicts with at least:
          class, confidence, bbox_xyxy, track_id
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        apply_classes: Sequence[str] | None = None,
        tau_on: float = 0.80,
        tau_off: float = 0.55,
        coast_frames: int = 6,
        drop_frames: int = 45,
        ema_alpha: float = 0.45,
        use_conf_ema: bool = True,
        require_track_id: bool = True,
        coast_conf_decay: float = 0.985,
    ) -> None:
        self.enabled = bool(enabled)
        self.apply_classes: set[str] | None = (
            {str(c).lower() for c in apply_classes} if apply_classes else None
        )
        self.tau_on = _clamp01(tau_on)
        self.tau_off = _clamp01(min(tau_off, self.tau_on))
        self.coast_frames = max(0, int(coast_frames))
        self.drop_frames = max(int(drop_frames), self.coast_frames)
        self.ema_alpha = _clamp01(ema_alpha)
        self.use_conf_ema = bool(use_conf_ema)
        self.require_track_id = bool(require_track_id)
        self.coast_conf_decay = _clamp01(coast_conf_decay)

        self._states: dict[int, _TrackContinuityState] = {}
        self._step = 0

    def reset(self) -> None:
        self._states.clear()
        self._step = 0

    def _is_target_class(self, det: dict[str, Any]) -> bool:
        cls_name = str(det.get("class") or det.get("class_name") or "").lower()
        if self.apply_classes is None:
            return True
        return cls_name in self.apply_classes

    def _track_key(self, det: dict[str, Any]) -> int | None:
        track_id = det.get("track_id", None)
        if track_id is None:
            return None
        try:
            return int(track_id)
        except Exception:
            return None

    def _observe(self, det: dict[str, Any], track_key: int) -> _TrackContinuityState:
        cls_name = str(det.get("class") or det.get("class_name") or "obj").lower()
        raw_conf = _safe_float(det.get("confidence", 0.0), 0.0)

        state = self._states.get(track_key)
        if state is None:
            state = _TrackContinuityState(cls_name=cls_name)
            state.conf_ema = raw_conf
            self._states[track_key] = state
        else:
            state.cls_name = cls_name
            state.conf_ema = self.ema_alpha * raw_conf + (1.0 - self.ema_alpha) * state.conf_ema

        state.last_gate_conf = state.conf_ema if self.use_conf_ema else raw_conf
        state.last_raw_conf = raw_conf
        state.last_seen_step = self._step
        state.last_observed_det = _clone_detection(det)
        return state

    def _make_output_det(
        self,
        det: dict[str, Any],
        *,
        udp_confidence: float,
        continuity_state: str,
    ) -> dict[str, Any]:
        out = _clone_detection(det)
        out["udp_confidence"] = float(_clamp01(udp_confidence))
        out["force_udp_emit"] = True
        out["continuity_state"] = str(continuity_state)
        return out

    def _make_coasted_output(self, state: _TrackContinuityState) -> dict[str, Any] | None:
        if state.last_stable_det is None:
            return None

        decay_power = max(0, int(state.miss_count) - 1)
        coast_conf = state.last_gate_conf * (self.coast_conf_decay**decay_power)

        out = _clone_detection(state.last_stable_det)
        out["udp_confidence"] = float(_clamp01(coast_conf))
        out["force_udp_emit"] = True
        out["continuity_state"] = "coasted"
        return out

    def apply(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Return detections ready for UDP emission after continuity mitigation.

        Non-target classes pass through unchanged.
        Target classes are emitted only through the hysteresis/coasting policy.
        """
        if not self.enabled:
            return list(detections)

        self._step += 1

        passthrough: list[dict[str, Any]] = []
        outputs: list[dict[str, Any]] = []
        seen_track_keys: set[int] = set()
        emitted_track_keys: set[int] = set()

        for det in detections:
            if not self._is_target_class(det):
                passthrough.append(det)
                continue

            track_key = self._track_key(det)
            if track_key is None:
                if not self.require_track_id:
                    passthrough.append(det)
                continue

            seen_track_keys.add(track_key)
            state = self._observe(det, track_key)
            gate_conf = float(state.last_gate_conf)

            if not state.emitted:
                if gate_conf >= self.tau_on:
                    state.emitted = True
                    state.miss_count = 0
                    state.last_stable_det = _clone_detection(det)
                    outputs.append(
                        self._make_output_det(
                            det,
                            udp_confidence=gate_conf,
                            continuity_state="observed",
                        )
                    )
                    emitted_track_keys.add(track_key)
                continue

            if gate_conf >= self.tau_off:
                state.miss_count = 0
                state.last_stable_det = _clone_detection(det)
                outputs.append(
                    self._make_output_det(
                        det,
                        udp_confidence=gate_conf,
                        continuity_state="observed",
                    )
                )
                emitted_track_keys.add(track_key)
                continue

            state.miss_count += 1
            if state.miss_count <= self.coast_frames:
                coasted = self._make_coasted_output(state)
                if coasted is not None:
                    outputs.append(coasted)
                    emitted_track_keys.add(track_key)
            else:
                state.emitted = False

        stale_keys: list[int] = []

        for track_key, state in list(self._states.items()):
            if track_key in seen_track_keys:
                continue

            gap_steps = self._step - int(state.last_seen_step)

            if state.emitted:
                state.miss_count += 1
                if state.miss_count <= self.coast_frames and track_key not in emitted_track_keys:
                    coasted = self._make_coasted_output(state)
                    if coasted is not None:
                        outputs.append(coasted)
                        emitted_track_keys.add(track_key)
                else:
                    state.emitted = False

            if gap_steps > self.drop_frames:
                stale_keys.append(track_key)

        for track_key in stale_keys:
            self._states.pop(track_key, None)

        return passthrough + outputs
