"""
id_flicker_mitigation.py

Rust-backed continuity filtering for tracked detections.
This wrapper preserves the original Python-visible attributes used by the
runtime while delegating the hot-path implementation to xrdrone_native.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    from xrdrone_native import RobustIDFlickerMitigator as _NativeRobustIDFlickerMitigator
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "xrdrone_native is required for id_flicker_mitigation.py. "
        "Build it first with ./build_native.sh or maturin develop --release."
    ) from exc


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class RobustIDFlickerMitigator:
    """Python compatibility wrapper over the native continuity filter."""

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
        self.apply_classes = (
            {str(c).lower() for c in apply_classes} if apply_classes is not None else None
        )
        self.tau_on = _clamp01(tau_on)
        self.tau_off = _clamp01(min(float(tau_off), self.tau_on))
        self.coast_frames = max(0, int(coast_frames))
        self.drop_frames = max(int(drop_frames), self.coast_frames)
        self.ema_alpha = _clamp01(ema_alpha)
        self.use_conf_ema = bool(use_conf_ema)
        self.require_track_id = bool(require_track_id)
        self.coast_conf_decay = _clamp01(coast_conf_decay)

        native_apply_classes = None
        if self.apply_classes is not None:
            native_apply_classes = sorted(self.apply_classes)

        self._native = _NativeRobustIDFlickerMitigator(
            enabled=self.enabled,
            apply_classes=native_apply_classes,
            tau_on=self.tau_on,
            tau_off=self.tau_off,
            coast_frames=self.coast_frames,
            drop_frames=self.drop_frames,
            ema_alpha=self.ema_alpha,
            use_conf_ema=self.use_conf_ema,
            require_track_id=self.require_track_id,
            coast_conf_decay=self.coast_conf_decay,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native, name)

    def set_runtime_policy(
        self,
        *,
        tau_on: float | None = None,
        tau_off: float | None = None,
        coast_frames: int | None = None,
        coast_conf_decay: float | None = None,
    ) -> None:
        if tau_on is not None:
            self.tau_on = _clamp01(tau_on)
        if tau_off is not None:
            self.tau_off = _clamp01(min(float(tau_off), self.tau_on))
        if coast_frames is not None:
            self.coast_frames = max(0, int(coast_frames))
            self.drop_frames = max(int(self.drop_frames), self.coast_frames)
        if coast_conf_decay is not None:
            self.coast_conf_decay = _clamp01(coast_conf_decay)

        kwargs: dict[str, Any] = {}
        if tau_on is not None:
            kwargs["tau_on"] = self.tau_on
        if tau_off is not None:
            kwargs["tau_off"] = self.tau_off
        if coast_frames is not None:
            kwargs["coast_frames"] = self.coast_frames
        if coast_conf_decay is not None:
            kwargs["coast_conf_decay"] = self.coast_conf_decay
        self._native.set_runtime_policy(**kwargs)

    def reset(self) -> None:
        self._native.reset()

    def apply(self, detections):
        return self._native.apply(detections)


__all__ = ["RobustIDFlickerMitigator"]
