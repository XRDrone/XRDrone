"""
adaptive_tuning.py

Rust-backed bounded runtime adaptation controller.
Build xrdrone_native first, then import this module normally.
"""

from __future__ import annotations

try:
    from xrdrone_native import AdaptiveRuntimeTuner, AdaptiveTuningMetrics
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "xrdrone_native is required for adaptive_tuning.py. "
        "Build it first with ./build_native.sh or maturin develop --release."
    ) from exc

__all__ = ["AdaptiveRuntimeTuner", "AdaptiveTuningMetrics"]
