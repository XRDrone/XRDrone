"""
output_formatter.py

Rust-backed UDP packet formatting.
Build xrdrone_native first, then import this module normally.
"""

from __future__ import annotations

try:
    from xrdrone_native import to_unity_udp_packet
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "xrdrone_native is required for output_formatter.py. "
        "Build it first with ./build_native.sh or maturin develop --release."
    ) from exc

__all__ = ["to_unity_udp_packet"]
