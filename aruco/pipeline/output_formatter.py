"""
output_formatter.py

Python fallback UDP packet formatting for the XRDrone pipeline.
The Rust extension is used when available, but it is not required for the
pure ArUco video runner.
"""

from __future__ import annotations

from typing import Any

try:  # optional acceleration only
    from xrdrone_native import to_unity_udp_packet as _native_to_unity_udp_packet
except Exception:  # pragma: no cover
    _native_to_unity_udp_packet = None


def to_unity_udp_packet(
    detections: list[dict[str, Any]],
    *,
    frame_id: int,
    timestamp: float,
    width: int,
    height: int,
    class_map: dict[str, int],
    allowed_classes,
    min_conf: float,
) -> dict[str, Any]:
    if _native_to_unity_udp_packet is not None:
        return _native_to_unity_udp_packet(
            detections,
            frame_id=frame_id,
            timestamp=timestamp,
            width=width,
            height=height,
            class_map=class_map,
            allowed_classes=allowed_classes,
            min_conf=min_conf,
        )

    allowed = {str(c).lower() for c in allowed_classes}
    out_dets = []
    for det in detections:
        cls = str(det.get("class", "")).lower()
        conf = float(det.get("confidence", 0.0) or 0.0)
        if cls not in allowed or conf < float(min_conf):
            continue
        item = dict(det)
        item["class_id"] = int(class_map.get(cls, -1))
        out_dets.append(item)

    return {
        "frame_id": int(frame_id),
        "timestamp": float(timestamp),
        "width": int(width),
        "height": int(height),
        "detections": out_dets,
    }


__all__ = ["to_unity_udp_packet"]
