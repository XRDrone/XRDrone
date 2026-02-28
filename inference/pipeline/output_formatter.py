# File: output_formatter.py
"""
output_formatter.py

Formats merged detection results into the UDP schema expected by Unity.

Converts pixel-space detections into normalized bounding box coordinates
and maps class names to integer IDs.

Each UDP packet contains:
  - frame_id: int
  - timestamp: float
  - width, height: frame dimensions
  - detections[]:
      - id: persistent track ID
      - cls: integer class ID (Unity mapping)
      - conf: detection confidence
      - cx, cy: normalized center coordinates
      - w, h: normalized width and height
      - foot_x, foot_y: normalized bottom-center point of the bbox ("foot")
      - world_valid: bool (True if world_* are populated)
      - world_x, world_y, world_z: world-space coords (meters) when available

Handles:
  - confidence filtering
  - allowed class filtering
  - bbox normalization and clamping
  - fallback IDs when tracking is unavailable

Notes
-----
World registration fields are pass-through:
  - If merge-stage / pipeline code provides det["world_valid"] and world_*, they will be included.
  - Otherwise they default to false / 0.0.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _xyxy_to_xywhn(
    x1: float, y1: float, x2: float, y2: float, width: float, height: float
) -> Tuple[float, float, float, float]:
    # Convert pixel xyxy -> normalized cx,cy,w,h in [0,1]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    if width <= 0 or height <= 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        _clamp01(cx / width),
        _clamp01(cy / height),
        _clamp01(w / width),
        _clamp01(h / height),
    )


def to_unity_udp_packet(
    merged_detections: Sequence[Dict[str, Any]],
    *,
    frame_id: int,
    timestamp: float,
    width: int,
    height: int,
    class_map: Optional[Mapping[str, int]] = None,
    allowed_classes: Optional[Sequence[str]] = None,
    min_conf: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convert merger.merge_detections output into the UDP schema used by Unity:
      {
        "frame_id": int,
        "timestamp": float,
        "width": int,
        "height": int,
        "detections": [
          {
            "id","cls","conf","cx","cy","w","h",
            "foot_x","foot_y",
            "world_valid","world_x","world_y","world_z"
          }, ...
        ]
      }
    """
    class_map = class_map or {}
    allow = set(c.lower() for c in allowed_classes) if allowed_classes else None
    min_conf_f = float(min_conf) if min_conf is not None else None

    dets: List[Dict[str, Any]] = []
    for i, det in enumerate(merged_detections):
        cls_name = str(det.get("class") or det.get("class_name") or "").lower()
        if allow is not None and cls_name not in allow:
            continue

        conf = float(det.get("confidence", 0.0))
        if min_conf_f is not None and conf < min_conf_f:
            continue

        bbox = det.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = (float(b) for b in bbox)
        cx, cy, w, h = _xyxy_to_xywhn(x1, y1, x2, y2, float(width), float(height))

        # Default foot point is bottom-center of bbox.
        default_foot_x = float(cx)
        default_foot_y = float(_clamp01(cy + h / 2.0))

        raw_id = det.get("track_id", i)
        try:
            det_id = int(raw_id)
        except Exception:
            det_id = int(i)

        world_valid = bool(det.get("world_valid", False))
        try:
            world_x = float(det.get("world_x", 0.0))
            world_y = float(det.get("world_y", 0.0))
            world_z = float(det.get("world_z", 0.0))
        except Exception:
            world_x, world_y, world_z = 0.0, 0.0, 0.0
            world_valid = False

        dets.append(
            {
                "id": det_id,
                "cls": int(class_map.get(cls_name, -1)),
                "conf": float(conf),
                "cx": float(cx),
                "cy": float(cy),
                "w": float(w),
                "h": float(h),
                "foot_x": float(det.get("foot_x", default_foot_x)),
                "foot_y": float(det.get("foot_y", default_foot_y)),
                "world_valid": bool(world_valid),
                "world_x": float(world_x),
                "world_y": float(world_y),
                "world_z": float(world_z),
            }
        )

    return {
        "frame_id": int(frame_id),
        "timestamp": float(timestamp),
        "width": int(width),
        "height": int(height),
        "detections": dets,
    }