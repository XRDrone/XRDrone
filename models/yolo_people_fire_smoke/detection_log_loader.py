# output_formatter.py
"""
Takes merged detections from merger.py and writes them to JSON
with a local timestamp string that matches DJI's CUSTOM.updateTime [local],
e.g. "7:05:08.97 PM".
"""

from __future__ import annotations
from typing import Any, Dict, List
from datetime import datetime
import json


def format_timestamp_local(ts: float) -> str:
    """
    Convert a UNIX timestamp (seconds since epoch) to a local time string
    like "7:05:08.97 PM" (centisecond precision, 12-hour clock).
    """
    dt = datetime.fromtimestamp(ts)   # local time
    # 12-hour time without leading zero in the hour
    base = dt.strftime("%I:%M:%S")   # e.g. "07:05:08"
    base = base.lstrip("0")          # -> "7:05:08"

    # centiseconds (0.01s) similar to log format
    centiseconds = int((ts * 100) % 100)
    am_pm = dt.strftime("%p")

    return f"{base}.{centiseconds:02d} {am_pm}"


def prepare_detection_packet(det: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take a raw detection from merger.merge_detections and convert it into
    a JSON-safe dict with both epoch + local formatted timestamp.
    """
    ts = float(det.get("timestamp", 0.0))

    packet: Dict[str, Any] = {
        "timestamp_epoch": ts,
        "timestamp_local": format_timestamp_local(ts),
        "class": det.get("class"),
        "confidence": float(det.get("confidence", 0.0)),
        "bbox_xyxy": det.get("bbox_xyxy"),
        "source": det.get("source"),
    }

    # Avoid dumping huge numpy masks; just record whether one exists.
    mask = det.get("mask", None)
    packet["has_mask"] = mask is not None

    return packet


def save_detections_json(
    detections: List[Dict[str, Any]],
    output_path: str = "detections_log.json",
) -> None:
    """
    Save a list of merged detections into a JSON file.
    Each detection becomes one JSON object with:
      - timestamp_epoch: float
      - timestamp_local: "7:05:08.97 PM"
      - class, confidence, bbox_xyxy, source, has_mask
    """
    packets = [prepare_detection_packet(d) for d in detections]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(packets, f, indent=2)

    print(f"[output_formatter] Wrote {len(packets)} detections to {output_path}")
