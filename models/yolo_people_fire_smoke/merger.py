# merger.py
"""
merger.py

Merges per-frame YOLO detections from separate people and fire/smoke
models into a single, timestamped list of detection records.

Each merged detection is a dict with:
  - timestamp: float (seconds since epoch)
  - class: str ("person", "fire", "smoke", or other model label)
  - confidence: float in [0, 1]
  - bbox_xyxy: [x1, y1, x2, y2] in image coordinates
  - mask: optional numpy array for people masks when segmentation is enabled
  - source: str ("people" or "fire") indicating which model produced it

Also provides:
  - count_by_class(): small helper to aggregate detections by class label
    for HUD display and basic analytics.
"""


from __future__ import annotations
from typing import Any, Dict, List, Optional
import time


def merge_detections(
    people_results,
    fire_results,
    *,
    people_model,
    fire_model,
    seg_on: bool = False,
    timestamp: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Merge detections from people_results and fire_results into a unified list.
    - people_results, fire_results: Ultralytics Results lists from .predict(...)
    - people_model, fire_model: YOLO model objects (for .names lookup)
    - seg_on: if True, attach people masks when available
    - timestamp: if None, time.time() is used

    Returns: list of detection dicts.
    """
    ts = time.time() if timestamp is None else float(timestamp)
    dets: List[Dict[str, Any]] = []

    # ---- People detections ----
    if people_results:
        for r in people_results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0]) if b.conf is not None else 0.0
                xyxy = b.xyxy[0].tolist()
                label = people_model.names[cls_id].lower()

                dets.append(
                    {
                        "timestamp": ts,
                        "class": label,
                        "confidence": conf,
                        "bbox_xyxy": xyxy,
                        "mask": None,
                        "source": "people",
                    }
                )

        # Attach masks in the same order as boxes (Ultralytics aligns them)
        if seg_on and people_results[0].masks is not None:
            masks = people_results[0].masks.data.detach().cpu().numpy()
            pi = 0
            for d in dets:
                if d["source"] == "people":
                    if pi < len(masks):
                        d["mask"] = masks[pi]
                    pi += 1

    # ---- Fire / Smoke detections ----
    if fire_results:
        for r in fire_results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0]) if b.conf is not None else 0.0
                xyxy = b.xyxy[0].tolist()
                label = fire_model.names[cls_id].lower()

                dets.append(
                    {
                        "timestamp": ts,
                        "class": label,
                        "confidence": conf,
                        "bbox_xyxy": xyxy,
                        "mask": None,  # fire model is detect-only for now
                        "source": "fire",
                    }
                )

    return dets


def count_by_class(detections: List[Dict[str, Any]]) -> Dict[str, int]:
    """Utility: count detections by class label."""
    counts: Dict[str, int] = {}
    for d in detections:
        c = d["class"]
        counts[c] = counts.get(c, 0) + 1
    return counts