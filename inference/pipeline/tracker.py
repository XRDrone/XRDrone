"""
tracker.py

Persistent, lightweight multi-object tracking for XRDrone.

Assigns stable track IDs to detections across frames using:
  - OpenCV Kalman filtering (constant velocity motion model)
  - IoU-based data association

Works directly with merged detection dicts from merger.py.

Capabilities:
  - Maintains track continuity through short occlusions
  - Optional per-class tracking isolation
  - Configurable IoU thresholds and track lifetimes
  - SORT-style matching and track spawning

Limitations:
  - No appearance-based re-identification
  - IDs may switch for long occlusions or similar objects
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _xyxy_to_cxcywh(bbox_xyxy: Sequence[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx, cy, w, h


def _cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> List[float]:
    w = max(0.0, float(w))
    h = max(0.0, float(h))
    x1 = float(cx) - w / 2.0
    y1 = float(cy) - h / 2.0
    x2 = float(cx) + w / 2.0
    y2 = float(cy) + h / 2.0
    return [x1, y1, x2, y2]


def _iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    aw = max(0.0, ax2 - ax1)
    ah = max(0.0, ay2 - ay1)
    bw = max(0.0, bx2 - bx1)
    bh = max(0.0, by2 - by1)
    area_a = aw * ah
    area_b = bw * bh

    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


@dataclass
class _Track:
    track_id: int
    cls_name: str
    bbox_xyxy: List[float]
    w: float
    h: float
    conf: float
    hits: int = 1
    age: int = 1
    lost: int = 0
    kf: Optional[cv2.KalmanFilter] = None


class OpenCVKalmanIOUTracker:
    """SORT-like tracker using OpenCV KalmanFilter + IoU assignment."""

    def __init__(
        self,
        *,
        min_iou: float = 0.30,
        max_age_frames: int = 90,
        per_class: bool = True,
        dt: float = 1.0,
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
    ) -> None:
        self.min_iou = float(min_iou)
        self.max_age = int(max_age_frames)
        self.per_class = bool(per_class)
        self.dt = float(dt)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)

        self._next_id = 1
        self._tracks: List[_Track] = []

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()

    def _new_kf(self, cx: float, cy: float) -> cv2.KalmanFilter:
        # State: [cx, cy, vx, vy]
        # Meas:  [cx, cy]
        kf = cv2.KalmanFilter(4, 2)
        dt = self.dt

        kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

        kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.process_noise
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.measurement_noise
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        return kf

    def _predict_track_bbox(self, tr: _Track) -> List[float]:
        if tr.kf is None:
            return tr.bbox_xyxy
        pred = tr.kf.predict()
        cx = float(pred[0, 0])
        cy = float(pred[1, 0])
        return _cxcywh_to_xyxy(cx, cy, tr.w, tr.h)

    def _correct_track(self, tr: _Track, cx: float, cy: float) -> None:
        if tr.kf is None:
            tr.kf = self._new_kf(cx, cy)
            return
        meas = np.array([[cx], [cy]], dtype=np.float32)
        tr.kf.correct(meas)

    def _spawn_track(self, det: Dict[str, Any]) -> _Track:
        bbox = det.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            bbox = [0.0, 0.0, 0.0, 0.0]
        cx, cy, w, h = _xyxy_to_cxcywh(bbox)
        tr = _Track(
            track_id=int(self._next_id),
            cls_name=str(det.get("class") or "obj").lower(),
            bbox_xyxy=list(map(float, bbox)),
            w=float(w),
            h=float(h),
            conf=float(det.get("confidence", 0.0)),
            hits=1,
            age=1,
            lost=0,
            kf=self._new_kf(cx, cy),
        )
        self._next_id += 1
        return tr

    def _greedy_match(
        self,
        track_bboxes: List[List[float]],
        det_bboxes: List[List[float]],
        iou_thresh: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Return matches + unmatched indices using greedy max-IoU."""
        if not track_bboxes or not det_bboxes:
            return [], list(range(len(track_bboxes))), list(range(len(det_bboxes)))

        iou_mat = np.zeros((len(track_bboxes), len(det_bboxes)), dtype=np.float32)
        for i, tb in enumerate(track_bboxes):
            for j, db in enumerate(det_bboxes):
                iou_mat[i, j] = _iou_xyxy(tb, db)

        matches: List[Tuple[int, int]] = []
        used_tracks = set()
        used_dets = set()

        while True:
            # find best remaining
            best_val = -1.0
            best_i = -1
            best_j = -1
            for i in range(iou_mat.shape[0]):
                if i in used_tracks:
                    continue
                row = iou_mat[i]
                for j in range(iou_mat.shape[1]):
                    if j in used_dets:
                        continue
                    val = float(row[j])
                    if val > best_val:
                        best_val = val
                        best_i = i
                        best_j = j

            if best_i < 0 or best_j < 0 or best_val < float(iou_thresh):
                break

            matches.append((best_i, best_j))
            used_tracks.add(best_i)
            used_dets.add(best_j)

        unmatched_tracks = [i for i in range(len(track_bboxes)) if i not in used_tracks]
        unmatched_dets = [j for j in range(len(det_bboxes)) if j not in used_dets]
        return matches, unmatched_tracks, unmatched_dets

    def update(self, detections: List[Dict[str, Any]]) -> None:
        """Assign/maintain det['track_id'] for this frame (in-place)."""
        # 1) predict all track positions
        for tr in self._tracks:
            tr.age += 1
            tr.bbox_xyxy = self._predict_track_bbox(tr)

        # 2) group detections by class (optional)
        det_indices = list(range(len(detections)))
        groups: Dict[str, List[int]] = {}
        if self.per_class:
            for di in det_indices:
                cls_name = str(detections[di].get("class") or "obj").lower()
                groups.setdefault(cls_name, []).append(di)
        else:
            groups["__all__"] = det_indices

        # 3) mark detections as unmatched initially
        for d in detections:
            d.pop("track_id", None)

        # 4) match per group
        matched_track_ids: set[int] = set()

        for g_name, det_ids in groups.items():
            # eligible tracks
            if self.per_class and g_name != "__all__":
                track_ids = [i for i, tr in enumerate(self._tracks) if tr.cls_name == g_name]
            else:
                track_ids = list(range(len(self._tracks)))

            if not track_ids and det_ids:
                # spawn all as new tracks
                for di in det_ids:
                    tr = self._spawn_track(detections[di])
                    self._tracks.append(tr)
                    detections[di]["track_id"] = tr.track_id
                    matched_track_ids.add(len(self._tracks) - 1)
                continue

            if not det_ids:
                continue

            track_bboxes = [self._tracks[ti].bbox_xyxy for ti in track_ids]
            det_bboxes: List[List[float]] = []
            for di in det_ids:
                bb = detections[di].get("bbox_xyxy")
                if not bb or len(bb) != 4:
                    bb = [0.0, 0.0, 0.0, 0.0]
                det_bboxes.append(list(map(float, bb)))

            matches, _, un_det = self._greedy_match(track_bboxes, det_bboxes, self.min_iou)

            # apply matches
            for local_ti, local_di in matches:
                ti = track_ids[local_ti]
                di = det_ids[local_di]
                tr = self._tracks[ti]

                bbox = det_bboxes[local_di]
                cx, cy, w, h = _xyxy_to_cxcywh(bbox)

                self._correct_track(tr, cx, cy)
                tr.bbox_xyxy = list(map(float, bbox))
                tr.w = float(w)
                tr.h = float(h)
                tr.conf = float(detections[di].get("confidence", tr.conf))
                tr.hits += 1
                tr.lost = 0

                detections[di]["track_id"] = int(tr.track_id)
                matched_track_ids.add(ti)

            # spawn tracks for unmatched detections
            for local_di in un_det:
                di = det_ids[local_di]
                tr = self._spawn_track(detections[di])
                self._tracks.append(tr)
                detections[di]["track_id"] = int(tr.track_id)
                matched_track_ids.add(len(self._tracks) - 1)

        # 5) age unmatched tracks
        survivors: List[_Track] = []
        for i, tr in enumerate(self._tracks):
            if i not in matched_track_ids:
                tr.lost += 1
            if tr.lost <= self.max_age:
                survivors.append(tr)
        self._tracks = survivors