"""
tracker.py

Persistent, lightweight multi-object tracking for XRDrone.

Assigns stable track IDs to detections across frames using:
  - OpenCV Kalman filtering (constant velocity motion model)
  - Composite data association that can fuse:
      - bbox IoU
      - normalized foot-point distance
      - ground-plane distance when world registration is available

Works directly with merged detection dicts from merger.py.

Capabilities:
  - Maintains track continuity through short occlusions
  - Optional per-class tracking isolation
  - Configurable track lifetimes and matching thresholds
  - SORT-style motion prediction with stronger drone-aware matching

Limitations:
  - No learned appearance-based re-identification
  - Long disappear/reappear events can still receive new IDs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None  # type: ignore


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


def _iou_matrix_xyxy(track_bboxes: Sequence[Sequence[float]], det_bboxes: Sequence[Sequence[float]]) -> np.ndarray:
    """Vectorized IoU matrix with xyxy semantics."""
    if not track_bboxes or not det_bboxes:
        return np.zeros((len(track_bboxes), len(det_bboxes)), dtype=np.float32)

    tracks = np.asarray(track_bboxes, dtype=np.float32)
    dets = np.asarray(det_bboxes, dtype=np.float32)

    ax1 = tracks[:, 0:1]
    ay1 = tracks[:, 1:2]
    ax2 = tracks[:, 2:3]
    ay2 = tracks[:, 3:4]

    bx1 = dets[None, :, 0]
    by1 = dets[None, :, 1]
    bx2 = dets[None, :, 2]
    by2 = dets[None, :, 3]

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    denom = area_a + area_b - inter

    out = np.zeros_like(inter, dtype=np.float32)
    np.divide(inter, denom, out=out, where=denom > 0.0)
    return out.astype(np.float32, copy=False)


def _as_optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


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
    foot_x: Optional[float] = None
    foot_y: Optional[float] = None
    world_valid: bool = False
    world_x: float = 0.0
    world_z: float = 0.0


class OpenCVKalmanIOUTracker:
    """SORT-like tracker with drone-aware association."""

    def __init__(
        self,
        *,
        min_iou: float = 0.30,
        max_age_frames: int = 90,
        per_class: bool = True,
        dt: float = 1.0,
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
        matching_method: str = "greedy",
        min_match_score: float = 0.45,
        max_foot_distance_norm: float = 0.08,
        max_world_distance_m: float = 2.5,
        use_world_position: bool = True,
        world_score_weight: float = 0.65,
        iou_score_weight: float = 0.25,
        foot_score_weight: float = 0.10,
    ) -> None:
        self.min_iou = float(min_iou)
        self.max_age = int(max_age_frames)
        self.per_class = bool(per_class)
        self.dt = float(dt)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.matching_method = str(matching_method or "greedy").strip().lower()

        self.min_match_score = float(min_match_score)
        self.max_foot_distance_norm = max(1e-6, float(max_foot_distance_norm))
        self.max_world_distance_m = max(1e-6, float(max_world_distance_m))
        self.use_world_position = bool(use_world_position)
        self.world_score_weight = max(0.0, float(world_score_weight))
        self.iou_score_weight = max(0.0, float(iou_score_weight))
        self.foot_score_weight = max(0.0, float(foot_score_weight))

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

    def _update_track_spatial_memory(self, tr: _Track, det: Dict[str, Any]) -> None:
        tr.foot_x = _as_optional_float(det.get("foot_x"))
        tr.foot_y = _as_optional_float(det.get("foot_y"))
        tr.world_valid = bool(det.get("world_valid", False))
        if tr.world_valid:
            tr.world_x = float(det.get("world_x", 0.0))
            tr.world_z = float(det.get("world_z", 0.0))
        else:
            tr.world_x = 0.0
            tr.world_z = 0.0

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
        self._update_track_spatial_memory(tr, det)
        self._next_id += 1
        return tr

    def _foot_similarity(self, tr: _Track, det: Dict[str, Any]) -> float:
        if tr.foot_x is None or tr.foot_y is None:
            return 0.0
        det_fx = _as_optional_float(det.get("foot_x"))
        det_fy = _as_optional_float(det.get("foot_y"))
        if det_fx is None or det_fy is None:
            return 0.0
        dist = float(np.hypot(det_fx - tr.foot_x, det_fy - tr.foot_y))
        if dist > self.max_foot_distance_norm:
            return 0.0
        return max(0.0, 1.0 - dist / self.max_foot_distance_norm)

    def _world_similarity(self, tr: _Track, det: Dict[str, Any]) -> float:
        if not self.use_world_position or not tr.world_valid or not bool(det.get("world_valid", False)):
            return 0.0
        try:
            det_wx = float(det.get("world_x", 0.0))
            det_wz = float(det.get("world_z", 0.0))
        except Exception:
            return 0.0
        dist = float(np.hypot(det_wx - tr.world_x, det_wz - tr.world_z))
        if dist > self.max_world_distance_m:
            return 0.0
        return max(0.0, 1.0 - dist / self.max_world_distance_m)

    def _build_score_matrix(
        self,
        track_ids: List[int],
        det_ids: List[int],
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        if not track_ids or not det_ids:
            return np.zeros((len(track_ids), len(det_ids)), dtype=np.float32)

        track_bboxes = [self._tracks[ti].bbox_xyxy for ti in track_ids]
        det_bboxes: List[List[float]] = []
        for di in det_ids:
            bb = detections[di].get("bbox_xyxy")
            if not bb or len(bb) != 4:
                bb = [0.0, 0.0, 0.0, 0.0]
            det_bboxes.append(list(map(float, bb)))

        iou_mat = _iou_matrix_xyxy(track_bboxes, det_bboxes)
        score_mat = np.zeros_like(iou_mat, dtype=np.float32)

        for local_ti, ti in enumerate(track_ids):
            tr = self._tracks[ti]
            for local_di, di in enumerate(det_ids):
                det = detections[di]
                iou = float(iou_mat[local_ti, local_di])
                foot_sim = self._foot_similarity(tr, det)
                world_sim = self._world_similarity(tr, det)

                use_world_pair = self.use_world_position and tr.world_valid and bool(det.get("world_valid", False))
                use_foot_pair = foot_sim > 0.0

                if use_world_pair:
                    # World-space association is the primary drone-specific cue; IoU/foot act as tie-breakers.
                    numer = self.world_score_weight * world_sim
                    denom = self.world_score_weight

                    if self.iou_score_weight > 0.0:
                        numer += self.iou_score_weight * iou
                        denom += self.iou_score_weight
                    if use_foot_pair and self.foot_score_weight > 0.0:
                        numer += self.foot_score_weight * foot_sim
                        denom += self.foot_score_weight

                    score = numer / denom if denom > 0.0 else 0.0
                    if world_sim <= 0.0:
                        score = 0.0
                else:
                    # Fallback when world registration is unavailable: require either enough IoU or a close foot point.
                    if iou < self.min_iou and not use_foot_pair:
                        score_mat[local_ti, local_di] = 0.0
                        continue

                    numer = 0.0
                    denom = 0.0
                    if self.iou_score_weight > 0.0:
                        numer += self.iou_score_weight * iou
                        denom += self.iou_score_weight
                    if use_foot_pair and self.foot_score_weight > 0.0:
                        numer += self.foot_score_weight * foot_sim
                        denom += self.foot_score_weight
                    score = numer / denom if denom > 0.0 else 0.0

                score_mat[local_ti, local_di] = float(max(0.0, min(1.0, score)))

        return score_mat

    def _greedy_score_match(
        self,
        score_mat: np.ndarray,
        min_score: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        n_tracks, n_dets = score_mat.shape
        if n_tracks == 0 or n_dets == 0:
            return [], list(range(n_tracks)), list(range(n_dets))

        work = score_mat.copy()
        matches: List[Tuple[int, int]] = []
        used_tracks = np.zeros(n_tracks, dtype=bool)
        used_dets = np.zeros(n_dets, dtype=bool)
        thresh = float(min_score)

        while work.size:
            flat_idx = int(np.argmax(work))
            best_val = float(work.reshape(-1)[flat_idx])
            if best_val < thresh:
                break

            best_i, best_j = np.unravel_index(flat_idx, work.shape)
            matches.append((int(best_i), int(best_j)))
            used_tracks[int(best_i)] = True
            used_dets[int(best_j)] = True
            work[int(best_i), :] = -1.0
            work[:, int(best_j)] = -1.0

        unmatched_tracks = np.flatnonzero(~used_tracks).astype(int).tolist()
        unmatched_dets = np.flatnonzero(~used_dets).astype(int).tolist()
        return matches, unmatched_tracks, unmatched_dets

    def _hungarian_score_match(
        self,
        score_mat: np.ndarray,
        min_score: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        n_tracks, n_dets = score_mat.shape
        if n_tracks == 0 or n_dets == 0:
            return [], list(range(n_tracks)), list(range(n_dets))

        if linear_sum_assignment is None:
            return self._greedy_score_match(score_mat, min_score)

        thresh = float(min_score)
        cost = np.where(score_mat >= thresh, 1.0 - score_mat, 1e6).astype(np.float32, copy=False)
        row_ind, col_ind = linear_sum_assignment(cost)

        matches: List[Tuple[int, int]] = []
        used_tracks = np.zeros(n_tracks, dtype=bool)
        used_dets = np.zeros(n_dets, dtype=bool)

        for ti, di in zip(row_ind.tolist(), col_ind.tolist()):
            if float(score_mat[ti, di]) < thresh:
                continue
            matches.append((int(ti), int(di)))
            used_tracks[int(ti)] = True
            used_dets[int(di)] = True

        unmatched_tracks = np.flatnonzero(~used_tracks).astype(int).tolist()
        unmatched_dets = np.flatnonzero(~used_dets).astype(int).tolist()
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
            if self.per_class and g_name != "__all__":
                track_ids = [i for i, tr in enumerate(self._tracks) if tr.cls_name == g_name]
            else:
                track_ids = list(range(len(self._tracks)))

            if not track_ids and det_ids:
                for di in det_ids:
                    tr = self._spawn_track(detections[di])
                    self._tracks.append(tr)
                    detections[di]["track_id"] = tr.track_id
                    matched_track_ids.add(len(self._tracks) - 1)
                continue

            if not det_ids:
                continue

            score_mat = self._build_score_matrix(track_ids, det_ids, detections)
            if self.matching_method == "hungarian":
                matches, _, un_det = self._hungarian_score_match(score_mat, self.min_match_score)
            else:
                matches, _, un_det = self._greedy_score_match(score_mat, self.min_match_score)

            for local_ti, local_di in matches:
                ti = track_ids[local_ti]
                di = det_ids[local_di]
                tr = self._tracks[ti]

                bbox = detections[di].get("bbox_xyxy")
                if not bbox or len(bbox) != 4:
                    bbox = [0.0, 0.0, 0.0, 0.0]
                bbox = list(map(float, bbox))
                cx, cy, w, h = _xyxy_to_cxcywh(bbox)

                self._correct_track(tr, cx, cy)
                tr.bbox_xyxy = bbox
                tr.w = float(w)
                tr.h = float(h)
                tr.conf = float(detections[di].get("confidence", tr.conf))
                tr.hits += 1
                tr.lost = 0
                self._update_track_spatial_memory(tr, detections[di])

                detections[di]["track_id"] = int(tr.track_id)
                matched_track_ids.add(ti)

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
