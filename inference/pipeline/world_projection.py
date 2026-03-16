"""
world_projection.py

Helpers for attaching image-footpoint and ArUco-world registration data to
merged detections before UDP formatting.
"""

from __future__ import annotations

from collections.abc import Sequence

from pose_estimator import PoseSolution


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def passes_udp_world_projection_filter(
    det: dict,
    *,
    allowed_classes: Sequence[str] | None = None,
    min_conf: float | None = None,
) -> bool:
    cls_name = str(det.get("class") or det.get("class_name") or "").lower()
    if allowed_classes is not None:
        allow = {str(c).lower() for c in allowed_classes}
        if cls_name not in allow:
            return False

    if min_conf is not None:
        try:
            conf = float(det.get("confidence", 0.0))
        except Exception:
            return False
        if conf < float(min_conf):
            return False

    return True


def attach_foot_and_world(
    detections: list[dict],
    *,
    pose_data: dict,
    pose_solution: PoseSolution | None,
    width: int,
    height: int,
    projection_classes: Sequence[str] | None = None,
    projection_min_conf: float | None = None,
) -> None:
    """Attach foot_* and world_* fields to merged detections in-place."""
    width = max(1, int(width))
    height = max(1, int(height))

    pose_valid = bool(pose_data.get("pose_valid", False)) and pose_solution is not None

    for det in detections:
        det["foot_x"] = float(det.get("foot_x", 0.0))
        det["foot_y"] = float(det.get("foot_y", 0.0))
        det["world_valid"] = bool(det.get("world_valid", False))
        det["world_x"] = float(det.get("world_x", 0.0))
        det["world_y"] = float(det.get("world_y", 0.0))
        det["world_z"] = float(det.get("world_z", 0.0))

        bbox = det.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue

        x1, _y1, x2, y2 = (float(v) for v in bbox)

        foot_x_px = (x1 + x2) / 2.0
        foot_y_px = y2

        det["foot_x"] = float(clamp01(foot_x_px / float(width)))
        det["foot_y"] = float(clamp01(foot_y_px / float(height)))

        should_project_world = passes_udp_world_projection_filter(
            det,
            allowed_classes=projection_classes,
            min_conf=projection_min_conf,
        )

        if not pose_valid or not should_project_world:
            det["world_valid"] = False
            det["world_x"] = 0.0
            det["world_y"] = 0.0
            det["world_z"] = 0.0
            continue

        try:
            point_world = pose_solution.intersect_plane_y0(foot_x_px, foot_y_px)
        except Exception:
            point_world = None

        if point_world is None or getattr(point_world, "size", 0) < 3:
            det["world_valid"] = False
            det["world_x"] = 0.0
            det["world_y"] = 0.0
            det["world_z"] = 0.0
            continue

        det["world_valid"] = True
        det["world_x"] = float(point_world[0])
        det["world_y"] = float(point_world[1])
        det["world_z"] = float(point_world[2])
