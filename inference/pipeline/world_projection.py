"""
world_projection.py

Rust-backed foot-point and world-projection helpers.
The Python wrapper only extracts PoseSolution primitives and writes results
back into the provided detection list in-place.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from pose_estimator import PoseSolution

try:
    from xrdrone_native import (
        attach_foot_and_world as _native_attach_foot_and_world,
    )
    from xrdrone_native import (
        clamp01,
        passes_udp_world_projection_filter,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "xrdrone_native is required for world_projection.py. "
        "Build it first with ./build_native.sh or maturin develop --release."
    ) from exc


def _flatten_vec3(value: Any) -> list[float]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError("Expected at least 3 elements")
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _flatten_mat3(value: Any) -> list[float]:
    arr = np.asarray(value, dtype=np.float64).reshape(3, 3)
    return [float(v) for v in arr.reshape(-1)]


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
    pose_valid = bool(pose_data.get("pose_valid", False)) and pose_solution is not None

    pose_camera_world: list[float] | None = None
    pose_rotation_world_to_camera: list[float] | None = None
    pose_intrinsics: list[float] | None = None

    if pose_valid and pose_solution is not None:
        try:
            pose_camera_world = _flatten_vec3(pose_solution.C_w)
            pose_rotation_world_to_camera = _flatten_mat3(pose_solution.R_wc)
            pose_intrinsics = _flatten_mat3(pose_solution.K)
        except Exception:
            pose_valid = False
            pose_camera_world = None
            pose_rotation_world_to_camera = None
            pose_intrinsics = None

    updated = _native_attach_foot_and_world(
        detections,
        pose_valid=pose_valid,
        pose_camera_world=pose_camera_world,
        pose_rotation_world_to_camera=pose_rotation_world_to_camera,
        pose_intrinsics=pose_intrinsics,
        width=int(width),
        height=int(height),
        projection_classes=projection_classes,
        projection_min_conf=projection_min_conf,
    )
    detections[:] = list(updated)


__all__ = [
    "attach_foot_and_world",
    "clamp01",
    "passes_udp_world_projection_filter",
    "PoseSolution",
]
