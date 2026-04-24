"""
runtime_builders.py

Factory helpers for constructing the XRDrone runtime subsystems.

Keeps object creation and settings translation out of the entrypoint and
out of the live/test runners so those modules can focus on orchestration.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import settings as S
import torch
from adaptive_tuning import AdaptiveRuntimeTuner
from id_flicker_mitigation import RobustIDFlickerMitigator
from motion_smoothing import PoseMotionSmoother, WorldTrackSmoother
from pose_estimator import ArucoPoseEstimator
from ultralytics import YOLO

# Fixed optimized pipeline policy (removed from settings.py).
ID_FLICKER_MITIGATION_ENABLED = True
ID_FLICKER_REQUIRE_TRACK_ID = True
ID_FLICKER_USE_CONF_EMA = True

POSE_ENABLE_REFINEMENT = True
POSE_LOSS_HOLD_ENABLED = True
POSE_MOTION_SMOOTHING_ENABLED = True
WORLD_MOTION_SMOOTHING_ENABLED = True


def normalize_names(names):
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list | tuple):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def remap_people_names(names_dict: dict[int, str]) -> dict[int, str]:
    """Normalize common custom-label variants to COCO-like labels when possible."""
    names = dict(names_dict)
    inv = {v.lower(): k for k, v in names.items()}

    if len(names) == 1:
        return {0: "person"}

    if "person" not in inv and "item" in inv:
        names[int(inv["item"])] = "person"

    return names


def find_class_idx(names_dict: dict[int, str], want: str):
    want = (want or "").lower()
    for k, v in names_dict.items():
        if str(v).lower() == want:
            return int(k)
    return None


def resolve_class_ids(names_dict: dict[int, str], wanted_names: Sequence[str]) -> list[int]:
    """Resolve human-friendly class names to model class IDs, with a few aliases."""
    inv = {str(v).lower(): int(k) for k, v in names_dict.items()}

    aliases = {
        "sofa": "couch",
        "dining_table": "dining table",
        "diningtable": "dining table",
    }

    ids: list[int] = []
    for raw in wanted_names:
        name = str(raw).strip().lower()
        if name in inv:
            ids.append(inv[name])
            continue

        ali = aliases.get(name)
        if ali and ali in inv:
            ids.append(inv[ali])
            continue

        if name == "couch" and "sofa" in inv:
            ids.append(inv["sofa"])

    return sorted(set(ids))


def build_models():
    people_seg_model = YOLO(S.PEOPLE_MODEL_PATH)
    fire_model = YOLO(S.FIRE_MODEL_PATH)

    seg_names = normalize_names(people_seg_model.names)
    fire_names = normalize_names(fire_model.names)

    people_seg_label = SimpleNamespace(names=remap_people_names(seg_names))
    fire_label = SimpleNamespace(names=fire_names)

    person_class = find_class_idx(people_seg_label.names, "person")
    if person_class is None:
        person_class = 0

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    for model in (people_seg_model, fire_model):
        try:
            model.to(S.DEVICE)
        except Exception:
            pass
        try:
            model.fuse()
        except Exception:
            pass

    detect_class_ids = resolve_class_ids(people_seg_label.names, S.DETECT_CLASSES)
    if not detect_class_ids:
        detect_class_ids = [int(person_class)]

    return (
        people_seg_model,
        fire_model,
        people_seg_label,
        fire_label,
        int(person_class),
        detect_class_ids,
    )


def build_pose_estimator() -> ArucoPoseEstimator:
    return ArucoPoseEstimator(
        enabled=bool(getattr(S, "POSE_ENABLED_DEFAULT", True)),
        hfov_deg=float(getattr(S, "POSE_HFOV_DEG", 84.0)),
        marker_size_m=float(getattr(S, "POSE_MARKER_SIZE_M", 0.1645)),
        marker_world_positions=getattr(S, "POSE_MARKER_WORLD_POSITIONS", {0: (0.0, 0.0, 0.0)}),
        aruco_dict_name=str(getattr(S, "POSE_ARUCO_DICT", "DICT_4X4_50")),
        use_case=str(getattr(S, "POSE_USE_CASE", "auto")),
        single_init_solver=str(getattr(S, "POSE_SINGLE_INIT_SOLVER", "ippe_square")),
        multi_init_solver=str(getattr(S, "POSE_MULTI_INIT_SOLVER", "ransac")),
        refiner=str(getattr(S, "POSE_REFINER", "vvs")),
        enable_refinement=POSE_ENABLE_REFINEMENT,
        min_markers_for_multi=int(getattr(S, "POSE_MIN_MARKERS_FOR_MULTI", 2)),
        corner_refinement=str(getattr(S, "POSE_CORNER_REFINEMENT", "none")),
        ransac_reproj_threshold_px=float(getattr(S, "POSE_RANSAC_REPROJ_THRESHOLD_PX", 4.0)),
        ransac_confidence=float(getattr(S, "POSE_RANSAC_CONFIDENCE", 0.99)),
        ransac_iterations=int(getattr(S, "POSE_RANSAC_ITERATIONS", 100)),
        pose_loss_hold_enabled=POSE_LOSS_HOLD_ENABLED,
        pose_loss_hold_timeout_s=float(getattr(S, "POSE_LOSS_HOLD_TIMEOUT_S", 0.35)),
        pose_loss_preserve_last_numbers_during_hold=bool(
            getattr(S, "POSE_LOSS_PRESERVE_LAST_NUMBERS_DURING_HOLD", True)
        ),
        pose_loss_clear_numbers_after_timeout=bool(
            getattr(S, "POSE_LOSS_CLEAR_NUMBERS_AFTER_TIMEOUT", True)
        ),
    )


def make_pose_motion_smoother() -> PoseMotionSmoother:
    return PoseMotionSmoother(
        enabled=POSE_MOTION_SMOOTHING_ENABLED,
        smoothness=float(getattr(S, "MOTION_SMOOTHING", 0.0)),
        derivative_cutoff_hz=float(getattr(S, "MOTION_SMOOTHING_DERIVATIVE_CUTOFF_HZ", 1.0)),
        reset_timeout_s=float(getattr(S, "MOTION_SMOOTHING_RESET_TIMEOUT_S", 0.75)),
        default_fps=float(getattr(S, "DEFAULT_FPS", 30.0)),
    )


def make_world_motion_smoother() -> WorldTrackSmoother:
    return WorldTrackSmoother(
        enabled=WORLD_MOTION_SMOOTHING_ENABLED,
        smoothness=float(getattr(S, "MOTION_SMOOTHING", 0.0)),
        derivative_cutoff_hz=float(getattr(S, "MOTION_SMOOTHING_DERIVATIVE_CUTOFF_HZ", 1.0)),
        reset_timeout_s=float(getattr(S, "MOTION_SMOOTHING_RESET_TIMEOUT_S", 0.75)),
        max_track_age_s=float(getattr(S, "WORLD_MOTION_SMOOTHING_MAX_TRACK_AGE_S", 1.5)),
        default_fps=float(getattr(S, "DEFAULT_FPS", 30.0)),
    )


def make_id_flicker_mitigator() -> RobustIDFlickerMitigator:
    return RobustIDFlickerMitigator(
        enabled=ID_FLICKER_MITIGATION_ENABLED,
        apply_classes=getattr(S, "ID_FLICKER_APPLY_CLASSES", ("person",)),
        tau_on=float(getattr(S, "ID_FLICKER_TAU_ON", getattr(S, "UDP_MIN_CONF", 0.80))),
        tau_off=float(getattr(S, "ID_FLICKER_TAU_OFF", 0.55)),
        coast_frames=int(getattr(S, "ID_FLICKER_COAST_FRAMES", 6)),
        drop_frames=int(getattr(S, "ID_FLICKER_DROP_FRAMES", 45)),
        ema_alpha=float(getattr(S, "ID_FLICKER_EMA_ALPHA", 0.45)),
        use_conf_ema=ID_FLICKER_USE_CONF_EMA,
        require_track_id=ID_FLICKER_REQUIRE_TRACK_ID,
        coast_conf_decay=float(getattr(S, "ID_FLICKER_COAST_CONF_DECAY", 0.985)),
    )


def make_adaptive_runtime_tuner() -> AdaptiveRuntimeTuner:
    return AdaptiveRuntimeTuner(
        enabled=bool(getattr(S, "ADAPTIVE_TUNING_ENABLED", True)),
        target_classes=getattr(S, "ADAPTIVE_TUNING_TARGET_CLASSES", ("person",)),
        window_frames=int(getattr(S, "ADAPTIVE_TUNING_WINDOW_FRAMES", 45)),
        update_interval_frames=int(getattr(S, "ADAPTIVE_TUNING_UPDATE_INTERVAL_FRAMES", 15)),
        cooldown_frames=int(getattr(S, "ADAPTIVE_TUNING_COOLDOWN_FRAMES", 30)),
        iou_match_threshold=float(getattr(S, "ADAPTIVE_TUNING_IOU_MATCH_THRESHOLD", 0.35)),
        motion_smoothing_min=float(getattr(S, "ADAPTIVE_MOTION_SMOOTHING_MIN", 0.30)),
        motion_smoothing_max=float(getattr(S, "ADAPTIVE_MOTION_SMOOTHING_MAX", 0.85)),
        motion_smoothing_step=float(getattr(S, "ADAPTIVE_MOTION_SMOOTHING_STEP", 0.05)),
        id_tau_on_min=float(getattr(S, "ADAPTIVE_ID_TAU_ON_MIN", 0.75)),
        id_tau_on_max=float(getattr(S, "ADAPTIVE_ID_TAU_ON_MAX", 0.90)),
        id_tau_off_min=float(getattr(S, "ADAPTIVE_ID_TAU_OFF_MIN", 0.45)),
        id_tau_off_max=float(getattr(S, "ADAPTIVE_ID_TAU_OFF_MAX", 0.65)),
        id_tau_step=float(getattr(S, "ADAPTIVE_ID_TAU_STEP", 0.02)),
        id_coast_frames_min=int(getattr(S, "ADAPTIVE_ID_COAST_FRAMES_MIN", 3)),
        id_coast_frames_max=int(getattr(S, "ADAPTIVE_ID_COAST_FRAMES_MAX", 10)),
        id_coast_step=int(getattr(S, "ADAPTIVE_ID_COAST_STEP", 1)),
        base_motion_smoothing=float(getattr(S, "MOTION_SMOOTHING", 0.0)),
        base_tau_on=float(getattr(S, "ID_FLICKER_TAU_ON", getattr(S, "UDP_MIN_CONF", 0.80))),
        base_tau_off=float(getattr(S, "ID_FLICKER_TAU_OFF", 0.55)),
        base_coast_frames=int(getattr(S, "ID_FLICKER_COAST_FRAMES", 6)),
    )
