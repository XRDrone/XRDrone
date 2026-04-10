"""
runtime_builders.py

Factory helpers for constructing the XRDrone runtime subsystems.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import settings as S
import torch
from id_flicker_mitigation import RobustIDFlickerMitigator
from ultralytics import YOLO

# Fixed optimized pipeline policy (removed from settings.py).
ID_FLICKER_MITIGATION_ENABLED = True
ID_FLICKER_REQUIRE_TRACK_ID = True
ID_FLICKER_USE_CONF_EMA = True


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
