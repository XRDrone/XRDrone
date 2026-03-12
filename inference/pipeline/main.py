"""
main.py

Entry point for the XRDrone local inference pipeline.

Coordinates the full runtime loop:
  - Loads YOLO people and fire/smoke models
  - Captures frames from webcam, capture card, or video file
  - Runs detection and optional tracking
  - Merges model outputs into a unified detection list
  - Builds UDP packets for Unity consumption
  - Displays locally with overlays and runtime toggles

Key responsibilities:
  - Pipeline orchestration and runtime control
  - Model initialization and inference scheduling
  - Frame processing, rendering, and UDP publishing
  - Handling keyboard controls and runtime toggles

Provides:
  - run_test(): single-image inference and UDP preview
  - run_live(): continuous live pipeline execution
  - main(): CLI entrypoint for selecting test vs live mode
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import settings as S
from merger import merge_detections
from output_formatter import to_unity_udp_packet
from overlay import apply_rgba_overlay_fullframe, load_rgba_overlay
from pose_estimator import ArucoPoseEstimator, PoseSolution
from streaming import UDPPublisher
from tracker import OpenCVKalmanIOUTracker


def draw_pose_mode_status(
    frame: np.ndarray,
    text: str,
    *,
    enabled: bool = True,
    origin: tuple = (20, 40),
    text_scale: float = 0.9,
    text_thickness: int = 2,
):
    """Draw the current ArUco visibility/mode label on the video frame."""
    if not enabled or frame is None or not text:
        return frame

    h_img, w_img = frame.shape[:2]
    x, y = int(origin[0]), int(origin[1])
    x = max(0, min(w_img - 1, x))
    y = max(20, min(h_img - 1, y))

    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, float(text_scale), int(text_thickness)
    )
    pad = 8
    bx1 = max(0, x - pad)
    by1 = max(0, y - th - pad)
    bx2 = min(w_img - 1, x + tw + pad)
    by2 = min(h_img - 1, y + baseline + pad)

    roi = frame[by1 : by2 + 1, bx1 : bx2 + 1]
    if roi.size > 0:
        black = np.zeros_like(roi)
        cv2.addWeighted(black, 0.45, roi, 0.55, 0.0, dst=roi)
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(text_scale),
        (255, 255, 255),
        int(text_thickness),
        cv2.LINE_AA,
    )
    return frame


def _normalize_names(names):
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def _remap_people_names(names_dict: Dict[int, str]) -> Dict[int, str]:
    """Normalize common custom-label variants to COCO-like labels when possible."""
    names = dict(names_dict)
    inv = {v.lower(): k for k, v in names.items()}

    if len(names) == 1:
        return {0: "person"}

    if "person" not in inv and "item" in inv:
        names[int(inv["item"])] = "person"

    return names


def _find_class_idx(names_dict: Dict[int, str], want: str):
    want = (want or "").lower()
    for k, v in names_dict.items():
        if str(v).lower() == want:
            return int(k)
    return None


def _resolve_class_ids(names_dict: Dict[int, str], wanted_names: Sequence[str]) -> List[int]:
    """Resolve human-friendly class names to model class IDs, with a few aliases."""
    inv = {str(v).lower(): int(k) for k, v in names_dict.items()}

    aliases = {
        "sofa": "couch",
        "dining_table": "dining table",
        "diningtable": "dining table",
    }

    ids: List[int] = []
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


def _letterbox(frame_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize to fit inside target while preserving aspect ratio; pad with black."""
    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return frame_bgr

    scale = min(float(target_w) / float(w), float(target_h) / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)

    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _format_frame(frame_bgr: np.ndarray) -> np.ndarray:
    if not S.FORCE_OUTPUT_1080P:
        return frame_bgr

    tw, th = int(S.OUTPUT_WIDTH), int(S.OUTPUT_HEIGHT)

    if not S.OUTPUT_KEEP_ASPECT:
        return cv2.resize(frame_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)

    return _letterbox(frame_bgr, tw, th)


def _cv_backend_flag(name: str) -> int:
    name = (name or "auto").lower().strip()
    if name == "auto":
        return 0
    mapping = {
        "dshow": "CAP_DSHOW",
        "msmf": "CAP_MSMF",
        "v4l2": "CAP_V4L2",
        "avfoundation": "CAP_AVFOUNDATION",
    }
    attr = mapping.get(name)
    if not attr:
        return 0
    return int(getattr(cv2, attr, 0) or 0)


def _open_capture(input_mode: str, camera_source: str):
    """Returns: cap, is_file_source, target_fps, video_start_wall, input_desc"""
    input_mode = input_mode.lower().strip()

    if input_mode == "file":
        cap = cv2.VideoCapture(S.VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {S.VIDEO_PATH}")
        input_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        target_fps = input_fps if input_fps > 1 else S.DEFAULT_FPS
        return cap, True, target_fps, time.time(), f"file: {S.VIDEO_PATH}"

    cam = camera_source.lower().strip()
    if cam == "capture_card":
        index = int(S.CAPTURE_CARD_INDEX)
    else:
        index = int(S.WEBCAM_INDEX)

    backend_flag = _cv_backend_flag(S.CAPTURE_BACKEND)
    if backend_flag != 0:
        cap = cv2.VideoCapture(index, backend_flag)
        backend_desc = S.CAPTURE_BACKEND
    else:
        cap = cv2.VideoCapture(index)
        backend_desc = "auto"

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index={index} backend={backend_desc}")

    if S.REQUEST_CAMERA_1080P and S.FORCE_OUTPUT_1080P:
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(S.OUTPUT_WIDTH))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(S.OUTPUT_HEIGHT))
        except Exception:
            pass

    input_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    target_fps = input_fps if input_fps > 1 else S.DEFAULT_FPS
    return (
        cap,
        False,
        target_fps,
        time.time(),
        f"camera: {cam} (index={index}, backend={backend_desc})",
    )


def draw_masks(
    frame,
    results,
    names,
    colors,
    default_color=(0, 255, 0),
    alpha=0.35,
    text_scale=0.6,
    text_thickness=2,
    show_label: bool = True,
):
    """Overlay masks (if present) + per-instance labels; supports per-class colors."""
    if not results:
        return frame

    r = results[0]
    boxes = getattr(r, "boxes", None)
    masks = getattr(r, "masks", None)

    if boxes is None or len(boxes) == 0:
        return frame

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy()

    mask_data = None
    if masks is not None and getattr(masks, "data", None) is not None:
        mask_data = masks.data.detach().cpu().numpy()  # (n, h, w)

    n = len(xyxy)
    if mask_data is not None:
        n = min(n, mask_data.shape[0])

    h_img, w_img = frame.shape[:2]

    for i in range(n):
        cls_id = int(cls[i]) if i < len(cls) else 0
        name = str(names.get(cls_id, "obj")).lower()
        color = colors.get(name, default_color)
        color_arr = np.array(color, dtype=np.float32)

        x1, y1, x2, y2 = xyxy[i]
        x1i, y1i = int(x1), int(y1)

        if mask_data is not None:
            m = mask_data[i]
            m = np.squeeze(m)
            if m.ndim == 2:
                if m.shape[:2] != (h_img, w_img):
                    m = cv2.resize(
                        m.astype(np.float32),
                        (w_img, h_img),
                        interpolation=cv2.INTER_NEAREST,
                    )
                m = m > 0.5
                frame[m] = (
                    frame[m].astype(np.float32) * (1.0 - alpha) + color_arr * alpha
                ).astype(np.uint8)

        if show_label:
            label = f"{name} {float(conf[i]) * 100:.1f}%"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
            )
            tx = max(0, min(w_img - tw - 6, x1i))
            ty = max(th + 6, min(h_img - 2, y1i))
            bx1, by1 = tx, ty - th - 6
            bx2, by2 = tx + tw + 6, ty
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, thickness=-1)
            cv2.putText(
                frame,
                label,
                (tx + 3, by2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

    return frame


def draw_tracked_boxes(
    frame: np.ndarray,
    detections: Sequence[dict],
    *,
    colors: Dict[str, tuple],
    default_color=(255, 255, 255),
    text_scale: float = 0.6,
    text_thickness: int = 2,
    box_thickness: int = 2,
):
    """Draw bbox + (class, conf, track_id) from merged detections."""
    h_img, w_img = frame.shape[:2]
    for det in detections:
        bbox = det.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = (int(float(v)) for v in bbox)
        x1 = max(0, min(w_img - 1, x1))
        x2 = max(0, min(w_img - 1, x2))
        y1 = max(0, min(h_img - 1, y1))
        y2 = max(0, min(h_img - 1, y2))

        cls_name = str(det.get("class") or "obj").lower()
        color = colors.get(cls_name, default_color)
        conf = float(det.get("confidence", 0.0))
        tid = det.get("track_id", None)

        if tid is None:
            label = f"{cls_name} {conf * 100:.1f}%"
        else:
            try:
                label = f"{cls_name} #{int(tid)} {conf * 100:.1f}%"
            except Exception:
                label = f"{cls_name} {conf * 100:.1f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, int(box_thickness))

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, float(text_scale), int(text_thickness)
        )
        tx = max(0, min(w_img - tw - 6, x1))
        ty = max(th + 6, min(h_img - 2, y1))
        bx1, by1 = tx, ty - th - 6
        bx2, by2 = tx + tw + 6, ty
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, thickness=-1)
        cv2.putText(
            frame,
            label,
            (tx + 3, by2 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(text_scale),
            (0, 0, 0),
            int(text_thickness),
            cv2.LINE_AA,
        )

    return frame


def _build_models():
    people_seg_model = YOLO(S.PEOPLE_MODEL_PATH)
    fire_model = YOLO(S.FIRE_MODEL_PATH)

    seg_names = _normalize_names(people_seg_model.names)
    fire_names = _normalize_names(fire_model.names)

    people_seg_label = SimpleNamespace(names=_remap_people_names(seg_names))
    fire_label = SimpleNamespace(names=fire_names)

    person_class = _find_class_idx(people_seg_label.names, "person")
    if person_class is None:
        person_class = 0

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    for m in (people_seg_model, fire_model):
        try:
            m.to(S.DEVICE)
        except Exception:
            pass
        try:
            m.fuse()
        except Exception:
            pass

    detect_class_ids = _resolve_class_ids(people_seg_label.names, S.DETECT_CLASSES)
    if not detect_class_ids:
        detect_class_ids = [int(person_class)]

    return people_seg_model, fire_model, people_seg_label, fire_label, int(person_class), detect_class_ids


def _parse_args():
    p = argparse.ArgumentParser(description="XRDrone local YOLO pipeline")
    p.add_argument(
        "-test",
        "--test",
        action="store_true",
        help="Run a single-image test and print the UDP JSON payload.",
    )
    p.add_argument(
        "--test-image",
        default=S.TEST_IMAGE_PATH,
        help="Path to the image used with -test.",
    )
    p.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable OpenCV imshow window (useful for headless runs).",
    )
    return p.parse_args()


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _passes_udp_world_projection_filter(
    det: dict,
    *,
    allowed_classes: Optional[Sequence[str]] = None,
    min_conf: Optional[float] = None,
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


def _attach_foot_and_world(
    detections: List[dict],
    *,
    pose_data: dict,
    pose_solution: Optional[PoseSolution],
    width: int,
    height: int,
    projection_classes: Optional[Sequence[str]] = None,
    projection_min_conf: Optional[float] = None,
) -> None:
    """Attach foot_* and world_* fields to merged detections in-place.

    - foot_* are normalized (0..1) image coords of the bbox bottom-center.
    - world_* are a ray-plane (Y=0) intersection in the ArUco world frame.
    """
    w = max(1, int(width))
    h = max(1, int(height))

    pose_valid = bool(pose_data.get("pose_valid", False)) and pose_solution is not None

    for det in detections:
        # Defaults required by Unity schema.
        det["foot_x"] = float(det.get("foot_x", 0.0))
        det["foot_y"] = float(det.get("foot_y", 0.0))
        det["world_valid"] = bool(det.get("world_valid", False))
        det["world_x"] = float(det.get("world_x", 0.0))
        det["world_y"] = float(det.get("world_y", 0.0))
        det["world_z"] = float(det.get("world_z", 0.0))

        bbox = det.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = (float(v) for v in bbox)

        foot_x_px = (x1 + x2) / 2.0
        foot_y_px = y2

        foot_x_n = _clamp01(foot_x_px / float(w))
        foot_y_n = _clamp01(foot_y_px / float(h))

        det["foot_x"] = float(foot_x_n)
        det["foot_y"] = float(foot_y_n)

        should_project_world = _passes_udp_world_projection_filter(
            det,
            allowed_classes=projection_classes,
            min_conf=projection_min_conf,
        )

        # If pose is valid, try to register the foot point onto the plane Y=0.
        if not pose_valid or not should_project_world:
            det["world_valid"] = False
            det["world_x"] = 0.0
            det["world_y"] = 0.0
            det["world_z"] = 0.0
            continue

        try:
            P_w = pose_solution.intersect_plane_y0(foot_x_px, foot_y_px)
        except Exception:
            P_w = None

        if P_w is None or getattr(P_w, "size", 0) < 3:
            det["world_valid"] = False
            det["world_x"] = 0.0
            det["world_y"] = 0.0
            det["world_z"] = 0.0
            continue

        det["world_valid"] = True
        det["world_x"] = float(P_w[0])
        det["world_y"] = float(P_w[1])
        det["world_z"] = float(P_w[2])


def run_test(args) -> int:
    people_seg_model, fire_model, people_seg_label, fire_label, _, detect_class_ids = _build_models()

    img = cv2.imread(args.test_image)
    if img is None:
        raise RuntimeError(f"Could not read test image: {args.test_image}")

    frame = _format_frame(img)
    now = time.time()
    frame_id = 1

    pose_estimator = ArucoPoseEstimator(
        enabled=bool(getattr(S, "POSE_ENABLED_DEFAULT", True)),
        hfov_deg=float(getattr(S, "POSE_HFOV_DEG", 84.0)),
        marker_size_m=float(getattr(S, "POSE_MARKER_SIZE_M", 0.1645)),
        marker_world_positions=getattr(S, "POSE_MARKER_WORLD_POSITIONS", {0: (0.0, 0.0, 0.0)}),
        aruco_dict_name=str(getattr(S, "POSE_ARUCO_DICT", "DICT_4X4_50")),
        use_case=str(getattr(S, "POSE_USE_CASE", "auto")),
        single_init_solver=str(getattr(S, "POSE_SINGLE_INIT_SOLVER", "ippe_square")),
        multi_init_solver=str(getattr(S, "POSE_MULTI_INIT_SOLVER", "ransac")),
        refiner=str(getattr(S, "POSE_REFINER", "vvs")),
        enable_refinement=bool(getattr(S, "POSE_ENABLE_REFINEMENT", True)),
        min_markers_for_multi=int(getattr(S, "POSE_MIN_MARKERS_FOR_MULTI", 2)),
        corner_refinement=str(getattr(S, "POSE_CORNER_REFINEMENT", "none")),
        ransac_reproj_threshold_px=float(getattr(S, "POSE_RANSAC_REPROJ_THRESHOLD_PX", 4.0)),
        ransac_confidence=float(getattr(S, "POSE_RANSAC_CONFIDENCE", 0.99)),
        ransac_iterations=int(getattr(S, "POSE_RANSAC_ITERATIONS", 100)),
    )
    pose_draw = bool(getattr(S, "POSE_DRAW_ARUCO", False))
    pose_mode_overlay_on = bool(getattr(S, "POSE_MODE_OVERLAY_ENABLED_DEFAULT", True))

    infer_frame = frame.copy() if pose_draw else frame
    pose_data, pose_solution = pose_estimator.estimate_with_solution(frame, draw=pose_draw)

    pred_kw = dict(device=S.DEVICE, half=S.USE_FP16, imgsz=S.IMGSZ, verbose=False)

    people_results = people_seg_model.predict(
        infer_frame, conf=S.PEOPLE_CONF, classes=detect_class_ids, **pred_kw
    )

    fire_results = []
    if S.FIRE_ON_DEFAULT:
        fire_results = fire_model.predict(infer_frame, conf=S.FIRE_CONF, **pred_kw)

    merged = merge_detections(
        people_results,
        fire_results,
        people_model=people_seg_label,
        fire_model=fire_label,
    )

    for det in merged:
        if det.get("class") == "item":
            det["class"] = "person"

    h, w = frame.shape[:2]
    _attach_foot_and_world(
        merged,
        pose_data=pose_data,
        pose_solution=pose_solution,
        width=w,
        height=h,
        projection_classes=S.UDP_SEND_CLASSES,
        projection_min_conf=S.UDP_MIN_CONF,
    )

    pkt = to_unity_udp_packet(
        merged,
        frame_id=frame_id,
        timestamp=now,
        width=w,
        height=h,
        class_map=S.UNITY_CLASS_ID,
        allowed_classes=S.UDP_SEND_CLASSES,
        min_conf=S.UDP_MIN_CONF,
    )

    pkt["pose"] = pose_data

    print("[UDP] JSON payload (one-line):")
    print(json.dumps(pkt))

    print("\n[UDP] JSON payload (pretty):")
    print(json.dumps(pkt, indent=2))

    if S.ENABLE_UDP:
        udp = UDPPublisher(S.UDP_IP, S.UDP_PORT)
        try:
            udp.send_json(pkt)
        finally:
            udp.close()

    if not args.no_gui:
        dji_overlay = load_rgba_overlay(S.DJI_MENU_OVERLAY_PATH)
        if S.DJI_MENU_OVERLAY_ENABLED_DEFAULT and dji_overlay is not None:
            frame = apply_rgba_overlay_fullframe(frame, dji_overlay)

        if pose_mode_overlay_on:
            frame = draw_pose_mode_status(
                frame,
                pose_estimator.get_pose_mode_overlay_text(),
                enabled=pose_mode_overlay_on,
                origin=getattr(S, "POSE_MODE_OVERLAY_ORIGIN", (20, 40)),
                text_scale=float(getattr(S, "POSE_MODE_OVERLAY_TEXT_SCALE", 0.9)),
                text_thickness=int(getattr(S, "POSE_MODE_OVERLAY_TEXT_THICKNESS", 2)),
            )

        cv2.imshow(S.WINDOW_NAME, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


def run_live(args) -> int:
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    people_seg_model, fire_model, people_seg_label, fire_label, _, detect_class_ids = _build_models()

    people_on = bool(S.PEOPLE_ON_DEFAULT)
    fire_on = bool(S.FIRE_ON_DEFAULT)
    recording_enabled = bool(S.RECORDING_ENABLED_DEFAULT)

    draw_detections = bool(S.DRAW_DETECTIONS_DEFAULT)

    tracking_enabled = bool(getattr(S, "TRACKING_ENABLED_DEFAULT", False))
    tracking_method = str(getattr(S, "TRACKING_METHOD", "opencv")).lower().strip()
    draw_track_ids = bool(getattr(S, "DRAW_TRACK_IDS", True))

    tracker = None
    if tracking_enabled and tracking_method == "opencv":
        tracker = OpenCVKalmanIOUTracker(
            min_iou=float(getattr(S, "TRACK_MIN_IOU", 0.30)),
            max_age_frames=int(getattr(S, "TRACK_MAX_AGE_FRAMES", 90)),
            per_class=bool(getattr(S, "TRACK_PER_CLASS", True)),
            process_noise=float(getattr(S, "TRACK_KF_PROCESS_NOISE", 1e-2)),
            measurement_noise=float(getattr(S, "TRACK_KF_MEAS_NOISE", 1e-1)),
        )

    pose_estimator = ArucoPoseEstimator(
        enabled=bool(getattr(S, "POSE_ENABLED_DEFAULT", True)),
        hfov_deg=float(getattr(S, "POSE_HFOV_DEG", 84.0)),
        marker_size_m=float(getattr(S, "POSE_MARKER_SIZE_M", 0.1645)),
        marker_world_positions=getattr(S, "POSE_MARKER_WORLD_POSITIONS", {0: (0.0, 0.0, 0.0)}),
        aruco_dict_name=str(getattr(S, "POSE_ARUCO_DICT", "DICT_4X4_50")),
        use_case=str(getattr(S, "POSE_USE_CASE", "auto")),
        single_init_solver=str(getattr(S, "POSE_SINGLE_INIT_SOLVER", "ippe_square")),
        multi_init_solver=str(getattr(S, "POSE_MULTI_INIT_SOLVER", "ransac")),
        refiner=str(getattr(S, "POSE_REFINER", "vvs")),
        enable_refinement=bool(getattr(S, "POSE_ENABLE_REFINEMENT", True)),
        min_markers_for_multi=int(getattr(S, "POSE_MIN_MARKERS_FOR_MULTI", 2)),
        corner_refinement=str(getattr(S, "POSE_CORNER_REFINEMENT", "none")),
        ransac_reproj_threshold_px=float(getattr(S, "POSE_RANSAC_REPROJ_THRESHOLD_PX", 4.0)),
        ransac_confidence=float(getattr(S, "POSE_RANSAC_CONFIDENCE", 0.99)),
        ransac_iterations=int(getattr(S, "POSE_RANSAC_ITERATIONS", 100)),
    )
    pose_draw = bool(getattr(S, "POSE_DRAW_ARUCO", False))
    pose_mode_overlay_on = bool(getattr(S, "POSE_MODE_OVERLAY_ENABLED_DEFAULT", True))

    dji_overlay_on = bool(S.DJI_MENU_OVERLAY_ENABLED_DEFAULT)
    dji_overlay_bgra = load_rgba_overlay(S.DJI_MENU_OVERLAY_PATH)

    active_camera_source = S.CAMERA_SOURCE_DEFAULT
    fire_class_names = {str(v).lower() for v in fire_label.names.values()}

    fps_hist = deque(maxlen=30)
    drop_hist = deque(maxlen=30)
    t_prev = time.time()

    cap, is_file_source, target_fps, video_start_wall, _input_desc = _open_capture(
        S.INPUT_MODE, active_camera_source
    )

    fourcc = cv2.VideoWriter_fourcc(*S.OUTPUT_CODEC)
    out = None

    window_frames = 0
    window_start = time.time()
    frame_id = 0

    udp = UDPPublisher(S.UDP_IP, S.UDP_PORT) if S.ENABLE_UDP else None

    pred_kw = dict(device=S.DEVICE, half=S.USE_FP16, imgsz=S.IMGSZ, verbose=False)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            if is_file_source:
                t_video = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                now = video_start_wall + t_video
            else:
                now = time.time()

            frame = _format_frame(frame)

            infer_frame = frame.copy() if pose_draw else frame
            pose_data, pose_solution = pose_estimator.estimate_with_solution(frame, draw=pose_draw)

            wall_now = time.time()
            dt = wall_now - t_prev
            t_prev = wall_now
            if dt > 0:
                fps_hist.append(1.0 / dt)

            window_frames += 1
            if wall_now - window_start >= 1.0:
                window_len = wall_now - window_start
                expected_frames = target_fps * window_len
                drops = max(0, int(round(expected_frames - window_frames)))
                drop_hist.append(drops)
                window_frames = 0
                window_start = wall_now

            people_results = []
            fire_results = []

            use_ultra_track = bool(tracking_enabled and tracking_method == "ultralytics")
            ultra_tracker_yaml = str(getattr(S, "ULTRALYTICS_TRACKER", "bytetrack.yaml"))

            if people_on:
                if use_ultra_track:
                    people_results = people_seg_model.track(
                        infer_frame,
                        conf=S.PEOPLE_CONF,
                        classes=detect_class_ids,
                        persist=True,
                        tracker=ultra_tracker_yaml,
                        **pred_kw,
                    )
                else:
                    people_results = people_seg_model.predict(
                        infer_frame,
                        conf=S.PEOPLE_CONF,
                        classes=detect_class_ids,
                        **pred_kw,
                    )

            if fire_on:
                if use_ultra_track:
                    fire_results = fire_model.track(
                        infer_frame,
                        conf=S.FIRE_CONF,
                        persist=True,
                        tracker=ultra_tracker_yaml,
                        **pred_kw,
                    )
                else:
                    fire_results = fire_model.predict(infer_frame, conf=S.FIRE_CONF, **pred_kw)

            merged = merge_detections(
                people_results,
                fire_results,
                people_model=people_seg_label,
                fire_model=fire_label,
            )

            for det in merged:
                if det.get("class") == "item":
                    det["class"] = "person"

                if use_ultra_track and det.get("track_id") is not None:
                    try:
                        base_id = int(det["track_id"])
                        cls_name = str(det.get("class", "")).lower()
                        if cls_name in fire_class_names:
                            det["track_id"] = base_id + int(getattr(S, "TRACK_ID_OFFSET_FIRE", 1_000_000))
                        else:
                            det["track_id"] = base_id + int(getattr(S, "TRACK_ID_OFFSET_PEOPLE", 0))
                    except Exception:
                        pass

            if tracking_enabled and tracking_method == "opencv" and tracker is not None:
                tracker.update(merged)

            # Attach "foot" + optional world registration fields for UDP consumers.
            h_img, w_img = frame.shape[:2]
            _attach_foot_and_world(
                merged,
                pose_data=pose_data,
                pose_solution=pose_solution,
                width=w_img,
                height=h_img,
                projection_classes=S.UDP_SEND_CLASSES,
                projection_min_conf=S.UDP_MIN_CONF,
            )

            want_track_overlay = bool(tracking_enabled and draw_track_ids)

            if draw_detections and people_on:
                frame = draw_masks(
                    frame,
                    people_results,
                    names=people_seg_label.names,
                    colors=S.COLORS,
                    default_color=S.COLORS.get("person", (0, 255, 0)),
                    alpha=S.MASK_ALPHA,
                    text_scale=S.MASK_TEXT_SCALE,
                    text_thickness=S.MASK_TEXT_THICKNESS,
                    show_label=not want_track_overlay,
                )

            if draw_detections and fire_on:
                frame = draw_masks(
                    frame,
                    fire_results,
                    names=fire_label.names,
                    colors=S.COLORS,
                    default_color=S.COLORS.get("fire", (255, 255, 255)),
                    alpha=S.MASK_ALPHA,
                    text_scale=S.MASK_TEXT_SCALE,
                    text_thickness=S.MASK_TEXT_THICKNESS,
                    show_label=not want_track_overlay,
                )

            if draw_detections and want_track_overlay:
                frame = draw_tracked_boxes(
                    frame,
                    merged,
                    colors=S.COLORS,
                    default_color=(255, 255, 255),
                    text_scale=S.MASK_TEXT_SCALE,
                    text_thickness=S.MASK_TEXT_THICKNESS,
                    box_thickness=2,
                )

            if dji_overlay_on and dji_overlay_bgra is not None:
                frame = apply_rgba_overlay_fullframe(frame, dji_overlay_bgra)

            if pose_mode_overlay_on:
                frame = draw_pose_mode_status(
                    frame,
                    pose_estimator.get_pose_mode_overlay_text(),
                    enabled=pose_mode_overlay_on,
                    origin=getattr(S, "POSE_MODE_OVERLAY_ORIGIN", (20, 40)),
                    text_scale=float(getattr(S, "POSE_MODE_OVERLAY_TEXT_SCALE", 0.9)),
                    text_thickness=int(getattr(S, "POSE_MODE_OVERLAY_TEXT_THICKNESS", 2)),
                )

            if S.SAVE_OUTPUT and recording_enabled:
                if out is None:
                    h, w = frame.shape[:2]
                    out = cv2.VideoWriter(S.OUTPUT_VIDEO, fourcc, target_fps, (w, h))
                out.write(frame)

            if udp is not None:
                h, w = frame.shape[:2]
                pkt = to_unity_udp_packet(
                    merged,
                    frame_id=frame_id,
                    timestamp=now,
                    width=w,
                    height=h,
                    class_map=S.UNITY_CLASS_ID,
                    allowed_classes=S.UDP_SEND_CLASSES,
                    min_conf=S.UDP_MIN_CONF,
                )
                pkt["pose"] = pose_data
                try:
                    udp.send_json(pkt)
                except Exception:
                    pass

            if not args.no_gui:
                cv2.imshow(S.WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255

            if key == S.KEY_ESC:
                break

            if key in S.KEY_TOGGLE_RECORDING:
                if not recording_enabled:
                    print("Recording ENABLED at", time.strftime("%Y-%m-%d %H:%M:%S"))
                    recording_enabled = True
                else:
                    print("Recording DISABLED at", time.strftime("%Y-%m-%d %H:%M:%S"))
                    recording_enabled = False

            elif key in S.KEY_TOGGLE_PEOPLE:
                people_on = not people_on

            elif key in S.KEY_TOGGLE_FIRE:
                fire_on = not fire_on

            elif key in S.KEY_TOGGLE_DRAW:
                draw_detections = not draw_detections

            elif key in S.KEY_TOGGLE_DJI_OVERLAY:
                dji_overlay_on = not dji_overlay_on

            elif key in getattr(S, "KEY_TOGGLE_TRACKING", (ord("t"), ord("T"))):
                tracking_enabled = not tracking_enabled
                if tracking_enabled and tracking_method == "opencv":
                    if tracker is None:
                        tracker = OpenCVKalmanIOUTracker(
                            min_iou=float(getattr(S, "TRACK_MIN_IOU", 0.30)),
                            max_age_frames=int(getattr(S, "TRACK_MAX_AGE_FRAMES", 90)),
                            per_class=bool(getattr(S, "TRACK_PER_CLASS", True)),
                            process_noise=float(getattr(S, "TRACK_KF_PROCESS_NOISE", 1e-2)),
                            measurement_noise=float(getattr(S, "TRACK_KF_MEAS_NOISE", 1e-1)),
                        )
                    else:
                        tracker.reset()

            elif key in getattr(S, "KEY_TOGGLE_POSE_MODE_OVERLAY", (ord("m"), ord("M"))):
                pose_mode_overlay_on = not pose_mode_overlay_on

            elif key in S.KEY_TOGGLE_INPUT and S.INPUT_MODE.lower() == "camera":
                prev = active_camera_source
                active_camera_source = "capture_card" if prev == "webcam" else "webcam"

                try:
                    cap.release()
                except Exception:
                    pass

                try:
                    cap, is_file_source, target_fps, video_start_wall, _input_desc = _open_capture(
                        "camera", active_camera_source
                    )

                    fps_hist.clear()
                    drop_hist.clear()
                    t_prev = time.time()
                    window_frames = 0
                    window_start = time.time()

                except Exception as e:
                    print("Toggle input failed:", e)
                    active_camera_source = prev
                    cap, is_file_source, target_fps, video_start_wall, _input_desc = _open_capture(
                        "camera", active_camera_source
                    )

    finally:
        try:
            cap.release()
        except Exception:
            pass

        if out is not None:
            try:
                out.release()
            except Exception:
                pass

        if not args.no_gui:
            cv2.destroyAllWindows()

        if udp is not None:
            udp.close()

    return 0


def main() -> int:
    args = _parse_args()
    if args.test:
        return run_test(args)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())