"""
main.py

Entry point for the XRDrone local inference pipeline.

Coordinates the full runtime loop:
  - Loads YOLO people and fire/smoke models
  - Captures frames from webcam, capture card, or video file
  - Runs detection and optional tracking
  - Merges model outputs into a unified detection list
  - Builds UDP packets for Unity consumption
  - Streams frames over RTSP and/or displays locally
  - Applies overlays, HUD elements, and runtime toggles

Key responsibilities:
  - Pipeline orchestration and runtime control
  - Model initialization and inference scheduling
  - Frame processing, rendering, and networking
  - Handling keyboard controls and user consent toggles

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
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import settings as S
from merger import count_by_class, merge_detections
from output_formatter import to_unity_udp_packet
from overlay import apply_rgba_overlay_fullframe, load_rgba_overlay
from pose_estimator import ArucoPoseEstimator
from streaming import RTSPStreamer, UDPPublisher
from tracker import OpenCVKalmanIOUTracker


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


def _attach_masks_to_merged(merged, yolo_results, source_prefix: str):
    if not yolo_results:
        return
    r = yolo_results[0]
    masks = getattr(r, "masks", None)
    if masks is None or getattr(masks, "data", None) is None:
        return

    mask_data = masks.data.detach().cpu().numpy()  # (n, h, w)
    idx = 0
    for det in merged:
        if str(det.get("source", "")).lower().startswith(source_prefix.lower()):
            if idx < mask_data.shape[0]:
                det["mask"] = mask_data[idx]
            idx += 1


def _network_allowed(recording_enabled: bool) -> bool:
    return (not S.REQUIRE_CONSENT_FOR_NETWORK) or bool(recording_enabled)


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


def _send_udp_once(udp: UDPPublisher, pkt: dict) -> None:
    msg = json.dumps(pkt)
    print(msg)
    try:
        udp.send_json(pkt)
    except Exception:
        pass


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
    )
    pose_draw = bool(getattr(S, "POSE_DRAW_ARUCO", False))

    infer_frame = frame.copy() if pose_draw else frame
    pose_data = pose_estimator.estimate(frame, draw=pose_draw)

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
        seg_on=bool(S.ATTACH_PEOPLE_MASKS_TO_LOG),
        timestamp=now,
    )

    if S.ATTACH_FIRE_MASKS_TO_LOG:
        _attach_masks_to_merged(merged, fire_results, "fire")

    for det in merged:
        if det.get("class") == "item":
            det["class"] = "person"

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
    )
    pose_draw = bool(getattr(S, "POSE_DRAW_ARUCO", False))

    dji_overlay_on = bool(S.DJI_MENU_OVERLAY_ENABLED_DEFAULT)
    dji_overlay_bgra = load_rgba_overlay(S.DJI_MENU_OVERLAY_PATH)

    active_camera_source = S.CAMERA_SOURCE_DEFAULT

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

    rtsp = RTSPStreamer(S.RTSP_URL, fps=target_fps) if S.ENABLE_RTSP else None
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
            pose_data = pose_estimator.estimate(frame, draw=pose_draw)

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
                seg_on=bool(people_on and S.ATTACH_PEOPLE_MASKS_TO_LOG),
                timestamp=now,
            )

            if fire_on and S.ATTACH_FIRE_MASKS_TO_LOG:
                _attach_masks_to_merged(merged, fire_results, "fire")

            for det in merged:
                det["frame_id"] = frame_id
                if det.get("class") == "item":
                    det["class"] = "person"

                if use_ultra_track and det.get("track_id") is not None:
                    try:
                        base_id = int(det["track_id"])
                        if str(det.get("source", "")).lower().startswith("fire"):
                            det["track_id"] = base_id + int(getattr(S, "TRACK_ID_OFFSET_FIRE", 1_000_000))
                        else:
                            det["track_id"] = base_id + int(getattr(S, "TRACK_ID_OFFSET_PEOPLE", 0))
                    except Exception:
                        pass

            if tracking_enabled and tracking_method == "opencv" and tracker is not None:
                tracker.update(merged)

            counts = count_by_class(merged)
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

            _ = counts

            net_on = _network_allowed(recording_enabled)

            if dji_overlay_on and dji_overlay_bgra is not None:
                frame = apply_rgba_overlay_fullframe(frame, dji_overlay_bgra)

            allow_output = (not S.REQUIRE_CONSENT_FOR_OUTPUT) or recording_enabled
            if S.SAVE_OUTPUT and allow_output:
                if out is None:
                    h, w = frame.shape[:2]
                    out = cv2.VideoWriter(S.OUTPUT_VIDEO, fourcc, target_fps, (w, h))
                out.write(frame)

            if net_on:
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

                if rtsp is not None:
                    rtsp.write(frame)

            if not args.no_gui:
                cv2.imshow(S.WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255

            if key == S.KEY_ESC:
                break

            if key in S.KEY_TOGGLE_RECORDING:
                if not recording_enabled:
                    print(
                        "[PII] USER_CONSENT: Recording ENABLED by user at",
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    recording_enabled = True
                else:
                    print(
                        "[PII] USER_CONSENT: Recording DISABLED at",
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
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

                    if rtsp is not None:
                        rtsp.close()
                        rtsp = RTSPStreamer(S.RTSP_URL, fps=target_fps)

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

        if rtsp is not None:
            rtsp.close()

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