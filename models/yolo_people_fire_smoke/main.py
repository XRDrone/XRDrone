# main.py
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
from detection_logger import save_detections_json
from hud import draw_hud, load_rgba_overlay, apply_rgba_overlay_fullframe
from merger import count_by_class, merge_detections
from output_formatter import to_unity_udp_packet
from streaming import RTSPStreamer, UDPPublisher


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

    # Single-class custom models often label the only class as 0.
    if len(names) == 1:
        return {0: "person"}

    # Some custom models call it "item".
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

        # If the model uses "sofa" instead of "couch" (rare), accept it.
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

    # Best-effort request to camera.
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
    """Send + print the exact JSON string that UDPPublisher sends."""
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

    pred_kw = dict(device=S.DEVICE, half=S.USE_FP16, imgsz=S.IMGSZ, verbose=False)

    people_results = people_seg_model.predict(
        frame, conf=S.PEOPLE_CONF, classes=detect_class_ids, **pred_kw
    )

    fire_results = []
    if S.FIRE_ON_DEFAULT:
        fire_results = fire_model.predict(frame, conf=S.FIRE_CONF, **pred_kw)

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

    # Normalize a couple legacy variants.
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

    # Print the *exact* JSON string the UDP sender uses.
    print("[UDP] JSON payload (one-line):")
    print(json.dumps(pkt))

    # Also print a readable version.
    print("\n[UDP] JSON payload (pretty):")
    print(json.dumps(pkt, indent=2))

    # Optional: send the packet once.
    if S.ENABLE_UDP:
        udp = UDPPublisher(S.UDP_IP, S.UDP_PORT)
        try:
            udp.send_json(pkt)
        finally:
            udp.close()

    # Optional GUI display
    if not args.no_gui:
        # Apply DJI overlay in test view only (purely visual).
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

    # Visual toggles
    draw_detections = bool(S.DRAW_DETECTIONS_DEFAULT)
    hud_enabled = bool(S.HUD_ENABLED_DEFAULT)

    # DJI overlay toggle (purely visual)
    dji_overlay_on = bool(S.DJI_MENU_OVERLAY_ENABLED_DEFAULT)
    dji_overlay_bgra = load_rgba_overlay(S.DJI_MENU_OVERLAY_PATH)

    # Track which camera we are using at runtime (only relevant when INPUT_MODE="camera")
    active_camera_source = S.CAMERA_SOURCE_DEFAULT  # "webcam" | "capture_card"

    all_detections: List[dict] = []

    fps_hist = deque(maxlen=30)
    inf_hist = deque(maxlen=30)
    drop_hist = deque(maxlen=30)
    t_prev = time.time()

    cap, is_file_source, target_fps, video_start_wall, input_desc = _open_capture(
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

            # Timestamp
            if is_file_source:
                t_video = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                now = video_start_wall + t_video
            else:
                now = time.time()

            # Force output/display/UDP resolution.
            frame = _format_frame(frame)

            # FPS estimates
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

            # Inference
            people_results = []
            fire_results = []

            if people_on:
                people_results = people_seg_model.predict(
                    frame,
                    conf=S.PEOPLE_CONF,
                    classes=detect_class_ids,
                    **pred_kw,
                )

            if fire_on:
                fire_results = fire_model.predict(frame, conf=S.FIRE_CONF, **pred_kw)

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

            if merged:
                all_detections.extend(merged)

            counts = count_by_class(merged)

            # Visual overlays (drawing only; UDP still runs)
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
                )

            if draw_detections and fire_on:
                frame = draw_masks(
                    frame,
                    fire_results,
                    names=fire_label.names,
                    colors=S.COLORS,
                    default_color=S.COLORS.get("fire", (0, 255, 255)),
                    alpha=S.MASK_ALPHA,
                    text_scale=S.MASK_TEXT_SCALE,
                    text_thickness=S.MASK_TEXT_THICKNESS,
                )

            # Inference timing (Ultralytics provides ms)
            inf_times = []
            if people_on and people_results:
                inf_times.append(people_results[0].speed.get("inference", 0.0))
            if fire_on and fire_results:
                inf_times.append(fire_results[0].speed.get("inference", 0.0))
            if inf_times:
                inf_hist.append(sum(inf_times) / len(inf_times))

            avg_fps = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0
            avg_inf = sum(inf_hist) / len(inf_hist) if inf_hist else 0.0
            avg_drops = sum(drop_hist) / len(drop_hist) if drop_hist else 0.0

            people_count = counts.get("person", 0) + counts.get("item", 0)
            fire_count = counts.get("fire", 0)
            smoke_count = counts.get("smoke", 0)
            chair_count = counts.get("chair", 0)
            couch_count = counts.get("couch", 0) + counts.get("sofa", 0)
            table_count = counts.get("dining table", 0)

            net_on = _network_allowed(recording_enabled)

            # HUD
            if hud_enabled:
                lines = [
                    f"FPS: {avg_fps:5.2f}",
                    f"Model inference: {avg_inf:5.1f} ms",
                    f"People: {people_count}",
                    f"Chair: {chair_count}",
                    f"Couch/Sofa: {couch_count}",
                    f"Dining table: {table_count}",
                    f"Fire: {fire_count}",
                    f"Smoke: {smoke_count}",
                    f"Dropped frames (avg/s): {avg_drops:.1f}",
                    f"Input: {input_desc}",
                    f"HUD: {'ON' if hud_enabled else 'OFF'} (H)",
                    f"Det overlays: {'ON' if draw_detections else 'OFF'} (V)",
                    f"DJI overlay: {'ON' if (dji_overlay_on and dji_overlay_bgra is not None) else 'OFF'} (U)",
                    f"RTSP: {'ON' if (rtsp and net_on) else 'OFF'}",
                    f"UDP:  {'ON' if (udp and net_on) else 'OFF'}",
                    f"RECORDING: {'ON' if recording_enabled else 'OFF'} (R)",
                    f"People model: {'ON' if people_on else 'OFF'} (K)",
                    f"Fire/Smoke model: {'ON' if fire_on else 'OFF'} (L)",
                    f"Toggle input: (I)",
                ]
                frame = draw_hud(
                    frame,
                    lines,
                    anchor=S.HUD_ANCHOR,
                    margin=S.HUD_MARGIN,
                    alpha=S.HUD_ALPHA,
                    font_scale=S.HUD_FONT_SCALE,
                    thickness=S.HUD_THICKNESS,
                )

            # DJI menu overlay (PNG on top of everything; purely visual)
            if dji_overlay_on and dji_overlay_bgra is not None:
                frame = apply_rgba_overlay_fullframe(frame, dji_overlay_bgra)

            # Output video
            allow_output = (not S.REQUIRE_CONSENT_FOR_OUTPUT) or recording_enabled
            if S.SAVE_OUTPUT and allow_output:
                if out is None:
                    h, w = frame.shape[:2]
                    out = cv2.VideoWriter(S.OUTPUT_VIDEO, fourcc, target_fps, (w, h))
                out.write(frame)

            # UDP/RTSP
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

            elif key in S.KEY_TOGGLE_HUD:
                hud_enabled = not hud_enabled

            elif key in S.KEY_TOGGLE_DJI_OVERLAY:
                dji_overlay_on = not dji_overlay_on

            elif key in S.KEY_TOGGLE_INPUT and S.INPUT_MODE.lower() == "camera":
                prev = active_camera_source
                active_camera_source = "capture_card" if prev == "webcam" else "webcam"

                try:
                    cap.release()
                except Exception:
                    pass

                try:
                    cap, is_file_source, target_fps, video_start_wall, input_desc = _open_capture(
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
                    cap, is_file_source, target_fps, video_start_wall, input_desc = _open_capture(
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

    allow_log = (not S.REQUIRE_CONSENT_FOR_LOG) or recording_enabled
    if all_detections and allow_log:
        save_detections_json(all_detections, S.DETECTION_LOG_PATH)

    return 0


def main() -> int:
    args = _parse_args()
    if args.test:
        return run_test(args)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())