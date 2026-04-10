"""
live_runner.py

Continuous live runtime for the XRDrone pipeline.
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import settings as S
import torch
from frame_io import format_frame, open_capture
from merger import merge_detections
from output_formatter import to_unity_udp_packet
from overlay import apply_rgba_overlay_fullframe, load_rgba_overlay
from rendering import draw_masks, draw_tracked_boxes
from runtime_builders import build_models, make_id_flicker_mitigator
from runtime_controls import LiveRuntimeState, handle_runtime_key
from streaming import UDPPublisher

# Fixed optimized pipeline policy (removed from settings.py).
TRACKING_ENABLED = True
ULTRALYTICS_TRACKER_YAML = "botsort_drone.yaml"
TRACKING_INPUT_CONF_PEOPLE = 0.10
TRACKING_INPUT_CONF_FIRE = 0.10


def _print_device_info() -> None:
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))


def _normalize_merged_detections(
    merged: list[dict], *, use_ultra_track: bool, fire_class_names: set[str]
):
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


def _attach_detection_footpoints(merged: list[dict], *, width: int, height: int) -> None:
    width_f = float(max(1, width))
    height_f = float(max(1, height))
    for det in merged:
        bbox = det.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        x1, _y1, x2, y2 = (float(v) for v in bbox)
        det["foot_x"] = max(0.0, min(1.0, ((x1 + x2) * 0.5) / width_f))
        det["foot_y"] = max(0.0, min(1.0, y2 / height_f))


def _run_model_inference(
    *,
    infer_frame,
    state: LiveRuntimeState,
    people_seg_model,
    fire_model,
    detect_class_ids,
    pred_kw: dict,
):
    people_results = []
    fire_results = []

    use_ultra_track = bool(state.tracking_enabled)

    if state.people_on:
        if use_ultra_track:
            people_results = people_seg_model.track(
                infer_frame,
                conf=TRACKING_INPUT_CONF_PEOPLE,
                classes=detect_class_ids,
                persist=True,
                tracker=ULTRALYTICS_TRACKER_YAML,
                **pred_kw,
            )
        else:
            people_results = people_seg_model.predict(
                infer_frame,
                conf=S.PEOPLE_CONF,
                classes=detect_class_ids,
                **pred_kw,
            )

    if state.fire_on:
        if use_ultra_track:
            fire_results = fire_model.track(
                infer_frame,
                conf=TRACKING_INPUT_CONF_FIRE,
                persist=True,
                tracker=ULTRALYTICS_TRACKER_YAML,
                **pred_kw,
            )
        else:
            fire_results = fire_model.predict(infer_frame, conf=S.FIRE_CONF, **pred_kw)

    return people_results, fire_results, use_ultra_track


def _render_runtime_frame(
    *,
    frame,
    people_results,
    fire_results,
    people_seg_label,
    fire_label,
    udp_ready_detections,
    state: LiveRuntimeState,
    dji_overlay_bgra,
):
    want_track_overlay = bool(state.tracking_enabled and state.draw_track_ids)

    if state.draw_detections and state.people_on:
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

    if state.draw_detections and state.fire_on:
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

    if state.draw_detections and want_track_overlay:
        frame = draw_tracked_boxes(
            frame,
            udp_ready_detections,
            colors=S.COLORS,
            default_color=(255, 255, 255),
            text_scale=S.MASK_TEXT_SCALE,
            text_thickness=S.MASK_TEXT_THICKNESS,
            box_thickness=2,
        )

    if state.dji_overlay_on and dji_overlay_bgra is not None:
        frame = apply_rgba_overlay_fullframe(frame, dji_overlay_bgra)

    return frame


def run_live(args) -> int:
    _print_device_info()

    people_seg_model, fire_model, people_seg_label, fire_label, _, detect_class_ids = build_models()
    id_flicker_mitigator = make_id_flicker_mitigator()

    state = LiveRuntimeState(
        people_on=bool(S.PEOPLE_ON_DEFAULT),
        fire_on=bool(S.FIRE_ON_DEFAULT),
        recording_enabled=bool(S.RECORDING_ENABLED_DEFAULT),
        draw_detections=bool(S.DRAW_DETECTIONS_DEFAULT),
        tracking_enabled=TRACKING_ENABLED,
        draw_track_ids=bool(getattr(S, "DRAW_TRACK_IDS", True)),
        dji_overlay_on=bool(S.DJI_MENU_OVERLAY_ENABLED_DEFAULT),
        active_camera_source=S.CAMERA_SOURCE_DEFAULT,
    )

    dji_overlay_bgra = load_rgba_overlay(S.DJI_MENU_OVERLAY_PATH)
    fire_class_names = {str(v).lower() for v in fire_label.names.values()}

    fps_hist = deque(maxlen=30)
    drop_hist = deque(maxlen=30)
    t_prev = time.time()

    cap, is_file_source, target_fps, video_start_wall, _input_desc = open_capture(
        S.INPUT_MODE,
        state.active_camera_source,
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

            frame = format_frame(frame)
            infer_frame = frame

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

            people_results, fire_results, use_ultra_track = _run_model_inference(
                infer_frame=infer_frame,
                state=state,
                people_seg_model=people_seg_model,
                fire_model=fire_model,
                detect_class_ids=detect_class_ids,
                pred_kw=pred_kw,
            )

            merged = merge_detections(
                people_results,
                fire_results,
                people_model=people_seg_label,
                fire_model=fire_label,
            )
            _normalize_merged_detections(
                merged,
                use_ultra_track=use_ultra_track,
                fire_class_names=fire_class_names,
            )

            height, width = frame.shape[:2]
            _attach_detection_footpoints(merged, width=width, height=height)

            if state.tracking_enabled:
                udp_ready_detections = id_flicker_mitigator.apply(merged)
            else:
                udp_ready_detections = list(merged)

            frame = _render_runtime_frame(
                frame=frame,
                people_results=people_results,
                fire_results=fire_results,
                people_seg_label=people_seg_label,
                fire_label=fire_label,
                udp_ready_detections=udp_ready_detections,
                state=state,
                dji_overlay_bgra=dji_overlay_bgra,
            )

            if S.SAVE_OUTPUT and state.recording_enabled:
                if out is None:
                    out_height, out_width = frame.shape[:2]
                    out = cv2.VideoWriter(
                        S.OUTPUT_VIDEO, fourcc, target_fps, (out_width, out_height)
                    )
                out.write(frame)

            if udp is not None:
                out_height, out_width = frame.shape[:2]
                packet = to_unity_udp_packet(
                    udp_ready_detections,
                    frame_id=frame_id,
                    timestamp=now,
                    width=out_width,
                    height=out_height,
                    class_map=S.UNITY_CLASS_ID,
                    allowed_classes=S.UDP_SEND_CLASSES,
                    min_conf=S.UDP_MIN_CONF,
                )
                try:
                    udp.send_json(packet)
                except Exception:
                    pass

            if not args.no_gui:
                cv2.imshow(S.WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255

            control_result = handle_runtime_key(
                key=key,
                state=state,
                id_flicker_mitigator=id_flicker_mitigator,
                cap=cap,
                is_file_source=is_file_source,
                target_fps=target_fps,
                video_start_wall=video_start_wall,
                fps_hist=fps_hist,
                drop_hist=drop_hist,
                t_prev=t_prev,
                window_frames=window_frames,
                window_start=window_start,
            )
            if control_result["should_exit"]:
                break

            cap = control_result["cap"]
            is_file_source = control_result["is_file_source"]
            target_fps = control_result["target_fps"]
            video_start_wall = control_result["video_start_wall"]
            t_prev = control_result["t_prev"]
            window_frames = control_result["window_frames"]
            window_start = control_result["window_start"]

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
