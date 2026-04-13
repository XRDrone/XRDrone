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
from orbslam_fusion import (
    OrbSlamPoseReceiver,
    attach_foot_and_world_from_orbslam,
    build_failure_overlay_lines,
    build_fusion_status,
    build_pose_packet,
    build_slam_packet,
)
from output_formatter import to_unity_udp_packet
from overlay import apply_rgba_overlay_fullframe, load_rgba_overlay
from rendering import draw_masks, draw_pose_mode_status, draw_status_block, draw_tracked_boxes
from runtime_builders import (
    build_models,
    build_pose_estimator,
    make_adaptive_runtime_tuner,
    make_id_flicker_mitigator,
    make_pose_motion_smoother,
    make_world_motion_smoother,
)
from runtime_controls import LiveRuntimeState, format_adaptive_metrics, handle_runtime_key
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


def _apply_adaptive_tuning(
    *,
    adaptive_tuner,
    pose_data: dict,
    merged: list[dict],
    udp_ready_detections: list[dict],
    fps_hist,
    drop_hist,
    state: LiveRuntimeState,
    pose_smoother,
    world_smoother,
    id_flicker_mitigator,
) -> None:
    if adaptive_tuner is None:
        return

    adaptive_tuner.record_frame(
        pose_data=pose_data,
        raw_detections=merged,
        udp_ready_detections=udp_ready_detections,
    )
    avg_fps = float(sum(fps_hist) / len(fps_hist)) if fps_hist else 0.0
    avg_drops = float(sum(drop_hist) / len(drop_hist)) if drop_hist else 0.0
    tuning_update = adaptive_tuner.propose_adjustment(
        current_motion_smoothing=state.motion_smoothing_value,
        current_tau_on=id_flicker_mitigator.tau_on,
        current_tau_off=id_flicker_mitigator.tau_off,
        current_coast_frames=id_flicker_mitigator.coast_frames,
        avg_fps=avg_fps,
        avg_drop_frames=avg_drops,
    )
    if tuning_update is None or not bool(tuning_update.get("changed", False)):
        return

    from runtime_controls import update_motion_smoothing_value

    state.motion_smoothing_value = update_motion_smoothing_value(
        pose_smoother,
        world_smoother,
        float(tuning_update["motion_smoothing"]),
    )
    id_flicker_mitigator.set_runtime_policy(
        tau_on=float(tuning_update["tau_on"]),
        tau_off=float(tuning_update["tau_off"]),
        coast_frames=int(tuning_update["coast_frames"]),
    )
    if bool(getattr(S, "ADAPTIVE_TUNING_LOG_UPDATES", True)):
        metrics_text = format_adaptive_metrics(tuning_update["metrics"])
        print(
            "Adaptive tuning [{mode}] -> smooth={smooth:.2f}, tau_on={tau_on:.2f}, "
            "tau_off={tau_off:.2f}, coast={coast} | {metrics}".format(
                mode=str(tuning_update.get("mode", "?")),
                smooth=state.motion_smoothing_value,
                tau_on=float(id_flicker_mitigator.tau_on),
                tau_off=float(id_flicker_mitigator.tau_off),
                coast=int(id_flicker_mitigator.coast_frames),
                metrics=metrics_text,
            )
        )


def _render_runtime_frame(
    *,
    frame,
    people_results,
    fire_results,
    people_seg_label,
    fire_label,
    udp_ready_detections,
    state: LiveRuntimeState,
    pose_estimator,
    dji_overlay_bgra,
    fusion_status: dict | None = None,
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

    if state.pose_mode_overlay_on and pose_estimator is not None:
        frame = draw_pose_mode_status(
            frame,
            pose_estimator.get_pose_mode_overlay_text(),
            enabled=state.pose_mode_overlay_on,
            origin=getattr(S, "POSE_MODE_OVERLAY_ORIGIN", (20, 40)),
            text_scale=float(getattr(S, "POSE_MODE_OVERLAY_TEXT_SCALE", 0.9)),
            text_thickness=int(getattr(S, "POSE_MODE_OVERLAY_TEXT_THICKNESS", 2)),
        )

    if fusion_status is not None:
        frame = draw_status_block(
            frame,
            build_failure_overlay_lines(fusion_status),
            enabled=bool(getattr(S, "ORBSLAM_STATUS_OVERLAY_ENABLED", True)),
            origin=getattr(S, "ORBSLAM_STATUS_OVERLAY_ORIGIN", (20, 72)),
            text_scale=float(getattr(S, "ORBSLAM_STATUS_OVERLAY_TEXT_SCALE", 0.65)),
            text_thickness=int(getattr(S, "ORBSLAM_STATUS_OVERLAY_TEXT_THICKNESS", 2)),
        )

    return frame


def run_live(args) -> int:
    _print_device_info()

    people_seg_model, fire_model, people_seg_label, fire_label, _, detect_class_ids = build_models()
    use_orbslam_fusion = bool(getattr(S, "ORBSLAM_FUSION_ENABLED", False))
    pose_estimator = None if use_orbslam_fusion else build_pose_estimator()
    pose_draw = bool(getattr(S, "POSE_DRAW_ARUCO", False)) and not use_orbslam_fusion
    pose_smoother = None if use_orbslam_fusion else make_pose_motion_smoother()
    world_smoother = make_world_motion_smoother()
    id_flicker_mitigator = make_id_flicker_mitigator()
    adaptive_tuner = make_adaptive_runtime_tuner()
    orbslam_receiver = (
        OrbSlamPoseReceiver(
            getattr(S, "ORBSLAM_UDP_LISTEN_IP", "127.0.0.1"),
            int(getattr(S, "ORBSLAM_UDP_PORT", 5010)),
            max_entries=int(getattr(S, "ORBSLAM_POSE_BUFFER_SIZE", 4096)),
            stale_timeout_s=float(getattr(S, "ORBSLAM_PACKET_STALE_TIMEOUT_S", 0.50)),
            max_packet_bytes=int(getattr(S, "ORBSLAM_UDP_MAX_PACKET_BYTES", 65535)),
        )
        if use_orbslam_fusion
        else None
    )

    state = LiveRuntimeState(
        people_on=bool(S.PEOPLE_ON_DEFAULT),
        fire_on=bool(S.FIRE_ON_DEFAULT),
        recording_enabled=bool(S.RECORDING_ENABLED_DEFAULT),
        draw_detections=bool(S.DRAW_DETECTIONS_DEFAULT),
        tracking_enabled=TRACKING_ENABLED,
        draw_track_ids=bool(getattr(S, "DRAW_TRACK_IDS", True)),
        pose_mode_overlay_on=bool(getattr(S, "POSE_MODE_OVERLAY_ENABLED_DEFAULT", True)),
        dji_overlay_on=bool(S.DJI_MENU_OVERLAY_ENABLED_DEFAULT),
        active_camera_source=S.CAMERA_SOURCE_DEFAULT,
        motion_smoothing_value=float(getattr(S, "MOTION_SMOOTHING", 0.0)),
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

            infer_frame = frame.copy() if pose_draw else frame
            if use_orbslam_fusion:
                pose_solution = None
                pose_data = build_pose_packet(None, hfov_deg=float(S.POSE_HFOV_DEG))
                slam_packet = build_slam_packet(None, tracking_state="missing", match_mode="none")
                fusion_status = build_fusion_status(
                    source="orbslam",
                    pose=None,
                    match_mode="none",
                    receiver_error=orbslam_receiver.last_error
                    if orbslam_receiver is not None
                    else None,
                    projection_attempted=0,
                    projection_projected=0,
                )
            else:
                pose_data, pose_solution = pose_estimator.estimate_with_solution(
                    frame, draw=pose_draw
                )
                pose_data, pose_solution = pose_smoother.smooth(
                    pose_data, pose_solution, timestamp=now
                )
                slam_packet = {
                    "tracking_state": "disabled",
                    "match_mode": "none",
                    "pose_valid": False,
                    "frame_id": None,
                    "timestamp": None,
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                }
                fusion_status = {
                    "source": "aruco",
                    "slam_tracking": "disabled",
                    "match_mode": "none",
                    "projection_state": "ok"
                    if bool(pose_data.get("pose_valid", False))
                    else "unavailable",
                    "pose_valid": bool(pose_data.get("pose_valid", False)),
                    "projection_attempted": 0,
                    "projection_projected": 0,
                    "reason": "",
                }

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
            if use_orbslam_fusion:
                matched_pose = None
                match_mode = "none"
                receiver_error = (
                    orbslam_receiver.last_error if orbslam_receiver is not None else None
                )
                if orbslam_receiver is not None:
                    matched_pose, match_mode = orbslam_receiver.match(
                        frame_id=frame_id,
                        timestamp=now,
                        time_tolerance_s=float(getattr(S, "ORBSLAM_MATCH_TIME_TOLERANCE_S", 0.10)),
                    )
                    receiver_error = orbslam_receiver.last_error
                projection_counts = attach_foot_and_world_from_orbslam(
                    merged,
                    pose=matched_pose,
                    width=width,
                    height=height,
                    hfov_deg=float(S.POSE_HFOV_DEG),
                    projection_classes=tuple(S.UDP_SEND_CLASSES),
                    projection_min_conf=float(S.UDP_MIN_CONF),
                    ground_plane_y=float(getattr(S, "ORBSLAM_GROUND_PLANE_Y", 0.0)),
                )
                pose_data = build_pose_packet(matched_pose, hfov_deg=float(S.POSE_HFOV_DEG))
                slam_packet = build_slam_packet(
                    matched_pose,
                    tracking_state=(
                        matched_pose.tracking_state
                        if matched_pose is not None
                        else (
                            "stale"
                            if receiver_error and "recent" in receiver_error.lower()
                            else "missing"
                        )
                    ),
                    match_mode=match_mode,
                )
                fusion_status = build_fusion_status(
                    source="orbslam",
                    pose=matched_pose,
                    match_mode=match_mode,
                    receiver_error=receiver_error,
                    projection_attempted=projection_counts["attempted"],
                    projection_projected=projection_counts["projected"],
                )
            else:
                from world_projection import attach_foot_and_world

                attach_foot_and_world(
                    merged,
                    pose_data=pose_data,
                    pose_solution=pose_solution,
                    width=width,
                    height=height,
                    projection_classes=S.UDP_SEND_CLASSES,
                    projection_min_conf=S.UDP_MIN_CONF,
                )

            if world_smoother is not None:
                world_smoother.update_inplace(merged, timestamp=now)

            if state.tracking_enabled:
                udp_ready_detections = id_flicker_mitigator.apply(merged)
            else:
                udp_ready_detections = list(merged)

            _apply_adaptive_tuning(
                adaptive_tuner=adaptive_tuner,
                pose_data=pose_data,
                merged=merged,
                udp_ready_detections=udp_ready_detections,
                fps_hist=fps_hist,
                drop_hist=drop_hist,
                state=state,
                pose_smoother=pose_smoother,
                world_smoother=world_smoother,
                id_flicker_mitigator=id_flicker_mitigator,
            )

            frame = _render_runtime_frame(
                frame=frame,
                people_results=people_results,
                fire_results=fire_results,
                people_seg_label=people_seg_label,
                fire_label=fire_label,
                udp_ready_detections=udp_ready_detections,
                state=state,
                pose_estimator=pose_estimator,
                dji_overlay_bgra=dji_overlay_bgra,
                fusion_status=fusion_status,
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
                packet["pose"] = pose_data
                packet["slam"] = slam_packet
                packet["fusion_status"] = fusion_status
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
                pose_smoother=pose_smoother,
                world_smoother=world_smoother,
                id_flicker_mitigator=id_flicker_mitigator,
                adaptive_tuner=adaptive_tuner,
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
