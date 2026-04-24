"""
test_runner.py

Single-image inference path for the XRDrone pipeline.
"""

from __future__ import annotations

import json
import time

import cv2
import settings as S
from frame_io import format_frame
from merger import merge_detections
from output_formatter import to_unity_udp_packet
from overlay import apply_rgba_overlay_fullframe, load_rgba_overlay
from rendering import draw_pose_mode_status
from runtime_builders import build_models, build_pose_estimator, make_pose_motion_smoother
from streaming import UDPPublisher
from world_projection import attach_foot_and_world


def run_test(args) -> int:
    people_seg_model, fire_model, people_seg_label, fire_label, _, detect_class_ids = build_models()

    image = cv2.imread(args.test_image)
    if image is None:
        raise RuntimeError(f"Could not read test image: {args.test_image}")

    frame = format_frame(image)
    now = time.time()
    frame_id = 1

    pose_estimator = build_pose_estimator()
    pose_draw = bool(getattr(S, "POSE_DRAW_ARUCO", False))
    pose_mode_overlay_on = bool(getattr(S, "POSE_MODE_OVERLAY_ENABLED_DEFAULT", True))
    pose_smoother = make_pose_motion_smoother()

    infer_frame = frame.copy() if pose_draw else frame
    pose_data, pose_solution = pose_estimator.estimate_with_solution(frame, draw=pose_draw)
    pose_data, pose_solution = pose_smoother.smooth(pose_data, pose_solution, timestamp=now)

    pred_kw = dict(device=S.DEVICE, half=S.USE_FP16, imgsz=S.IMGSZ, verbose=False)

    people_results = people_seg_model.predict(
        infer_frame,
        conf=S.PEOPLE_CONF,
        classes=detect_class_ids,
        **pred_kw,
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

    height, width = frame.shape[:2]
    attach_foot_and_world(
        merged,
        pose_data=pose_data,
        pose_solution=pose_solution,
        width=width,
        height=height,
        projection_classes=S.UDP_SEND_CLASSES,
        projection_min_conf=S.UDP_MIN_CONF,
    )

    packet = to_unity_udp_packet(
        merged,
        frame_id=frame_id,
        timestamp=now,
        width=width,
        height=height,
        class_map=S.UNITY_CLASS_ID,
        allowed_classes=S.UDP_SEND_CLASSES,
        min_conf=S.UDP_MIN_CONF,
    )
    packet["pose"] = pose_data

    print("[UDP] JSON payload (one-line):")
    print(json.dumps(packet))

    print("\n[UDP] JSON payload (pretty):")
    print(json.dumps(packet, indent=2))

    if S.ENABLE_UDP:
        udp = UDPPublisher(S.UDP_IP, S.UDP_PORT)
        try:
            udp.send_json(packet)
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
