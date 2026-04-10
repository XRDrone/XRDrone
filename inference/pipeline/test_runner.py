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
from runtime_builders import build_models
from streaming import UDPPublisher


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


def run_test(args) -> int:
    people_seg_model, fire_model, people_seg_label, fire_label, _, detect_class_ids = build_models()

    image = cv2.imread(args.test_image)
    if image is None:
        raise RuntimeError(f"Could not read test image: {args.test_image}")

    frame = format_frame(image)
    now = time.time()
    frame_id = 1

    pred_kw = dict(device=S.DEVICE, half=S.USE_FP16, imgsz=S.IMGSZ, verbose=False)

    people_results = people_seg_model.predict(
        frame,
        conf=S.PEOPLE_CONF,
        classes=detect_class_ids,
        **pred_kw,
    )

    fire_results = []
    if S.FIRE_ON_DEFAULT:
        fire_results = fire_model.predict(frame, conf=S.FIRE_CONF, **pred_kw)

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
    _attach_detection_footpoints(merged, width=width, height=height)

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

        cv2.imshow(S.WINDOW_NAME, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0
