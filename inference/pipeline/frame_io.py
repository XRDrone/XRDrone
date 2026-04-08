"""
frame_io.py

Capture and frame-formatting helpers for the XRDrone runtime.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import settings as S


def letterbox(frame_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
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


def format_frame(frame_bgr: np.ndarray) -> np.ndarray:
    if not S.FORCE_OUTPUT_1080P:
        return frame_bgr

    target_w = int(S.OUTPUT_WIDTH)
    target_h = int(S.OUTPUT_HEIGHT)

    if not S.OUTPUT_KEEP_ASPECT:
        return cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return letterbox(frame_bgr, target_w, target_h)


def cv_backend_flag(name: str) -> int:
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


def open_capture(input_mode: str, camera_source: str):
    """Return cap, is_file_source, target_fps, video_start_wall, input_desc."""
    input_mode = input_mode.lower().strip()

    if input_mode == "file":
        cap = cv2.VideoCapture(S.VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {S.VIDEO_PATH}")
        input_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        target_fps = input_fps if input_fps > 1 else S.DEFAULT_FPS
        return cap, True, target_fps, time.time(), f"file: {S.VIDEO_PATH}"

    camera_source = camera_source.lower().strip()
    if camera_source == "capture_card":
        index = int(S.CAPTURE_CARD_INDEX)
    else:
        index = int(S.WEBCAM_INDEX)

    backend_flag = cv_backend_flag(S.CAPTURE_BACKEND)
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
        f"camera: {camera_source} (index={index}, backend={backend_desc})",
    )
