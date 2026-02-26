"""
overlay.py

Utility for loading and applying RGBA overlays on top of video frames.

Supports:
  - Loading PNG or RGBA assets from disk
  - Converting grayscale/BGR images into BGRA format
  - Alpha blending overlays over full video frames
  - Resizing overlays to match frame resolution

Used for:
  - DJI-style UI overlays
  - Demo HUD visuals
  - Presentation layer enhancements
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["load_rgba_overlay", "apply_rgba_overlay_fullframe"]


def load_rgba_overlay(path: str):
    """
    Loads an image from disk as BGRA (OpenCV order) if possible.
    Returns None if load fails.
    """
    if not path:
        return None

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Ensure BGRA (4 channels). If grayscale or BGR, convert/append alpha.
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.ndim == 3 and img.shape[2] == 3:
        bgr = img
        alpha = np.full((bgr.shape[0], bgr.shape[1], 1), 255, dtype=bgr.dtype)
        img = np.concatenate([bgr, alpha], axis=2)
    elif img.ndim == 3 and img.shape[2] == 4:
        pass
    else:
        return None

    return img


def apply_rgba_overlay_fullframe(frame_bgr, overlay_bgra):
    """
    Alpha-blend overlay (BGRA) on top of frame (BGR).
    Overlay is resized to exactly match the frame dimensions.
    """
    if frame_bgr is None or overlay_bgra is None:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return frame_bgr

    # Resize overlay to match frame
    ov = cv2.resize(overlay_bgra, (w, h), interpolation=cv2.INTER_LINEAR)

    # Split channels
    ov_bgr = ov[:, :, :3].astype(np.float32)
    alpha = ov[:, :, 3].astype(np.float32) / 255.0  # (h,w) in [0,1]
    alpha = alpha[:, :, None]  # (h,w,1)

    base = frame_bgr.astype(np.float32)
    out = base * (1.0 - alpha) + ov_bgr * alpha

    frame_bgr[:] = out.astype(np.uint8)
    return frame_bgr