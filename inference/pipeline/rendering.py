"""
rendering.py

Local visualization helpers for masks, track overlays, and pose-mode text.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


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
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        float(text_scale),
        int(text_thickness),
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

    result = results[0]
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)

    if boxes is None or len(boxes) == 0:
        return frame

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy()

    mask_data = None
    if masks is not None and getattr(masks, "data", None) is not None:
        mask_data = masks.data.detach().cpu().numpy()

    n = len(xyxy)
    if mask_data is not None:
        n = min(n, mask_data.shape[0])

    h_img, w_img = frame.shape[:2]

    for i in range(n):
        cls_id = int(cls[i]) if i < len(cls) else 0
        name = str(names.get(cls_id, "obj")).lower()
        color = colors.get(name, default_color)
        color_arr = np.array(color, dtype=np.float32)

        x1, y1, _x2, _y2 = xyxy[i]
        x1i, y1i = int(x1), int(y1)

        if mask_data is not None:
            mask = np.squeeze(mask_data[i])
            if mask.ndim == 2:
                if mask.shape[:2] != (h_img, w_img):
                    mask = cv2.resize(
                        mask.astype(np.float32),
                        (w_img, h_img),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask = mask > 0.5
                frame[mask] = (
                    frame[mask].astype(np.float32) * (1.0 - alpha) + color_arr * alpha
                ).astype(np.uint8)

        if show_label:
            label = f"{name} {float(conf[i]) * 100:.1f}%"
            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                text_thickness,
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
    colors: dict[str, tuple],
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
        conf = float(det.get("udp_confidence", det.get("confidence", 0.0)))
        track_id = det.get("track_id", None)
        continuity_state = str(det.get("continuity_state", "")).lower()

        if track_id is None:
            label = f"{cls_name} {conf * 100:.1f}%"
        else:
            try:
                label = f"{cls_name} #{int(track_id)} {conf * 100:.1f}%"
            except Exception:
                label = f"{cls_name} {conf * 100:.1f}%"

        if continuity_state == "coasted":
            label += " [hold]"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, int(box_thickness))

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            float(text_scale),
            int(text_thickness),
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
