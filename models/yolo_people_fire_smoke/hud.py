# hud.py
"""
hud.py

Rendering helpers for the XRDrone on-frame HUD and detection overlays.

This module:
  - draw_hud(): draws a semi-transparent HUD panel anchored to a corner
    of an OpenCV BGR frame, showing runtime stats such as FPS, latency,
    counts per class, and toggle states.
  - draw_boxes(): draws YOLO bounding boxes plus class labels and
    confidence percentages using a per-class color map and model.names.
  - load_rgba_overlay(): loads a PNG (RGBA) once.
  - apply_rgba_overlay_fullframe(): resizes overlay to frame size and alpha-blends it on top.

All operations modify the provided frame in place and return the updated
OpenCV image for real-time display or video encoding.
"""

import cv2
import numpy as np


def draw_hud(
    frame,
    lines,
    anchor="tl",
    margin=10,
    alpha=0.45,
    font_scale=0.55,
    thickness=1,
    position=None,  # NEW
):
    pad = 5
    line_h = 0
    width = 0

    for text in lines:
        (w, h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        width = max(width, w)
        line_h = max(line_h, h)

    box_w = width + pad * 2
    box_h = line_h * len(lines) + pad * (len(lines) + 1)

    H, W = frame.shape[:2]
    if position is not None:
        x1, y1 = position

    elif anchor == "tl":
        x1, y1 = margin, margin
    elif anchor == "tr":
        x1, y1 = W - margin - box_w, margin
    elif anchor == "bl":
        x1, y1 = margin, H - margin - box_h
    elif anchor == "br":
        x1, y1 = W - margin - box_w, H - margin - box_h
    elif anchor == "lb":
        # Left-bottom placement with fixed spacing from bottom
        x1 = 0
        y1 = H - margin - box_h
    else:
        x1, y1 = margin, margin

    # Clamp HUD inside frame bounds
    x1 = max(0, min(x1, W - box_w))
    y1 = max(0, min(y1, H - box_h))

    x2, y2 = x1 + box_w, y1 + box_h

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

    y = y1 + pad + line_h

    def draw_colored_segments(y, segments):
        """
        Draw multiple colored text segments on one HUD line.
        segments = [(string, color), ...]
        """
        x = x1 + pad
        for seg_text, seg_color in segments:
            cv2.putText(
                frame,
                seg_text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                seg_color,
                thickness,
                cv2.LINE_AA,
            )
            # Advance x by width of this segment
            (tw, _), _ = cv2.getTextSize(
                seg_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness,
            )
            x += tw

    for text in lines:
        # Only label+colon colored, number white
        if text.startswith("People:"):
            draw_colored_segments(
                y,
                [
                    ("People: ", (255, 255, 0)),  # Cyan
                    (text.split(":")[1].strip(), (255, 255, 255)),  # Count white
                ],
            )

        elif text.startswith("Fire:"):
            draw_colored_segments(
                y,
                [
                    ("Fire: ", (255, 0, 255)),  # Purple
                    (text.split(":")[1].strip(), (255, 255, 255)),  # Count white
                ],
            )

        elif text.startswith("Smoke:"):
            draw_colored_segments(
                y,
                [
                    ("Smoke: ", (0, 255, 255)),  # Yellow
                    (text.split(":")[1].strip(), (255, 255, 255)),  # Count white
                ],
            )

        # Only People cyan
        elif text.startswith("People model:"):
            # Color only "People" cyan, rest white
            rest = text[len("People"):]
            draw_colored_segments(
                y,
                [
                    ("People", (255, 255, 0)),  # Cyan
                    (rest, (255, 255, 255)),  # Everything else white
                ],
            )

        # Only Fire purple, Smoke yellow, rest white
        elif text.startswith("Fire/Smoke model:"):
            # Split into prefix and ON/OFF part
            prefix, state = text.split(":", 1)
            draw_colored_segments(
                y,
                [
                    ("Fire", (255, 0, 255)),  # Purple
                    ("/", (255, 255, 255)),  # White slash
                    ("Smoke", (0, 255, 255)),  # Yellow
                    (" model:", (255, 255, 255)),  # Rest white
                    (" " + state.strip(), (255, 255, 255)),  # OFF (L) white
                ],
            )

        # Default case: normal white text
        else:
            cv2.putText(
                frame,
                text,
                (x1 + pad, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        y += line_h + pad

    return frame


def draw_boxes(frame, results, colors, model):
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            label = model.names[cls_id].lower()
            color = colors.get(label, (255, 255, 255))

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Label text with confidence
            text = f"{label} {conf * 100:.1f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

            text_bg_x2 = min(x1 + tw + 8, x2)
            text_bg_y2 = min(y1 + th + 8, y2)
            cv2.rectangle(frame, (x1, y1), (text_bg_x2, text_bg_y2), color, -1)

            cv2.putText(
                frame,
                text,
                (x1 + 3, y1 + th),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    return frame


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

    # Ensure BGRA (4 channels). If RGB/BGR only, add fully-opaque alpha.
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        bgr = img
        alpha = np.full((bgr.shape[0], bgr.shape[1], 1), 255, dtype=bgr.dtype)
        img = np.concatenate([bgr, alpha], axis=2)
    elif img.shape[2] != 4:
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