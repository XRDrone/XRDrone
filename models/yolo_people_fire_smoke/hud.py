import cv2


def draw_hud(
    frame, lines, anchor="tl", margin=10, alpha=0.45, font_scale=0.55, thickness=1
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
    if anchor == "tl":
        x1, y1 = margin, margin
    elif anchor == "tr":
        x1, y1 = W - margin - box_w, margin
    elif anchor == "bl":
        x1, y1 = margin, H - margin - box_h
    elif anchor == "br":
        x1, y1 = W - margin - box_w, H - margin - box_h
    else:
        x1, y1 = margin, margin
    x2, y2 = x1 + box_w, y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    y = y1 + pad + line_h
    for text in lines:
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
