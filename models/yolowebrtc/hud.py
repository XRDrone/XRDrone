import cv2

def draw_hud(frame, lines, anchor="tl", margin=10, alpha=0.5, font_scale=0.7, thickness=2):
    pad = 8
    line_h = 0
    width = 0
    for text in lines:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        width = max(width, w)
        line_h = max(line_h, h)
    box_w = width + pad * 2
    box_h = line_h * len(lines) + pad * (len(lines) + 1)

    H, W = frame.shape[:2]
    if anchor == "tl":  x1, y1 = margin, margin
    if anchor == "tr":  x1, y1 = W - margin - box_w, margin
    if anchor == "bl":  x1, y1 = margin, H - margin - box_h
    if anchor == "br":  x1, y1 = W - margin - box_w, H - margin - box_h
    x2, y2 = x1 + box_w, y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,0,0), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    y = y1 + pad + line_h
    for text in lines:
        cv2.putText(frame, text, (x1 + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255,255,255), thickness, cv2.LINE_AA)
        y += line_h + pad

    return frame
