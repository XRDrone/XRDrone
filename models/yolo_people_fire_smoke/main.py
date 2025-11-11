from ultralytics import YOLO
import cv2, time, numpy as np
from collections import deque
from hud import draw_hud, draw_boxes

# ---- Settings ----
SAVE_OUTPUT = False
SHOW_BOXES = True       # toggle with 'O'
SEG_ON = False          # toggle with 'P'
TARGET_FPS = 30.0

# Models
people_det_model = YOLO("yolo12n.pt")     # people detection (boxes)
people_seg_model = YOLO("yolo11n-seg.pt")  # people instance segmentation (masks+boxes)
fire_model = YOLO("best.pt")               # fire/smoke detection (boxes)

colors = {
    'person': (255, 0, 0),   # Blue
    'fire':   (255, 0, 255), # Purple
    'smoke':  (0, 255, 255)  # Yellow
}

fps_hist = deque(maxlen=30)
inf_hist = deque(maxlen=30)
drop_hist = deque(maxlen=30)
t_prev = time.time()

output_video = "combined_detection.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None

cap = cv2.VideoCapture(0)

EXPECTED_FRAME_TIME = 1.0 / TARGET_FPS
frame_counter = 0
drop_counter = 0
window_start = time.time()

def draw_masks(frame, results, color=(0, 255, 0), alpha=0.35):
    """
    Overlay instance masks (Ultralytics seg results) with alpha blending.
    Resizes masks to match the current frame size.
    """
    if not results or results[0].masks is None:
        return frame

    r0 = results[0]
    masks = r0.masks.data  # (n, h_mask, w_mask) torch tensor
    if masks is None or len(masks) == 0:
        return frame

    H, W = frame.shape[:2]
    overlay = np.zeros_like(frame, dtype=np.uint8)
    overlay[:] = color

    # Convert and resize each mask to (H, W)
    for m in masks:
        m = m.detach().cpu().numpy().astype(np.uint8)         # (h_mask, w_mask) in {0,1}
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = m.astype(bool)

        if mask.any():
            frame[mask] = (frame[mask] * (1 - alpha) + overlay[mask] * alpha).astype(np.uint8)

    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    dt = now - t_prev
    t_prev = now
    frame_counter += 1

    if dt > EXPECTED_FRAME_TIME * 1.5:
        missed_frames = int(round(dt / EXPECTED_FRAME_TIME)) - 1
        drop_counter += max(0, missed_frames)

    if now - window_start >= 1.0:
        drop_hist.append(drop_counter)
        drop_counter = 0
        window_start = now

    if dt > 0:
        fps_hist.append(1.0 / dt)

    # --- Inference ---
    # People: use seg model when SEG_ON, else det model
    if SEG_ON:
        people_results = people_seg_model.predict(frame, conf=0.4, classes=[0], verbose=False)
    else:
        people_results = people_det_model.predict(frame, conf=0.4, classes=[0], verbose=False)

    fire_results = fire_model.predict(frame, conf=0.25, verbose=False)

    # --- Draw ---
    if SHOW_BOXES:
        frame = draw_boxes(frame, people_results, colors, people_det_model if not SEG_ON else people_seg_model)
        frame = draw_boxes(frame, fire_results, colors, fire_model)

    if SEG_ON:
        frame = draw_masks(frame, people_results, color=colors['person'], alpha=0.35)

    # timings / stats (use fire inference timing just like before)
    inf_time = fire_results[0].speed["inference"]
    inf_hist.append(inf_time)
    avg_fps = sum(fps_hist) / len(fps_hist)
    avg_inf = sum(inf_hist) / len(inf_hist)
    avg_drops = sum(drop_hist) / len(drop_hist) if drop_hist else 0

    # counts
    people_count = int((people_results[0].boxes.cls == 0).sum()) if people_results else 0
    fire_count, smoke_count = 0, 0
    for r in fire_results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = fire_model.names[cls_id].lower()
            if label == 'fire':  fire_count += 1
            elif label == 'smoke': smoke_count += 1

    lines = [
        f"FPS: {avg_fps:5.2f}",
        f"Model latency: {avg_inf:5.1f} ms",
        f"People: {people_count}",
        f"Fire: {fire_count}",
        f"Smoke: {smoke_count}",
        f"Dropped frames (avg/s): {avg_drops:.1f}",
        f"Boxes: {'ON' if SHOW_BOXES else 'OFF'} (O)",
        f"Seg: {'ON' if SEG_ON else 'OFF'} (P)"
    ]
    frame = draw_hud(frame, lines, anchor="tl")

    if SAVE_OUTPUT and out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter(output_video, fourcc, 24, (w, h))
    if SAVE_OUTPUT:
        out.write(frame)

    cv2.imshow("Live Combined Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key in (ord('o'), ord('O')):
        SHOW_BOXES = not SHOW_BOXES
    elif key in (ord('p'), ord('P')):
        SEG_ON = not SEG_ON

cap.release()
if SAVE_OUTPUT and out is not None:
    out.release()
cv2.destroyAllWindows()
if SAVE_OUTPUT:
    print(f"Combined detection video saved to: {output_video}")
