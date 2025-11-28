from ultralytics import YOLO
import cv2, time, numpy as np
from collections import deque
from hud import draw_hud, draw_boxes
from merger import merge_detections, count_by_class
import pprint
from detection_log_loader import save_detections_json

# ---- Settings ----
SAVE_OUTPUT = False
SHOW_BOXES = True  # toggle with 'O'
SEG_ON = False  # toggle with 'P' (desired seg setting)
TARGET_FPS = 30.0

# ---- Model toggles ----
PEOPLE_ON = True  # toggle with 'K'
FIRE_ON = True  # toggle with 'L'

# Models
people_det_model = YOLO("../yolo11_models/yolo11n.pt")  # people detection (boxes)
people_seg_model = YOLO(
    "../yolo11_models/yolo11n-seg.pt"
)  # people instance segmentation (masks+boxes)
fire_model = YOLO(
    "../yolo11_models/fire_smoke_detection.pt"
)  # fire/smoke detection (boxes)

DETECTION_LOG_PATH = "detections_log.json"
all_detections = []

colors = {
    "person": (255, 0, 0),  # Blue
    "fire": (255, 0, 255),  # Purple
    "smoke": (0, 255, 255),  # Yellow
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
window_frames = 0
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
        m = m.detach().cpu().numpy().astype(np.uint8)  # (h_mask, w_mask) in {0,1}
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = m.astype(bool)

        if mask.any():
            frame[mask] = (frame[mask] * (1 - alpha) + overlay[mask] * alpha).astype(
                np.uint8
            )

    return frame


frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    now = time.time()
    dt = now - t_prev
    t_prev = now
    frame_counter += 1
    window_frames += 1

    if now - window_start >= 1.0:
        window_len = now - window_start
        expected_frames = TARGET_FPS * window_len
        drops = max(0, int(round(expected_frames - window_frames)))

        drop_hist.append(drops)
        window_frames = 0
        window_start = now

    if dt > 0:
        fps_hist.append(1.0 / dt)

    # --- Inference ---
    people_results = []
    fire_results = []
    people_model_used = people_det_model  # default for names / merger

    # NOTE: effective_seg_on is what actually runs.
    # SEG_ON can be toggled anytime (even if PEOPLE_ON is False),
    # and will take effect immediately when PEOPLE_ON becomes True.
    effective_seg_on = SEG_ON and PEOPLE_ON

    if PEOPLE_ON:
        # People: use seg model when effective_seg_on, else det model
        if effective_seg_on:
            people_results = people_seg_model.predict(
                frame, conf=0.4, classes=[0], verbose=False
            )
            people_model_used = people_seg_model
        else:
            people_results = people_det_model.predict(
                frame, conf=0.4, classes=[0], verbose=False
            )
            people_model_used = people_det_model

    if FIRE_ON:
        fire_results = fire_model.predict(frame, conf=0.25, verbose=False)

    # --- Merge detections ---
    merged = merge_detections(
        people_results,
        fire_results,
        people_model=people_model_used,
        fire_model=fire_model,
        seg_on=effective_seg_on,
        timestamp=now,
    )

    # Attach frame ID to each detection
    for det in merged:
        det["frame_id"] = frame_id

    if merged:
        all_detections.extend(merged)

    # --- Debug ---
    # if frame_counter == 5:
    #     print("\n=== MERGER OUTPUT ===")
    #     pprint.pprint(merged)
    #     print("=== END MERGER OUTPUT ===\n")
    #     input("Press ENTER to continue...")

    counts = count_by_class(merged)

    # --- Draw ---
    if SHOW_BOXES:
        if PEOPLE_ON:
            frame = draw_boxes(frame, people_results, colors, people_model_used)
        if FIRE_ON:
            frame = draw_boxes(frame, fire_results, colors, fire_model)

    if effective_seg_on:
        frame = draw_masks(frame, people_results, color=colors["person"], alpha=0.35)

    # --- timings / stats ---
    inf_times = []
    if PEOPLE_ON and people_results:
        inf_times.append(people_results[0].speed["inference"])
    if FIRE_ON and fire_results:
        inf_times.append(fire_results[0].speed["inference"])

    if inf_times:
        inf_hist.append(sum(inf_times) / len(inf_times))

    avg_fps = sum(fps_hist) / len(fps_hist) if fps_hist else 0
    avg_inf = sum(inf_hist) / len(inf_hist) if inf_hist else 0
    avg_drops = sum(drop_hist) / len(drop_hist) if drop_hist else 0

    # counts (from merged list)
    people_count = counts.get("person", 0)
    fire_count = counts.get("fire", 0)
    smoke_count = counts.get("smoke", 0)

    lines = [
        f"FPS: {avg_fps:5.2f}",
        f"Model latency: {avg_inf:5.1f} ms",
        f"People: {people_count}",
        f"Fire: {fire_count}",
        f"Smoke: {smoke_count}",
        f"Dropped frames (avg/s): {avg_drops:.1f}",
        f"People model: {'ON' if PEOPLE_ON else 'OFF'} (K)",
        f"Fire model: {'ON' if FIRE_ON else 'OFF'} (L)",
        f"Boxes: {'ON' if SHOW_BOXES else 'OFF'} (O)",
        # Show desired SEG_ON state even if People model is OFF
        f"Seg setting: {'ON' if SEG_ON else 'OFF'} (P)",
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
    elif key in (ord("o"), ord("O")):
        SHOW_BOXES = not SHOW_BOXES
    elif key in (ord("p"), ord("P")):
        SEG_ON = not SEG_ON
    elif key in (ord("k"), ord("K")):
        PEOPLE_ON = not PEOPLE_ON
    elif key in (ord("l"), ord("L")):
        FIRE_ON = not FIRE_ON

cap.release()
if SAVE_OUTPUT and out is not None:
    out.release()
cv2.destroyAllWindows()
if SAVE_OUTPUT:
    print(f"Combined detection video saved to: {output_video}")

# --- Save detection log as JSON with CUSTOM.updateTime-style timestamps ---
if all_detections:
    save_detections_json(all_detections, DETECTION_LOG_PATH)
