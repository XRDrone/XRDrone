from ultralytics import YOLO
import cv2, time
from collections import deque
from hud import draw_hud, draw_boxes

people_model = YOLO("yolov8n.pt") # People detection
fire_model = YOLO("best.pt") # Fire and smoke detection

colors = {
    'person': (255, 0, 0), # Blue
    'fire': (255, 0, 255), # Purple
    'smoke': (0, 255, 255) # Yellow
}

fps_hist = deque(maxlen=30)
inf_hist = deque(maxlen=30)
drop_hist = deque(maxlen=30) # Stores average dropped frames per second
t_prev = time.time()

source_video = "original_short_demo.mp4"
output_video = "combined_detection.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None

cap = cv2.VideoCapture(source_video)

# Target FPS for drop estimation
TARGET_FPS = 30.0
EXPECTED_FRAME_TIME = 1.0 / TARGET_FPS
frame_counter = 0
drop_counter = 0
window_start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    dt = now - t_prev
    t_prev = now
    frame_counter += 1

    # Detect dropped frames
    if dt > EXPECTED_FRAME_TIME * 1.5:
        missed_frames = int(round(dt / EXPECTED_FRAME_TIME)) - 1
        drop_counter += missed_frames

    # Every 1 second, record drop average and reset counter
    if now - window_start >= 1.0:
        avg_dropped = drop_counter
        drop_hist.append(avg_dropped)
        drop_counter = 0
        window_start = now

    if dt > 0:
        fps_hist.append(1 / dt)

    # Run detections
    people_results = people_model.predict(frame, conf=0.4, classes=[0], verbose=False)
    fire_results = fire_model.predict(frame, conf=0.25, verbose=False)

    # Draw detections
    frame = draw_boxes(frame, people_results, colors, people_model)
    frame = draw_boxes(frame, fire_results, colors, fire_model)

    inf_time = fire_results[0].speed["inference"]
    inf_hist.append(inf_time)
    avg_fps = sum(fps_hist) / len(fps_hist)
    avg_inf = sum(inf_hist) / len(inf_hist)
    avg_drops = sum(drop_hist) / len(drop_hist) if drop_hist else 0

    # Count people, fire and smoke
    people_count = int((people_results[0].boxes.cls == 0).sum())

    fire_count, smoke_count = 0, 0
    for r in fire_results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = fire_model.names[cls_id].lower()
            if label == 'fire':
                fire_count += 1
            elif label == 'smoke':
                smoke_count += 1

    # HUD
    lines = [
        f"FPS: {avg_fps:5.2f}",
        f"Model latency: {avg_inf:5.1f} ms",
        f"People: {people_count}",
        f"Fire: {fire_count}",
        f"Smoke: {smoke_count}",
        f"Dropped frames (avg/s): {avg_drops:.1f}"
    ]
    frame = draw_hud(frame, lines, anchor="tl")

    if out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter(output_video, fourcc, 24, (w, h))

    out.write(frame)
    cv2.imshow("Live Combined Detection", frame)
    if cv2.waitKey(1) == 27: # Press ESC key to quit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Combined detection video saved to: {output_video}")
