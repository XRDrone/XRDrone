# main.py
"""
Entry point for the XRDrone local YOLO pipeline: runs inference from
a video source, renders HUD/overlays, and logs merged detections to JSON.
"""

from ultralytics import YOLO
import cv2, time, numpy as np
from collections import deque
from hud import draw_hud, draw_boxes
from merger import merge_detections, count_by_class
from detection_logger import save_detections_json
from types import SimpleNamespace
import torch

# -----------------------------
# Debug: GPU / Torch info
# -----------------------------
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

# -----------------------------
# Settings
# -----------------------------
VIDEO_PATH = r"E:\Detection_Segmentation_Demo.mp4"

SAVE_OUTPUT = True
OUTPUT_VIDEO = "Segmentation_Aeroscapes.mp4"

# If saving MP4 is slow on your system, switch to MJPG AVI (much faster):
# OUTPUT_VIDEO = "Segmentation_Base.avi"
# OUTPUT_CODEC = "MJPG"
OUTPUT_CODEC = "mp4v"

SHOW_BOXES = False      # toggle with 'O'
SEG_ON = True           # toggle with 'P'

# Speed knobs
DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_FP16 = bool(torch.cuda.is_available())   # half precision on CUDA
IMGSZ = 960                                  # lower is faster

# Privacy & Recording
RECORDING_ENABLED = True  # toggle with 'R'
recording_start_time = None

# Model toggles
PEOPLE_ON = True  # toggle with 'K'
FIRE_ON = False   # toggle with 'L'

# -----------------------------
# Models
# -----------------------------
people_det_model = YOLO("../yolo11_models/yolo11n.pt")
people_seg_model = YOLO("../yolo11_models/aeroscapes_human_seg_yolo11s_960.pt")
fire_model = YOLO("../yolo11_models/fire_smoke_detection.pt")

# -----------------------------
# Name remap (YOLO.names is read-only on YOLO wrapper)
# Use lightweight wrappers with corrected .names passed into draw/merge.
# -----------------------------
def _normalize_names(names):
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}

def _remap_people_names(names_dict):
    """
    If model is single-class (often custom), force that class to be "person".
    Else if "person" missing but "item" present, rename "item" -> "person".
    """
    names = dict(names_dict)
    inv = {v.lower(): k for k, v in names.items()}

    if len(names) == 1:
        return {0: "person"}

    if "person" not in inv and "item" in inv:
        names[int(inv["item"])] = "person"

    return names

def _find_class_idx(names_dict, want):
    want = want.lower()
    for k, v in names_dict.items():
        if str(v).lower() == want:
            return int(k)
    return None

# Build label-wrappers
det_names = _normalize_names(people_det_model.names)
seg_names = _normalize_names(people_seg_model.names)
fire_names = _normalize_names(fire_model.names)

people_det_label = SimpleNamespace(names=_remap_people_names(det_names))
people_seg_label = SimpleNamespace(names=_remap_people_names(seg_names))
fire_label = SimpleNamespace(names=fire_names)

# Determine correct person class index for filtering
PERSON_CLASS = _find_class_idx(people_det_label.names, "person")
if PERSON_CLASS is None:
    PERSON_CLASS = 0

# -----------------------------
# Speed optimizations
# -----------------------------
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

for m in (people_det_model, people_seg_model, fire_model):
    try:
        m.to(DEVICE)
    except Exception:
        pass
    try:
        m.fuse()
    except Exception:
        pass

PRED_KW = dict(device=DEVICE, half=USE_FP16, imgsz=IMGSZ, verbose=False)

# -----------------------------
# Logging / HUD state
# -----------------------------
DETECTION_LOG_PATH = "detections_log.json"
all_detections = []

colors = {
    "person": (255, 0, 0),    # Blue
    "item": (255, 0, 0),      # fallback
    "fire": (255, 0, 255),    # Purple
    "smoke": (0, 255, 255),   # Yellow
}

fps_hist = deque(maxlen=30)
inf_hist = deque(maxlen=30)
drop_hist = deque(maxlen=30)
t_prev = time.time()

# -----------------------------
# Video IO
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

input_fps = cap.get(cv2.CAP_PROP_FPS)
if not input_fps or input_fps <= 1:
    input_fps = 30.0
TARGET_FPS = float(input_fps)
EXPECTED_FRAME_TIME = 1.0 / TARGET_FPS

fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
out = None

video_start_wall = time.time()

frame_counter = 0
window_frames = 0
window_start = time.time()
frame_id = 0

# -----------------------------
# Mask drawing (WITH LABELS)
# -----------------------------
def draw_masks(
    frame,
    results,
    names=None,                 # pass people_label_used.names here
    color=(0, 255, 0),
    alpha=0.35,
    text_scale=0.6,
    text_thickness=2,
):
    """
    Overlay instance masks with alpha blending AND draw a label ("person 0.xx")
    for each instance using the corresponding box confidence.
    """
    if not results:
        return frame

    r0 = results[0]
    if r0.masks is None or r0.masks.data is None or len(r0.masks.data) == 0:
        return frame

    masks = r0.masks.data  # (n, h_mask, w_mask) torch tensor
    boxes = getattr(r0, "boxes", None)

    H, W = frame.shape[:2]
    overlay = np.zeros_like(frame, dtype=np.uint8)
    overlay[:] = color

    # Normalize names to dict[int,str]
    if names is None:
        names = {}
    elif isinstance(names, (list, tuple)):
        names = {i: str(v) for i, v in enumerate(names)}
    else:
        names = {int(k): str(v) for k, v in names.items()}

    n_masks = len(masks)
    n_boxes = len(boxes) if boxes is not None else 0
    n = min(n_masks, n_boxes) if n_boxes > 0 else n_masks

    # Pre-pull box tensors to CPU if present
    if boxes is not None and n_boxes > 0:
        xyxy = boxes.xyxy[:n].detach().cpu().numpy()
        confs = boxes.conf[:n].detach().cpu().numpy()
        clss = boxes.cls[:n].detach().cpu().numpy().astype(int)
    else:
        xyxy = None
        confs = None
        clss = None

    for i in range(n):
        m = masks[i].detach().cpu().numpy().astype(np.uint8)  # (h_mask, w_mask) in {0,1}
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = m.astype(bool)

        if mask.any():
            frame[mask] = (frame[mask] * (1 - alpha) + overlay[mask] * alpha).astype(np.uint8)

        # Draw label using bbox if available
        if xyxy is not None:
            x1, y1, x2, y2 = xyxy[i]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            cls_id = int(clss[i]) if clss is not None else 0
            conf = float(confs[i]) if confs is not None else 0.0

            name = names.get(cls_id, str(cls_id))
            if name.lower() == "item":
                name = "person"

            label = f"{name} {conf:.2f}"

            # label position (top-left of bbox)
            tx = max(0, x1)
            ty = max(0, y1 - 8)

            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
            )

            # Background box for readability
            bx1 = tx
            by1 = max(0, ty - th - baseline)
            bx2 = min(W - 1, tx + tw + 6)
            by2 = min(H - 1, ty + 6)

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

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    # Stable timestamp tied to video position
    t_video = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    now = video_start_wall + t_video

    # Processing FPS
    wall_now = time.time()
    dt = wall_now - t_prev
    t_prev = wall_now
    if dt > 0:
        fps_hist.append(1.0 / dt)

    # dropped-frame estimate vs target fps
    frame_counter += 1
    window_frames += 1
    if wall_now - window_start >= 1.0:
        window_len = wall_now - window_start
        expected_frames = TARGET_FPS * window_len
        drops = max(0, int(round(expected_frames - window_frames)))
        drop_hist.append(drops)
        window_frames = 0
        window_start = wall_now

    # --- Inference ---
    people_results = []
    fire_results = []

    effective_seg_on = SEG_ON and PEOPLE_ON

    if PEOPLE_ON:
        if effective_seg_on:
            people_results = people_seg_model.predict(
                frame, conf=0.4, classes=[PERSON_CLASS], **PRED_KW
            )
            people_label_used = people_seg_label
        else:
            people_results = people_det_model.predict(
                frame, conf=0.4, classes=[PERSON_CLASS], **PRED_KW
            )
            people_label_used = people_det_label
    else:
        people_label_used = people_det_label  # unused but defined

    if FIRE_ON:
        fire_results = fire_model.predict(frame, conf=0.25, **PRED_KW)

    # --- Merge detections ---
    merged = merge_detections(
        people_results,
        fire_results,
        people_model=people_label_used,  # wrapper with corrected names
        fire_model=fire_label,
        seg_on=effective_seg_on,
        timestamp=now,
    )

    for det in merged:
        det["frame_id"] = frame_id
        if det.get("class") == "item":
            det["class"] = "person"
        if det.get("class_name") == "item":
            det["class_name"] = "person"

    if merged:
        all_detections.extend(merged)

    counts = count_by_class(merged)

    # --- Draw ---
    if SHOW_BOXES:
        if PEOPLE_ON:
            frame = draw_boxes(frame, people_results, colors, people_label_used)
        if FIRE_ON:
            frame = draw_boxes(frame, fire_results, colors, fire_label)

    if effective_seg_on:
        frame = draw_masks(
            frame,
            people_results,
            names=people_label_used.names,     # <-- IMPORTANT: label from correct mapping
            color=colors["person"],
            alpha=0.35,
        )

    # --- timings / stats ---
    inf_times = []
    if PEOPLE_ON and people_results:
        inf_times.append(people_results[0].speed.get("inference", 0.0))
    if FIRE_ON and fire_results:
        inf_times.append(fire_results[0].speed.get("inference", 0.0))
    if inf_times:
        inf_hist.append(sum(inf_times) / len(inf_times))

    avg_fps = sum(fps_hist) / len(fps_hist) if fps_hist else 0
    avg_inf = sum(inf_hist) / len(inf_hist) if inf_hist else 0
    avg_drops = sum(drop_hist) / len(drop_hist) if drop_hist else 0

    people_count = counts.get("person", 0) + counts.get("item", 0)
    fire_count = counts.get("fire", 0)
    smoke_count = counts.get("smoke", 0)

    lines = [
        f"FPS: {avg_fps:5.2f} (proc) | Target: {TARGET_FPS:.1f}",
        f"Model inference: {avg_inf:5.1f} ms",
        f"People: {people_count}",
        f"Fire: {fire_count}",
        f"Smoke: {smoke_count}",
        f"Dropped frames (avg/s): {avg_drops:.1f}",
        f"RECORDING: {'ON' if RECORDING_ENABLED else 'OFF'} (R)",
        f"People model: {'ON' if PEOPLE_ON else 'OFF'} (K)",
        f"Fire model: {'ON' if FIRE_ON else 'OFF'} (L)",
        f"Boxes: {'ON' if SHOW_BOXES else 'OFF'} (O)",
        f"Seg setting: {'ON' if SEG_ON else 'OFF'} (P)",
        f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'} | FP16: {USE_FP16} | imgsz={IMGSZ}",
    ]
    frame = draw_hud(frame, lines, anchor="tl")

    # --- Output ---
    if SAVE_OUTPUT and out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, TARGET_FPS, (w, h))
    if SAVE_OUTPUT and out is not None:
        out.write(frame)

    cv2.imshow("Live Combined Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key in (ord("r"), ord("R")):
        if not RECORDING_ENABLED:
            print("[PII] USER_CONSENT: Recording ENABLED by user at", time.strftime("%Y-%m-%d %H:%M:%S"))
            RECORDING_ENABLED = True
            recording_start_time = time.time()
        else:
            print("[PII] USER_CONSENT: Recording DISABLED at", time.strftime("%Y-%m-%d %H:%M:%S"))
            RECORDING_ENABLED = False
            recording_start_time = None
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
    print(f"Combined detection video saved to: {OUTPUT_VIDEO}")

if RECORDING_ENABLED and recording_start_time:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"[PII] Writing recording file: recording_{timestamp}.mp4")

if all_detections:
    save_detections_json(all_detections, DETECTION_LOG_PATH)
