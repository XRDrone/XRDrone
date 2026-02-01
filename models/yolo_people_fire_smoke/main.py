# main.py
from ultralytics import YOLO
import cv2, time, numpy as np
from collections import deque
from hud import draw_hud
from merger import merge_detections, count_by_class
from detection_logger import save_detections_json
from types import SimpleNamespace
import torch
import settings as S

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

PEOPLE_ON = S.PEOPLE_ON_DEFAULT
FIRE_ON = S.FIRE_ON_DEFAULT
RECORDING_ENABLED = S.RECORDING_ENABLED_DEFAULT
recording_start_time = None

people_seg_model = YOLO(S.PEOPLE_MODEL_PATH)
fire_model = YOLO(S.FIRE_MODEL_PATH)

def _normalize_names(names):
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}

def _remap_people_names(names_dict):
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

seg_names = _normalize_names(people_seg_model.names)
fire_names = _normalize_names(fire_model.names)

people_seg_label = SimpleNamespace(names=_remap_people_names(seg_names))
fire_label = SimpleNamespace(names=fire_names)

PERSON_CLASS = _find_class_idx(people_seg_label.names, "person")
if PERSON_CLASS is None:
    PERSON_CLASS = 0

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

for m in (people_seg_model, fire_model):
    try:
        m.to(S.DEVICE)
    except Exception:
        pass
    try:
        m.fuse()
    except Exception:
        pass

PRED_KW = dict(device=S.DEVICE, half=S.USE_FP16, imgsz=S.IMGSZ, verbose=False)

all_detections = []
fps_hist = deque(maxlen=30)
inf_hist = deque(maxlen=30)
drop_hist = deque(maxlen=30)
t_prev = time.time()

cap = cv2.VideoCapture(S.VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {S.VIDEO_SOURCE}")

input_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
TARGET_FPS = input_fps if input_fps > 1 else S.DEFAULT_FPS

fourcc = cv2.VideoWriter_fourcc(*S.OUTPUT_CODEC)
out = None

video_start_wall = time.time()
window_frames = 0
window_start = time.time()
frame_id = 0
is_file_source = isinstance(S.VIDEO_SOURCE, str)

def _attach_masks_to_merged(merged, results, source_key):
    if not results:
        return
    r0 = results[0]
    if r0.masks is None or r0.masks.data is None:
        return
    masks = r0.masks.data.detach().cpu().numpy()
    mi = 0
    for d in merged:
        if d.get("source") == source_key:
            if mi < len(masks):
                d["mask"] = masks[mi]
            mi += 1

def draw_masks(
    frame,
    results,
    names=None,
    color=(0, 255, 0),
    alpha=0.35,
    text_scale=0.6,
    text_thickness=2,
):
    if not results:
        return frame

    r0 = results[0]
    if r0.masks is None or r0.masks.data is None or len(r0.masks.data) == 0:
        return frame

    masks = r0.masks.data
    boxes = getattr(r0, "boxes", None)

    H, W = frame.shape[:2]
    overlay = np.zeros_like(frame, dtype=np.uint8)
    overlay[:] = color

    if names is None:
        names = {}
    elif isinstance(names, (list, tuple)):
        names = {i: str(v) for i, v in enumerate(names)}
    else:
        names = {int(k): str(v) for k, v in names.items()}

    n_masks = len(masks)
    n_boxes = len(boxes) if boxes is not None else 0
    n = min(n_masks, n_boxes) if n_boxes > 0 else n_masks

    if boxes is not None and n_boxes > 0:
        xyxy = boxes.xyxy[:n].detach().cpu().numpy()
        confs = boxes.conf[:n].detach().cpu().numpy()
        clss = boxes.cls[:n].detach().cpu().numpy().astype(int)
    else:
        xyxy = None
        confs = None
        clss = None

    for i in range(n):
        m = masks[i].detach().cpu().numpy().astype(np.uint8)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = m.astype(bool)

        if mask.any():
            frame[mask] = (frame[mask] * (1 - alpha) + overlay[mask] * alpha).astype(np.uint8)

        if xyxy is not None:
            x1, y1, x2, y2 = map(int, xyxy[i])
            cls_id = int(clss[i]) if clss is not None else 0
            conf = float(confs[i]) if confs is not None else 0.0

            name = names.get(cls_id, str(cls_id))
            if str(name).lower() == "item":
                name = "person"

            label = f"{name} {conf:.2f}"
            tx = max(0, x1)
            ty = max(0, y1 - 8)

            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
            )

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

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    if is_file_source:
        t_video = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        now = video_start_wall + t_video
    else:
        now = time.time()

    wall_now = time.time()
    dt = wall_now - t_prev
    t_prev = wall_now
    if dt > 0:
        fps_hist.append(1.0 / dt)

    window_frames += 1
    if wall_now - window_start >= 1.0:
        window_len = wall_now - window_start
        expected_frames = TARGET_FPS * window_len
        drops = max(0, int(round(expected_frames - window_frames)))
        drop_hist.append(drops)
        window_frames = 0
        window_start = wall_now

    people_results = []
    fire_results = []

    if PEOPLE_ON:
        people_results = people_seg_model.predict(
            frame, conf=S.PEOPLE_CONF, classes=[PERSON_CLASS], **PRED_KW
        )

    if FIRE_ON:
        fire_results = fire_model.predict(frame, conf=S.FIRE_CONF, **PRED_KW)

    merged = merge_detections(
        people_results,
        fire_results,
        people_model=people_seg_label,
        fire_model=fire_label,
        seg_on=bool(PEOPLE_ON and S.ATTACH_PEOPLE_MASKS_TO_LOG),
        timestamp=now,
    )

    if FIRE_ON and S.ATTACH_FIRE_MASKS_TO_LOG:
        _attach_masks_to_merged(merged, fire_results, "fire")

    for det in merged:
        det["frame_id"] = frame_id
        if det.get("class") == "item":
            det["class"] = "person"
        if det.get("class_name") == "item":
            det["class_name"] = "person"

    if merged:
        all_detections.extend(merged)

    counts = count_by_class(merged)

    if PEOPLE_ON:
        frame = draw_masks(
            frame,
            people_results,
            names=people_seg_label.names,
            color=S.COLORS["person"],
            alpha=S.MASK_ALPHA,
            text_scale=S.MASK_TEXT_SCALE,
            text_thickness=S.MASK_TEXT_THICKNESS,
        )

    if FIRE_ON:
        frame = draw_masks(
            frame,
            fire_results,
            names=fire_label.names,
            color=S.COLORS["fire"],
            alpha=S.MASK_ALPHA,
            text_scale=S.MASK_TEXT_SCALE,
            text_thickness=S.MASK_TEXT_THICKNESS,
        )

    inf_times = []
    if PEOPLE_ON and people_results:
        inf_times.append(people_results[0].speed.get("inference", 0.0))
    if FIRE_ON and fire_results:
        inf_times.append(fire_results[0].speed.get("inference", 0.0))
    if inf_times:
        inf_hist.append(sum(inf_times) / len(inf_times))

    avg_fps = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0
    avg_inf = sum(inf_hist) / len(inf_hist) if inf_hist else 0.0
    avg_drops = sum(drop_hist) / len(drop_hist) if drop_hist else 0.0

    people_count = counts.get("person", 0) + counts.get("item", 0)
    fire_count = counts.get("fire", 0)
    smoke_count = counts.get("smoke", 0)

    lines = [
        f"FPS: {avg_fps:5.2f}",
        f"Model inference: {avg_inf:5.1f} ms",
        f"People: {people_count}",
        f"Fire: {fire_count}",
        f"Smoke: {smoke_count}",
        f"Dropped frames (avg/s): {avg_drops:.1f}",
        f"RECORDING: {'ON' if RECORDING_ENABLED else 'OFF'} (R)",
        f"People model: {'ON' if PEOPLE_ON else 'OFF'} (K)",
        f"Fire/Smoke model: {'ON' if FIRE_ON else 'OFF'} (L)",
    ]
    frame = draw_hud(
        frame,
        lines,
        anchor=S.HUD_ANCHOR,
        margin=S.HUD_MARGIN,
        alpha=S.HUD_ALPHA,
        font_scale=S.HUD_FONT_SCALE,
        thickness=S.HUD_THICKNESS,

        position=(1500, 275),
    )

    allow_output = (not S.REQUIRE_CONSENT_FOR_OUTPUT) or RECORDING_ENABLED
    if S.SAVE_OUTPUT and allow_output:
        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(S.OUTPUT_VIDEO, fourcc, TARGET_FPS, (w, h))
        out.write(frame)

    cv2.imshow(S.WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == S.KEY_ESC:
        break
    if key in S.KEY_TOGGLE_RECORDING:
        if not RECORDING_ENABLED:
            print("[PII] USER_CONSENT: Recording ENABLED by user at", time.strftime("%Y-%m-%d %H:%M:%S"))
            RECORDING_ENABLED = True
            recording_start_time = time.time()
        else:
            print("[PII] USER_CONSENT: Recording DISABLED at", time.strftime("%Y-%m-%d %H:%M:%S"))
            RECORDING_ENABLED = False
            recording_start_time = None
    elif key in S.KEY_TOGGLE_PEOPLE:
        PEOPLE_ON = not PEOPLE_ON
    elif key in S.KEY_TOGGLE_FIRE:
        FIRE_ON = not FIRE_ON

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()

allow_log = (not S.REQUIRE_CONSENT_FOR_LOG) or RECORDING_ENABLED
if all_detections and allow_log:
    save_detections_json(all_detections, S.DETECTION_LOG_PATH)
