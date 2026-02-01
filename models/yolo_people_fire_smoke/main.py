from ultralytics import YOLO
import cv2, time, numpy as np
from collections import deque
from hud import draw_hud
from merger import merge_detections, count_by_class
from detection_logger import save_detections_json
from types import SimpleNamespace
import torch
import settings as S

from streaming import RTSPStreamer, UDPPublisher
from output_formatter import to_unity_udp_packet


print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))


PEOPLE_ON = S.PEOPLE_ON_DEFAULT
FIRE_ON = S.FIRE_ON_DEFAULT
RECORDING_ENABLED = S.RECORDING_ENABLED_DEFAULT
recording_start_time = None

# NEW: drawing toggle (visuals only; UDP still runs)
DRAW_DETECTIONS = S.DRAW_DETECTIONS_DEFAULT

# Track which camera we are using at runtime (only relevant when INPUT_MODE="camera")
ACTIVE_CAMERA_SOURCE = S.CAMERA_SOURCE_DEFAULT  # "webcam" | "capture_card"


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


def draw_masks(
    frame,
    results,
    names,
    color=(0, 255, 0),
    alpha=0.35,
    text_scale=0.6,
    text_thickness=2,
):
    if not results:
        return frame

    r = results[0]
    boxes = getattr(r, "boxes", None)
    masks = getattr(r, "masks", None)

    if boxes is None or len(boxes) == 0:
        return frame

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy()

    mask_data = None
    if masks is not None and getattr(masks, "data", None) is not None:
        mask_data = masks.data.detach().cpu().numpy()  # (n, h, w)

    n = len(xyxy)
    if mask_data is not None:
        n = min(n, mask_data.shape[0])

    h_img, w_img = frame.shape[:2]
    color_arr = np.array(color, dtype=np.float32)

    for i in range(n):
        x1, y1, x2, y2 = xyxy[i]
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

        if mask_data is not None:
            m = mask_data[i]
            m = (m > 0.5)
            frame[m] = (
                frame[m].astype(np.float32) * (1.0 - alpha) + color_arr * alpha
            ).astype(np.uint8)

        cls_id = int(cls[i]) if i < len(cls) else 0
        name = str(names.get(cls_id, "obj"))
        label = f"{name} {float(conf[i]):.2f}"

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
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


def _attach_masks_to_merged(merged, yolo_results, source_prefix: str):
    if not yolo_results:
        return
    r = yolo_results[0]
    masks = getattr(r, "masks", None)
    if masks is None or getattr(masks, "data", None) is None:
        return

    mask_data = masks.data.detach().cpu().numpy()  # (n, h, w)
    idx = 0
    for det in merged:
        if str(det.get("source", "")).lower().startswith(source_prefix.lower()):
            if idx < mask_data.shape[0]:
                det["mask"] = mask_data[idx]
            idx += 1


def _cv_backend_flag(name: str) -> int:
    name = (name or "auto").lower().strip()
    if name == "auto":
        return 0
    mapping = {
        "dshow": "CAP_DSHOW",
        "msmf": "CAP_MSMF",
        "v4l2": "CAP_V4L2",
        "avfoundation": "CAP_AVFOUNDATION",
    }
    attr = mapping.get(name)
    if not attr:
        return 0
    return int(getattr(cv2, attr, 0) or 0)


def _open_capture(input_mode: str, camera_source: str):
    """
    Returns: cap, is_file_source, target_fps, video_start_wall, input_desc
    """
    input_mode = input_mode.lower().strip()

    if input_mode == "file":
        cap = cv2.VideoCapture(S.VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {S.VIDEO_PATH}")
        input_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        target_fps = input_fps if input_fps > 1 else S.DEFAULT_FPS
        return cap, True, target_fps, time.time(), f"file: {S.VIDEO_PATH}"

    cam = camera_source.lower().strip()
    if cam == "capture_card":
        index = int(S.CAPTURE_CARD_INDEX)
    else:
        index = int(S.WEBCAM_INDEX)

    backend_flag = _cv_backend_flag(S.CAPTURE_BACKEND)
    if backend_flag != 0:
        cap = cv2.VideoCapture(index, backend_flag)
        backend_desc = S.CAPTURE_BACKEND
    else:
        cap = cv2.VideoCapture(index)
        backend_desc = "auto"

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index={index} backend={backend_desc}"
        )

    input_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    target_fps = input_fps if input_fps > 1 else S.DEFAULT_FPS
    return cap, False, target_fps, time.time(), f"camera: {cam} (index={index}, backend={backend_desc})"


all_detections = []
fps_hist = deque(maxlen=30)
inf_hist = deque(maxlen=30)
drop_hist = deque(maxlen=30)
t_prev = time.time()

cap, is_file_source, TARGET_FPS, video_start_wall, input_desc = _open_capture(
    S.INPUT_MODE, ACTIVE_CAMERA_SOURCE
)

fourcc = cv2.VideoWriter_fourcc(*S.OUTPUT_CODEC)
out = None

window_frames = 0
window_start = time.time()
frame_id = 0

rtsp = RTSPStreamer(S.RTSP_URL, fps=TARGET_FPS) if S.ENABLE_RTSP else None
udp = UDPPublisher(S.UDP_IP, S.UDP_PORT) if S.ENABLE_UDP else None


def _network_allowed() -> bool:
    return (not S.REQUIRE_CONSENT_FOR_NETWORK) or RECORDING_ENABLED


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

    # NEW: drawing is gated; inference/merge/UDP are NOT.
    if DRAW_DETECTIONS and PEOPLE_ON:
        frame = draw_masks(
            frame,
            people_results,
            names=people_seg_label.names,
            color=S.COLORS["person"],
            alpha=S.MASK_ALPHA,
            text_scale=S.MASK_TEXT_SCALE,
            text_thickness=S.MASK_TEXT_THICKNESS,
        )

    if DRAW_DETECTIONS and FIRE_ON:
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

    net_on = _network_allowed()
    lines = [
        f"FPS: {avg_fps:5.2f}",
        f"Model inference: {avg_inf:5.1f} ms",
        f"People: {people_count}",
        f"Fire: {fire_count}",
        f"Smoke: {smoke_count}",
        f"Dropped frames (avg/s): {avg_drops:.1f}",
        f"Input: {input_desc}",
        f"Det overlays: {'ON' if DRAW_DETECTIONS else 'OFF'} (V)",
        f"RTSP: {'ON' if (rtsp and net_on) else 'OFF'}",
        f"UDP:  {'ON' if (udp and net_on) else 'OFF'}",
        f"RECORDING: {'ON' if RECORDING_ENABLED else 'OFF'} (R)",
        f"People model: {'ON' if PEOPLE_ON else 'OFF'} (K)",
        f"Fire/Smoke model: {'ON' if FIRE_ON else 'OFF'} (L)",
        f"Toggle input: (I)",
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

    # UDP/RTSP remain independent of DRAW_DETECTIONS.
    if net_on:
        if udp is not None:
            h, w = frame.shape[:2]
            pkt = to_unity_udp_packet(
                merged,
                frame_id=frame_id,
                timestamp=now,
                width=w,
                height=h,
                class_map=S.UNITY_CLASS_ID,
                allowed_classes=S.UDP_SEND_CLASSES,
            )
            try:
                udp.send_json(pkt)
            except Exception:
                pass

        if rtsp is not None:
            rtsp.write(frame)

    cv2.imshow(S.WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == S.KEY_ESC:
        break

    if key in S.KEY_TOGGLE_RECORDING:
        if not RECORDING_ENABLED:
            print(
                "[PII] USER_CONSENT: Recording ENABLED by user at",
                time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            RECORDING_ENABLED = True
            recording_start_time = time.time()
        else:
            print(
                "[PII] USER_CONSENT: Recording DISABLED at",
                time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            RECORDING_ENABLED = False
            recording_start_time = None

    elif key in S.KEY_TOGGLE_PEOPLE:
        PEOPLE_ON = not PEOPLE_ON

    elif key in S.KEY_TOGGLE_FIRE:
        FIRE_ON = not FIRE_ON

    elif key in S.KEY_TOGGLE_DRAW:
        DRAW_DETECTIONS = not DRAW_DETECTIONS

    elif key in S.KEY_TOGGLE_INPUT and S.INPUT_MODE.lower() == "camera":
        prev = ACTIVE_CAMERA_SOURCE
        ACTIVE_CAMERA_SOURCE = "capture_card" if prev == "webcam" else "webcam"

        try:
            cap.release()
        except Exception:
            pass

        try:
            cap, is_file_source, TARGET_FPS, video_start_wall, input_desc = _open_capture(
                "camera", ACTIVE_CAMERA_SOURCE
            )

            if rtsp is not None:
                rtsp.close()
                rtsp = RTSPStreamer(S.RTSP_URL, fps=TARGET_FPS)

            fps_hist.clear()
            drop_hist.clear()
            t_prev = time.time()
            window_frames = 0
            window_start = time.time()

        except Exception as e:
            print("Toggle input failed:", e)
            ACTIVE_CAMERA_SOURCE = prev
            cap, is_file_source, TARGET_FPS, video_start_wall, input_desc = _open_capture(
                "camera", ACTIVE_CAMERA_SOURCE
            )


try:
    cap.release()
except Exception:
    pass

if out is not None:
    try:
        out.release()
    except Exception:
        pass

cv2.destroyAllWindows()

if rtsp is not None:
    rtsp.close()

if udp is not None:
    udp.close()

allow_log = (not S.REQUIRE_CONSENT_FOR_LOG) or RECORDING_ENABLED
if all_detections and allow_log:
    save_detections_json(all_detections, S.DETECTION_LOG_PATH)
