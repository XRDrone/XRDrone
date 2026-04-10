# Settings Reference

## Core pipeline settings

| Setting | What it does |
|---|---|
| `VIDEO_PATH` | Video file path used when `INPUT_MODE="file"`. |
| `VIDEO_SOURCE` | Default numeric video source. |
| `INPUT_MODE` | Selects `camera` or `file` input. |
| `CAMERA_SOURCE_DEFAULT` | Chooses the default camera source name. |
| `WEBCAM_INDEX` | Camera index for the webcam source. |
| `CAPTURE_CARD_INDEX` | Camera index for the capture-card source. |
| `CAPTURE_BACKEND` | Capture backend selection hint. |
| `SAVE_OUTPUT` | Enables writing the rendered output video. |
| `OUTPUT_VIDEO` | Output video path. |
| `OUTPUT_CODEC` | FourCC used for saved video. |
| `FORCE_OUTPUT_1080P` | If `True`, attempts to normalize output resolution to 1080p. |
| `OUTPUT_WIDTH` / `OUTPUT_HEIGHT` | Requested output dimensions. |
| `OUTPUT_KEEP_ASPECT` | Preserves aspect ratio during resize when possible. |
| `REQUEST_CAMERA_1080P` | Requests 1080p from the live camera source. |

## UDP output

| Setting | What it does |
|---|---|
| `ENABLE_UDP` | Turns UDP publishing on or off. |
| `UDP_IP` / `UDP_PORT` | Destination address for Unity or a local listener. |
| `UNITY_CLASS_ID` | Maps class names to Unity-facing integer IDs. |
| `UDP_MIN_CONF` | Minimum confidence required for emission unless forced by continuity logic. |
| `UDP_SEND_CLASSES` | Classes allowed into the UDP packet. |

## Models and inference

| Setting | What it does |
|---|---|
| `PEOPLE_MODEL_PATH` | Path to the people model. |
| `FIRE_MODEL_PATH` | Path to the fire/smoke model. |
| `DETECT_CLASSES` | Class names requested from the people model. |
| `PEOPLE_CONF` | Confidence threshold for people inference when tracking overlay is not driving emission. |
| `FIRE_CONF` | Confidence threshold for fire inference. |
| `IMGSZ` | Ultralytics inference image size. |
| `DEVICE` | Runtime device selection (`cpu` or CUDA index). |
| `USE_FP16` | Enables half precision on supported CUDA hardware. |

## Runtime defaults

| Setting | What it does |
|---|---|
| `PEOPLE_ON_DEFAULT` | Enables people inference on startup. |
| `FIRE_ON_DEFAULT` | Enables fire inference on startup. |
| `RECORDING_ENABLED_DEFAULT` | Starts with recording enabled or disabled. |
| `TRACK_ID_OFFSET_PEOPLE` | Base ID offset for person tracks. |
| `TRACK_ID_OFFSET_FIRE` | Base ID offset for fire tracks. |
| `DRAW_TRACK_IDS` | If `True`, draws each track ID on the output frame. |
| `DRAW_DETECTIONS_DEFAULT` | If `True`, draws detection overlays by default. |

## Object-ID flicker mitigation

| Setting | What it does |
|---|---|
| `ID_FLICKER_APPLY_CLASSES` | Classes that should use flicker mitigation. |
| `ID_FLICKER_EMA_ALPHA` | Smoothing factor for the confidence EMA. Higher values react faster to new scores. |
| `ID_FLICKER_TAU_ON` | Confidence level required for a new ID to start being emitted. |
| `ID_FLICKER_TAU_OFF` | Lower confidence level an already-emitted ID can fall to before being dropped. |
| `ID_FLICKER_COAST_FRAMES` | Number of recent missing frames a track can survive while still being emitted. |
| `ID_FLICKER_DROP_FRAMES` | Hard limit on how long a missing track can persist before it is fully removed. |
| `ID_FLICKER_COAST_CONF_DECAY` | Confidence decay applied while a missing track is being coasted forward. |

## Rendering and overlays

| Setting | What it does |
|---|---|
| `MASK_ALPHA` | Alpha used for rendered masks. |
| `MASK_TEXT_SCALE` | Text scale used for mask and track labels. |
| `MASK_TEXT_THICKNESS` | Text thickness used for mask and track labels. |
| `COLORS` | Per-class rendering colors. |
| `DJI_MENU_OVERLAY_PATH` | RGBA overlay image path. |
| `DJI_MENU_OVERLAY_ENABLED_DEFAULT` | Enables the DJI overlay on startup. |

## Keybinds

| Setting | What it does |
|---|---|
| `KEY_ESC` | Exit the live loop. |
| `KEY_TOGGLE_RECORDING` | Toggle recording. |
| `KEY_TOGGLE_PEOPLE` | Toggle people inference. |
| `KEY_TOGGLE_FIRE` | Toggle fire inference. |
| `KEY_TOGGLE_INPUT` | Swap webcam/capture-card input when camera mode is active. |
| `KEY_TOGGLE_DRAW` | Toggle drawing of detection overlays. |
| `KEY_TOGGLE_DJI_OVERLAY` | Toggle the DJI overlay. |
| `KEY_TOGGLE_TRACKING` | Toggle tracked-box mode. |

## Test mode

| Setting | What it does |
|---|---|
| `TEST_IMAGE_PATH` | Default image path used by `main.py --test`. |
