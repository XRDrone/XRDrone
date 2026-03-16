# settings.md

This file explains the runtime settings in `settings.py`.
Edit these values to control video input, models, UDP output, tracking, pose estimation, smoothing, overlays, and keybinds.

## Input / Output

| Setting | What it does |
|---|---|
| `VIDEO_PATH` | Path to the input video file used when file mode is active. |
| `VIDEO_SOURCE` | Default camera source index for OpenCV if a generic camera source is used. |
| `INPUT_MODE` | Chooses whether input comes from a live camera (`"camera"`) or a saved video (`"file"`). |
| `CAMERA_SOURCE_DEFAULT` | Chooses which camera type to prefer by default: webcam or capture card. |
| `WEBCAM_INDEX` | Camera index to use for the webcam. |
| `CAPTURE_CARD_INDEX` | Camera index to use for the capture card. |
| `CAPTURE_BACKEND` | OpenCV capture backend selection. `"auto"` lets OpenCV choose. |
| `SAVE_OUTPUT` | If `True`, saves the processed output video to disk. |
| `OUTPUT_VIDEO` | Output file path for the saved processed video. |
| `OUTPUT_CODEC` | Video codec used when saving output. |
| `FORCE_OUTPUT_1080P` | If `True`, forces the output frame size to 1080p. |
| `OUTPUT_WIDTH` | Output video width in pixels. |
| `OUTPUT_HEIGHT` | Output video height in pixels. |
| `OUTPUT_KEEP_ASPECT` | If `True`, preserves aspect ratio when resizing output. |
| `REQUEST_CAMERA_1080P` | If `True`, asks the live camera for a 1080p feed. |

## UDP Output

| Setting | What it does |
|---|---|
| `ENABLE_UDP` | Turns UDP packet sending on or off. |
| `UDP_IP` | Destination IP address for UDP packets. |
| `UDP_PORT` | Destination UDP port. |
| `UNITY_CLASS_ID` | Maps class names to the numeric IDs expected by Unity. |
| `UDP_MIN_CONF` | Minimum confidence required before a detection is allowed into the UDP stream. |
| `UDP_SEND_CLASSES` | Tuple of class names that are allowed to be sent over UDP. |

## Logging

| Setting | What it does |
|---|---|
| `DETECTION_LOG_PATH` | File path used to save detection logs. |

## Models

| Setting | What it does |
|---|---|
| `PEOPLE_MODEL_PATH` | Path to the people-detection model. |
| `FIRE_MODEL_PATH` | Path to the fire/smoke model. |
| `DETECT_CLASSES` | Classes the main detection pipeline should keep. |
| `PEOPLE_CONF` | Confidence threshold for the people model. |
| `FIRE_CONF` | Confidence threshold for the fire/smoke model. |
| `IMGSZ` | Inference image size sent to the model. Larger sizes may improve detail but cost more speed. |

## Compute / Performance

| Setting | What it does |
|---|---|
| `DEVICE` | Selects the compute device. Uses CUDA GPU when available, otherwise CPU. |
| `USE_FP16` | Enables half-precision inference on CUDA for better speed and lower memory use. |

## Serialization / Transport

| Setting | What it does |
|---|---|
| `JSON_SERIALIZER` | Chooses the JSON encoder. `"orjson"` is faster when installed; `"json"` uses Python’s standard library. |
| `JSON_ENSURE_ASCII` | Controls whether JSON text is forced to ASCII-only output. |
| `DEFAULT_FPS` | Default FPS value used when a source FPS is unavailable. |
| `WINDOW_NAME` | Title of the OpenCV display window. |

## Default Runtime Toggles

| Setting | What it does |
|---|---|
| `PEOPLE_ON_DEFAULT` | Starts the app with people detection enabled or disabled. |
| `FIRE_ON_DEFAULT` | Starts the app with fire detection enabled or disabled. |
| `RECORDING_ENABLED_DEFAULT` | Starts the app with recording enabled or disabled. |
| `TRACKING_ENABLED_DEFAULT` | Starts the app with persistent object tracking enabled or disabled. |
| `TRACKING_METHOD` | Chooses the tracking backend: lightweight OpenCV tracking or Ultralytics tracking. |

## OpenCV Tracker Tuning

| Setting | What it does |
|---|---|
| `TRACK_MIN_IOU` | Minimum overlap needed for a detection and track to be considered a match. |
| `TRACK_MAX_AGE_FRAMES` | Number of frames a track can disappear before it is removed. Higher values keep IDs alive longer. |
| `TRACK_PER_CLASS` | If `True`, only matches objects within the same class. |
| `TRACK_KF_PROCESS_NOISE` | Kalman filter process noise. Higher values make the tracker adapt faster but can be less stable. |
| `TRACK_KF_MEAS_NOISE` | Kalman filter measurement noise. Higher values trust detections less. |

## OpenCV Matching Settings

| Setting | What it does |
|---|---|
| `TRACK_MATCHING_METHOD` | Chooses how matches are assigned: Hungarian matching or greedy matching. |
| `TRACK_MIN_MATCH_SCORE` | Minimum total match score required to accept a detection-to-track match. |
| `TRACK_MAX_FOOT_DISTANCE_NORM` | Maximum allowed normalized foot-point distance for a match. |
| `TRACK_MAX_WORLD_DISTANCE_M` | Maximum allowed world-space distance in meters for a match. |
| `TRACK_USE_WORLD_POSITION` | If `True`, uses estimated world position as part of matching. |
| `TRACK_WORLD_SCORE_WEIGHT` | Weight given to world-position agreement in the match score. |
| `TRACK_IOU_SCORE_WEIGHT` | Weight given to bounding-box overlap in the match score. |
| `TRACK_FOOT_SCORE_WEIGHT` | Weight given to foot-point distance in the match score. |

## Ultralytics Tracker Settings

| Setting | What it does |
|---|---|
| `ULTRALYTICS_TRACKER` | Tracker config file used by Ultralytics, such as BoT-SORT or ByteTrack. |
| `TRACKING_INPUT_CONF_PEOPLE` | Lower confidence threshold used during tracking for people, to help keep IDs alive. |
| `TRACKING_INPUT_CONF_FIRE` | Lower confidence threshold used during tracking for fire/smoke, to help keep IDs alive. |
| `TRACK_ID_OFFSET_PEOPLE` | ID offset applied to people tracks. |
| `TRACK_ID_OFFSET_FIRE` | ID offset applied to fire tracks so their IDs do not collide with people IDs. |
| `DRAW_TRACK_IDS` | If `True`, draws each track ID on the output frame. |
| `DRAW_DETECTIONS_DEFAULT` | If `True`, draws detection overlays by default. |

## Object-ID Flicker Mitigation

These settings reduce temporary ID dropouts in the UDP stream without changing the packet schema.

| Setting | What it does |
|---|---|
| `ID_FLICKER_MITIGATION_ENABLED_DEFAULT` | Turns flicker mitigation on or off by default. |
| `ID_FLICKER_APPLY_CLASSES` | Classes that should use flicker mitigation. |
| `ID_FLICKER_REQUIRE_TRACK_ID` | If `True`, mitigation only applies when a stable track ID exists. |
| `ID_FLICKER_USE_CONF_EMA` | If `True`, smooths confidence over time with an exponential moving average. |
| `ID_FLICKER_EMA_ALPHA` | Smoothing factor for the confidence EMA. Higher values react faster to new scores. |
| `ID_FLICKER_TAU_ON` | Confidence level required for a new ID to start being emitted. |
| `ID_FLICKER_TAU_OFF` | Lower confidence level an already-emitted ID can fall to before being dropped. |
| `ID_FLICKER_COAST_FRAMES` | Number of recent missing frames a track can survive while still being emitted. |
| `ID_FLICKER_DROP_FRAMES` | Hard limit on how long a missing track can persist before it is fully removed. |
| `ID_FLICKER_COAST_CONF_DECAY` | Confidence decay applied while a missing track is being coasted forward. |

## Pose Estimation (Camera Pose via ArUco)

These settings control ArUco-based camera pose estimation and the `pose` object added to UDP.

| Setting | What it does |
|---|---|
| `POSE_ENABLED_DEFAULT` | Turns pose estimation on or off by default. |
| `POSE_HFOV_DEG` | Horizontal field of view used to approximate camera intrinsics. |
| `POSE_MARKER_SIZE_M` | Real-world marker size in meters. Must match the printed markers. |
| `POSE_ARUCO_DICT` | ArUco dictionary name used for marker detection. |
| `POSE_MARKER_WORLD_POSITIONS` | World-space positions of known marker IDs, in meters. |
| `POSE_USE_CASE` | Chooses pose strategy: automatic, single-marker only, or multi-marker board only. |
| `POSE_SINGLE_INIT_SOLVER` | Initial solver used when only one marker is visible. |
| `POSE_MULTI_INIT_SOLVER` | Initial solver used when multiple markers are visible together. |
| `POSE_REFINER` | Refinement method applied after the initial pose solve. |
| `POSE_ENABLE_REFINEMENT` | Enables or disables the nonlinear refinement step. |
| `POSE_MIN_MARKERS_FOR_MULTI` | Minimum number of visible known markers required to use the multi-marker path. |
| `POSE_CORNER_REFINEMENT` | Optional corner refinement method for ArUco detection. |
| `POSE_RANSAC_REPROJ_THRESHOLD_PX` | RANSAC reprojection error threshold in pixels. |
| `POSE_RANSAC_CONFIDENCE` | RANSAC confidence level. |
| `POSE_RANSAC_ITERATIONS` | Maximum RANSAC iterations. |
| `POSE_DRAW_ARUCO` | If `True`, draws detected ArUco markers on the frame. |
| `POSE_MODE_OVERLAY_ENABLED_DEFAULT` | If `True`, shows a small overlay describing the current pose mode. |
| `POSE_MODE_OVERLAY_ORIGIN` | On-screen position of the pose mode overlay. |
| `POSE_MODE_OVERLAY_TEXT_SCALE` | Text size for the pose mode overlay. |
| `POSE_MODE_OVERLAY_TEXT_THICKNESS` | Text thickness for the pose mode overlay. |
| `POSE_LOSS_HOLD_ENABLED_DEFAULT` | If `True`, enables the pose-loss fallback when no valid marker corners are visible. |
| `POSE_LOSS_HOLD_TIMEOUT_S` | How long the last valid pose numbers may be retained during full marker occlusion before the hold expires. |
| `POSE_LOSS_PRESERVE_LAST_NUMBERS_DURING_HOLD` | If `True`, keeps the last valid numeric pose values during the short pose-loss hold window while still marking the pose invalid. |
| `POSE_LOSS_CLEAR_NUMBERS_AFTER_TIMEOUT` | If `True`, clears the retained numeric pose values after the pose-loss hold timeout expires. |

## Motion Smoothing

These settings smooth camera pose and projected object motion.

| Setting | What it does |
|---|---|
| `MOTION_SMOOTHING` | Main 0.0 to 1.0 smoothing slider. Lower is more responsive; higher is smoother. |
| `MOTION_SMOOTHING_STEP` | Amount the smoothing slider changes when adjusted. |
| `MOTION_SMOOTHING_DERIVATIVE_CUTOFF_HZ` | Cutoff used in the smoothing filter for motion derivatives. |
| `MOTION_SMOOTHING_RESET_TIMEOUT_S` | Time after which smoothing state resets if updates stop arriving. |
| `POSE_MOTION_SMOOTHING_ENABLED_DEFAULT` | Turns pose smoothing on or off by default. |
| `WORLD_MOTION_SMOOTHING_ENABLED_DEFAULT` | Turns projected world-position smoothing on or off by default. |
| `WORLD_MOTION_SMOOTHING_MAX_TRACK_AGE_S` | Maximum age of a track that can still be smoothed in world space. |

## Mask Rendering

| Setting | What it does |
|---|---|
| `MASK_ALPHA` | Transparency of segmentation masks drawn on the frame. |
| `MASK_TEXT_SCALE` | Text size used for mask labels. |
| `MASK_TEXT_THICKNESS` | Text thickness used for mask labels. |
| `COLORS` | Per-class color map used for rendering overlays. |

## DJI Menu Overlay

| Setting | What it does |
|---|---|
| `DJI_MENU_OVERLAY_PATH` | Path to the PNG image used as the DJI-style menu overlay. |
| `DJI_MENU_OVERLAY_ENABLED_DEFAULT` | Turns the DJI-style overlay on or off by default. |

## Keybinds

| Setting | What it does |
|---|---|
| `KEY_ESC` | Exit key. |
| `KEY_TOGGLE_RECORDING` | Keys that toggle recording on or off. |
| `KEY_TOGGLE_PEOPLE` | Keys that toggle people detection on or off. |
| `KEY_TOGGLE_FIRE` | Keys that toggle fire detection on or off. |
| `KEY_TOGGLE_INPUT` | Keys that switch input source or mode. |
| `KEY_TOGGLE_DRAW` | Keys that toggle drawing overlays on or off. |
| `KEY_TOGGLE_DJI_OVERLAY` | Keys that toggle the DJI menu overlay on or off. |
| `KEY_TOGGLE_TRACKING` | Keys that toggle tracking on or off. |
| `KEY_TOGGLE_POSE_MODE_OVERLAY` | Keys that toggle the pose mode overlay on or off. |
| `KEY_TOGGLE_MOTION_SMOOTHING` | Keys that toggle motion smoothing on or off. |
| `KEY_DECREASE_MOTION_SMOOTHING` | Keys that decrease the smoothing amount. |
| `KEY_INCREASE_MOTION_SMOOTHING` | Keys that increase the smoothing amount. |

## Test Mode

| Setting | What it does |
|---|---|
| `TEST_IMAGE_PATH` | Path to the image used for test-mode runs. |

## Practical Notes

- For normal use, the most commonly changed settings are usually:
  - `INPUT_MODE`
  - `WEBCAM_INDEX` / `CAPTURE_CARD_INDEX`
  - `ENABLE_UDP`, `UDP_IP`, `UDP_PORT`
  - `PEOPLE_CONF`, `FIRE_CONF`
  - `TRACKING_METHOD`
  - `MOTION_SMOOTHING`
  - `POSE_ENABLED_DEFAULT`
- `DEVICE` and `USE_FP16` are chosen automatically from CUDA availability.
- The flicker-mitigation and motion-smoothing sections are the main controls for making tracked objects look more stable.
