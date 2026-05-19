"""
settings.py

Runtime configuration for the pure ArUco XRDrone video pipeline.
This configuration uses ArUco marker pose estimation only.
"""

from __future__ import annotations

try:
    import torch
except Exception:  # Keep settings importable even before torch is installed.
    torch = None

# -----------------------------
# Input / Output
# -----------------------------
VIDEO_PATH = "2026_05_18_15_28_04_Cache_Trimmed.mp4"
VIDEO_SOURCE = 0
INPUT_MODE = "file"  # "file" | "camera"

SAVE_OUTPUT = False
OUTPUT_VIDEO = "aruco_output.mp4"
OUTPUT_CODEC = "mp4v"

FORCE_OUTPUT_1080P = True
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_KEEP_ASPECT = True
REQUEST_CAMERA_1080P = True

DEFAULT_FPS = 30.0
WINDOW_NAME = "XRDrone Pure ArUco Pipeline"
PRINT_EVERY_N_FRAMES = 30

# -----------------------------
# Logging
# -----------------------------
LOG_ROOT = "logs"

# Logs are disabled by default. Run `python main.py --logs` to create:
#   logs/run_YYYYMMDD_HHMMSS/run_metadata.json
#   logs/run_YYYYMMDD_HHMMSS/summary.json
#   logs/run_YYYYMMDD_HHMMSS/pose_log.jsonl
#   logs/run_YYYYMMDD_HHMMSS/marker_log.jsonl
#   logs/run_YYYYMMDD_HHMMSS/detections_log.jsonl
#   logs/run_YYYYMMDD_HHMMSS/packets_log.jsonl
#   logs/run_YYYYMMDD_HHMMSS/frames_log.csv
#   logs/run_YYYYMMDD_HHMMSS/errors_log.jsonl

# -----------------------------
# UDP output to Unity or other listeners
# -----------------------------
ENABLE_UDP = True
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
JSON_SERIALIZER = "orjson"  # "orjson" | "json"
JSON_ENSURE_ASCII = False

UNITY_CLASS_ID = {
    "person": 0,
}
UDP_SEND_CLASSES = ("person",)
UDP_MIN_CONF = 0.80

# -----------------------------
# Optional YOLO human detection
# -----------------------------
# If this file is missing, main.py still runs and logs empty detections.
PEOPLE_MODEL_PATH = "../models/yolo26n-seg.pt"
DETECT_CLASSES = ("person",)
PEOPLE_CONF = 0.40
IMGSZ = 960
TRACKING_ENABLED = True
ULTRALYTICS_TRACKER_YAML = "botsort_drone.yaml"

if torch is not None and torch.cuda.is_available():
    DEVICE = 0
    USE_FP16 = True
else:
    DEVICE = "cpu"
    USE_FP16 = False

# Fire detection is off for this pure ArUco video run.
FIRE_ON_DEFAULT = False
FIRE_MODEL_PATH = "../models/fire_smoke_detection.pt"
FIRE_CONF = 0.25

# -----------------------------
# Pose estimation: camera pose via ArUco only
# -----------------------------
POSE_ENABLED_DEFAULT = True
POSE_HFOV_DEG = 84.0

# Physical marker size in meters. Change this if your printed markers are not 16.45 cm.
POSE_MARKER_SIZE_M = 0.1645

# The provided video uses marker IDs 0, 1, and 2 at these world coordinates.
# Units are meters. Coordinate format is (x, y, z).
POSE_MARKER_WORLD_POSITIONS = {
    0: (0.0, 0.0, 0.0),
    1: (-3.03, 0.0, 0.0),
    2: (-5.0, 0.0, 0.0),
}

POSE_ARUCO_DICT = "DICT_4X4_50"
POSE_CORNER_REFINEMENT = "subpix"  # "none" | "subpix" | "contour" | "apriltag"
POSE_DRAW_ARUCO = True

# -----------------------------
# Rendering / keybinds
# -----------------------------
DRAW_DETECTIONS_DEFAULT = True
KEY_ESC = 27

# Kept for compatibility with test_runner.py.
TEST_IMAGE_PATH = "people_furniture.avif"
