"""
settings.py

Central configuration file for the XRDrone local inference pipeline.

Defines:
  - input/output sources and resolutions
  - YOLO model paths and thresholds
  - UDP and RTSP network streaming parameters
  - tracking configuration and tuning values
  - HUD rendering and overlay behavior
  - runtime toggles and keyboard controls
  - logging and consent requirements

Purpose:
  - isolate runtime configuration from pipeline logic
  - avoid hard-coded values in main.py
  - allow quick environment and deployment changes
"""

from __future__ import annotations
import torch

# -----------------------------
# Input / Output
# -----------------------------
VIDEO_PATH = r"E:\Detection_Segmentation_Demo.mp4"
VIDEO_SOURCE = 0

INPUT_MODE = "camera"  # "camera" | "file"

CAMERA_SOURCE_DEFAULT = "webcam"  # "webcam" | "capture_card"
WEBCAM_INDEX = 0
CAPTURE_CARD_INDEX = 1

CAPTURE_BACKEND = "auto"

SAVE_OUTPUT = False
OUTPUT_VIDEO = "Segmentation_Aeroscapes.mp4"
OUTPUT_CODEC = "mp4v"

FORCE_OUTPUT_1080P = True
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_KEEP_ASPECT = True
REQUEST_CAMERA_1080P = True

# -----------------------------
# Network streaming
# -----------------------------
ENABLE_RTSP = False
RTSP_URL = "rtsp://127.0.0.1:8554/stream"

ENABLE_UDP = True
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

REQUIRE_CONSENT_FOR_NETWORK = False

UNITY_CLASS_ID = {
    "person": 0,
    "fire": 1,
    "smoke": 2,
    "chair": 3,
    "couch": 4,
    "sofa": 4,
    "dining table": 5,
}

UDP_MIN_CONF = 0.80

UDP_SEND_CLASSES = (
    "person",
    "fire",
    "smoke",
    "chair",
    "couch",
    "dining table",
)

# -----------------------------
# Logging
# -----------------------------
DETECTION_LOG_PATH = "detections_log.json"

REQUIRE_CONSENT_FOR_OUTPUT = True
REQUIRE_CONSENT_FOR_LOG = True

# -----------------------------
# Models
# -----------------------------
PEOPLE_MODEL_PATH = "../models/yolo26n-seg.pt"
FIRE_MODEL_PATH = "../models/fire_smoke_detection.pt"

DETECT_CLASSES = ("person", "chair", "couch", "dining table")

PEOPLE_CONF = 0.40
FIRE_CONF = 0.25

IMGSZ = 960

# -----------------------------
# Compute / Performance
# -----------------------------
DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_FP16 = bool(torch.cuda.is_available())

DEFAULT_FPS = 30.0
WINDOW_NAME = "Live Pipeline"

# -----------------------------
# Default runtime toggles
# -----------------------------
PEOPLE_ON_DEFAULT = True
FIRE_ON_DEFAULT = False
RECORDING_ENABLED_DEFAULT = False

# Persistent multi-object tracking (stable IDs)
TRACKING_ENABLED_DEFAULT = True

# Choose tracker backend:
#  - "opencv": lightweight Kalman+IoU tracker in tracker.py
#  - "ultralytics": Ultralytics built-in trackers (BoT-SORT/ByteTrack)
TRACKING_METHOD = "opencv"  # "opencv" | "ultralytics"

# OpenCV tracker tuning
TRACK_MIN_IOU = 0.30
TRACK_MAX_AGE_FRAMES = 90  # how long an object can be missing and still keep its ID
TRACK_PER_CLASS = True
TRACK_KF_PROCESS_NOISE = 1e-2
TRACK_KF_MEAS_NOISE = 1e-1

# Ultralytics tracker config (only used when TRACKING_METHOD="ultralytics")
# Ultralytics supports "botsort.yaml" (default) and "bytetrack.yaml".
ULTRALYTICS_TRACKER = "bytetrack.yaml"

# If you use separate trackers per model (people vs fire), offsets prevent ID collisions.
TRACK_ID_OFFSET_PEOPLE = 0
TRACK_ID_OFFSET_FIRE = 1_000_000

# Draw per-instance IDs on the output frame (useful for demos)
DRAW_TRACK_IDS = True

DRAW_DETECTIONS_DEFAULT = True

# -----------------------------
# Mask rendering
# -----------------------------
MASK_ALPHA = 0.35
MASK_TEXT_SCALE = 0.6
MASK_TEXT_THICKNESS = 2

ATTACH_PEOPLE_MASKS_TO_LOG = True
ATTACH_FIRE_MASKS_TO_LOG = True

COLORS = {
    "person": (255, 255, 0),
    "item": (255, 0, 0),
    "fire": (255, 0, 255),
    "smoke": (0, 255, 255),
    "chair": (0, 200, 0),
    "couch": (200, 0, 0),
    "dining table": (0, 165, 255),
}

# -----------------------------
# DJI menu overlay (PNG on top of video)
# -----------------------------
DJI_MENU_OVERLAY_PATH = "DJImenu.png"
DJI_MENU_OVERLAY_ENABLED_DEFAULT = True

# -----------------------------
# Keybinds
# -----------------------------
KEY_ESC = 27
KEY_TOGGLE_RECORDING = (ord("r"), ord("R"))
KEY_TOGGLE_PEOPLE = (ord("k"), ord("K"))
KEY_TOGGLE_FIRE = (ord("l"), ord("L"))

KEY_TOGGLE_INPUT = (ord("i"), ord("I"))
KEY_TOGGLE_DRAW = (ord("v"), ord("V"))
KEY_TOGGLE_DJI_OVERLAY = (ord("u"), ord("U"))
KEY_TOGGLE_TRACKING = (ord("t"), ord("T"))

# -----------------------------
# Test mode
# -----------------------------
TEST_IMAGE_PATH = "/Users/troy/Desktop/XRDrone/models/yolo_people_fire_smoke/people_furniture.avif"