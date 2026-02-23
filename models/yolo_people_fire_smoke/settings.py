# settings.py
"""
XRDrone local pipeline settings.

This file centralizes all runtime configuration for the local YOLO
inference demo (video source, model paths, thresholds, HUD, logging, keys,
and network streaming).
Edit values here; main.py should not contain hard-coded settings.
"""

from __future__ import annotations
import torch

# -----------------------------
# Input / Output
# -----------------------------
VIDEO_PATH = r"E:\Detection_Segmentation_Demo.mp4"  # file path if using video file input
VIDEO_SOURCE = 0  # legacy: set to VIDEO_PATH for file input, or keep as camera index (e.g., 0)

# Input mode + camera source toggle
# INPUT_MODE:
#   - "camera": read from webcam/capture-card
#   - "file":   read from VIDEO_PATH
INPUT_MODE = "camera"  # "camera" | "file"

# When INPUT_MODE="camera", choose which device is default at startup.
# You can also toggle at runtime with KEY_TOGGLE_INPUT.
CAMERA_SOURCE_DEFAULT = "webcam"  # "webcam" | "capture_card"
WEBCAM_INDEX = 0
CAPTURE_CARD_INDEX = 1

# Backend hint for cv2.VideoCapture(index, backend_flag)
# Common options: "auto", "dshow", "msmf", "v4l2", "avfoundation"
CAPTURE_BACKEND = "auto"

SAVE_OUTPUT = False  # if True, writes annotated output video (requires consent if enabled below)
OUTPUT_VIDEO = "Segmentation_Aeroscapes.mp4"
OUTPUT_CODEC = "mp4v"

# Force the displayed/encoded/streamed frame size to 1080p (1920x1080).
# This does NOT require the camera itself to run at 1080p; frames are resized/letterboxed.
FORCE_OUTPUT_1080P = True
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_KEEP_ASPECT = True  # letterbox to preserve aspect ratio

# Best-effort: ask the camera for 1080p (many webcams/capture cards will honor this).
REQUEST_CAMERA_1080P = True

# -----------------------------
# Network streaming
# -----------------------------
ENABLE_RTSP = False
RTSP_URL = "rtsp://127.0.0.1:8554/stream"

ENABLE_UDP = True
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# If True, RTSP/UDP only run while RECORDING is enabled (mirrors consent gating).
REQUIRE_CONSENT_FOR_NETWORK = False

# Unity class-id mapping for UDP packets
UNITY_CLASS_ID = {
    "person": 0,
    "fire": 1,
    "smoke": 2,
    "chair": 3,
    "couch": 4,
    "sofa": 4,  # alias
    "dining table": 5,
}

# Only send detections with >= this confidence in the UDP JSON payload.
UDP_MIN_CONF = 0.80

# Which classes are allowed into the UDP JSON payload.
# (Set to None to send everything that passes UDP_MIN_CONF.)
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
DETECTION_LOG_PATH = "detections_log.json"  # merged detections output

# If True, only write output video/log when RECORDING is enabled at runtime (user consent)
REQUIRE_CONSENT_FOR_OUTPUT = True
REQUIRE_CONSENT_FOR_LOG = True

# -----------------------------
# Models
# -----------------------------
PEOPLE_MODEL_PATH = "../yolo11_models/yolo11n-seg.pt"              # instance segmentation model
FIRE_MODEL_PATH = "../yolo11_models/fire_smoke_detection.pt"       # fire/smoke model

# Classes to detect from the COCO-style model (YOLO11 COCO pretrained):
# COCO includes "chair", "couch", and "dining table". ("sofa" is a common synonym for "couch".)
DETECT_CLASSES = ("person", "chair", "couch", "dining table")

# Confidence thresholds passed into Ultralytics predict()
PEOPLE_CONF = 0.40
FIRE_CONF = 0.25

# Inference image size (higher can be more accurate but slower)
IMGSZ = 960

# -----------------------------
# Compute / Performance
# -----------------------------
DEVICE = 0 if torch.cuda.is_available() else "cpu"  # CUDA device index or "cpu"
USE_FP16 = bool(torch.cuda.is_available())          # half precision on CUDA

DEFAULT_FPS = 30.0  # used if capture reports invalid FPS
WINDOW_NAME = "Live Pipeline"

# -----------------------------
# Default runtime toggles
# -----------------------------
PEOPLE_ON_DEFAULT = True
FIRE_ON_DEFAULT = False
RECORDING_ENABLED_DEFAULT = False

# If False, no masks/boxes/labels are drawn, but inference + merge + UDP still run.
DRAW_DETECTIONS_DEFAULT = True

# -----------------------------
# Mask rendering
# -----------------------------
MASK_ALPHA = 0.35
MASK_TEXT_SCALE = 0.6
MASK_TEXT_THICKNESS = 2

# If True, attempts to attach mask arrays into the merged detection dicts (can bloat memory/log size)
ATTACH_PEOPLE_MASKS_TO_LOG = True
ATTACH_FIRE_MASKS_TO_LOG = True

# Overlay colors (BGR for OpenCV)
COLORS = {
    "person": (255, 255, 0), # Cyan
    "item": (255, 0, 0), # Blue
    "fire": (255, 0, 255), # Purple
    "smoke": (0, 255, 255), # Yellow
    "chair": (0, 200, 0), # Green
    "couch": (200, 0, 0), # Blue-ish
    "dining table": (0, 165, 255), # Orange-ish
}

# -----------------------------
# HUD
# -----------------------------
HUD_ENABLED_DEFAULT = True
HUD_ANCHOR = "lb"
HUD_MARGIN = 40 # space above bottom
HUD_ALPHA = 0.45
HUD_FONT_SCALE = 0.55
HUD_THICKNESS = 1

# -----------------------------
# DJI menu overlay (PNG on top of video)
# -----------------------------
# Loads this PNG (expects RGBA with alpha) and composites it over each frame.
# It is resized every frame to match the current frame dimensions.
DJI_MENU_OVERLAY_PATH = "DJImenu.png"
DJI_MENU_OVERLAY_ENABLED_DEFAULT = True

# -----------------------------
# Keybinds
# -----------------------------
KEY_ESC = 27
KEY_TOGGLE_RECORDING = (ord("r"), ord("R"))
KEY_TOGGLE_PEOPLE = (ord("k"), ord("K"))
KEY_TOGGLE_FIRE = (ord("l"), ord("L"))

# Toggle HUD on/off.
KEY_TOGGLE_HUD = (ord("h"), ord("H"))

# Toggle camera input (webcam <-> capture_card) while running (only when INPUT_MODE="camera")
KEY_TOGGLE_INPUT = (ord("i"), ord("I"))

# Toggle visual overlays (masks/boxes/labels). UDP unaffected.
KEY_TOGGLE_DRAW = (ord("v"), ord("V"))

# Toggle DJI overlay PNG.
KEY_TOGGLE_DJI_OVERLAY = (ord("u"), ord("U"))

# -----------------------------
# Test mode
# -----------------------------
# Used when running: python main.py -test
TEST_IMAGE_PATH = "test.jpg"