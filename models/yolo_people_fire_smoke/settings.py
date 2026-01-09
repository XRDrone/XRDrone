# settings.py
"""
XRDrone local pipeline settings.

This file centralizes all runtime configuration for the local YOLO
inference demo (video source, model paths, thresholds, HUD, logging, and keys).
Edit values here; main.py should not contain hard-coded settings.
"""

from __future__ import annotations
import torch

# -----------------------------
# Input / Output
# -----------------------------
VIDEO_PATH = r"E:\Detection_Segmentation_Demo.mp4"  # file path if using video file input
VIDEO_SOURCE = 0  # set to VIDEO_PATH to run on a file, or keep as camera index (e.g., 0)

SAVE_OUTPUT = False  # if True, writes annotated output video (requires consent if enabled below)
OUTPUT_VIDEO = "Segmentation_Aeroscapes.mp4"
OUTPUT_CODEC = "mp4v"

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
FIRE_MODEL_PATH = "../yolo11_models/fire_smoke_detection.pt"       # fire/smoke model (det/seg depending on weights)

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

# -----------------------------
# Mask rendering
# -----------------------------
MASK_ALPHA = 0.35
MASK_TEXT_SCALE = 0.6
MASK_TEXT_THICKNESS = 2

# If True, attempts to attach mask arrays into the merged detection dicts (can bloat logs)
ATTACH_PEOPLE_MASKS_TO_LOG = True
ATTACH_FIRE_MASKS_TO_LOG = True

# Overlay colors (BGR for OpenCV)
COLORS = {
    "person": (255, 0, 0),
    "item": (255, 0, 0),
    "fire": (255, 0, 255),
    "smoke": (0, 255, 255),
}

# -----------------------------
# HUD
# -----------------------------
HUD_ANCHOR = "tl"
HUD_MARGIN = 10
HUD_ALPHA = 0.45
HUD_FONT_SCALE = 0.55
HUD_THICKNESS = 1

# -----------------------------
# Keybinds
# -----------------------------
KEY_ESC = 27
KEY_TOGGLE_RECORDING = (ord("r"), ord("R"))
KEY_TOGGLE_PEOPLE = (ord("k"), ord("K"))
KEY_TOGGLE_FIRE = (ord("l"), ord("L"))
