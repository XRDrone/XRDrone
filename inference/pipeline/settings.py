"""
XRDrone local pipeline settings.

This file centralizes all runtime configuration for the local YOLO
inference demo (video source, model paths, thresholds, HUD, logging, keys,
and network streaming).
"""

from __future__ import annotations

import torch

# -----------------------------
# Input / Output
# -----------------------------
VIDEO_PATH = "/Users/troy/Desktop/XRDrone/inference/pipeline/input_test.mp4"
VIDEO_SOURCE = 0

INPUT_MODE = "camera"  # "camera" | "file"

CAMERA_SOURCE_DEFAULT = "webcam"  # "webcam" | "capture_card"
WEBCAM_INDEX = 0
CAPTURE_CARD_INDEX = 1

CAPTURE_BACKEND = "auto"

SAVE_OUTPUT = False
OUTPUT_VIDEO = "/Users/troy/Desktop/XRDrone/inference/pipeline/output.mp4"
OUTPUT_CODEC = "mp4v"

FORCE_OUTPUT_1080P = True
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_KEEP_ASPECT = True
REQUEST_CAMERA_1080P = True

# -----------------------------
# UDP output
# -----------------------------
ENABLE_UDP = True
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

UNITY_CLASS_ID = {
    "person": 0,
}

UDP_MIN_CONF = 0.80
UDP_SEND_CLASSES = ("person",)

# -----------------------------
# Logging
# -----------------------------
DETECTION_LOG_PATH = "detections_log.json"

# -----------------------------
# Models
# -----------------------------
PEOPLE_MODEL_PATH = "../models/yolo26n-seg.pt"
FIRE_MODEL_PATH = "../models/fire_smoke_detection.pt"

DETECT_CLASSES = ("person",)

PEOPLE_CONF = 0.40
FIRE_CONF = 0.25

IMGSZ = 960

# -----------------------------
# Compute / Performance
# -----------------------------
DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_FP16 = bool(torch.cuda.is_available())

# -----------------------------
# Serialization / transport
# -----------------------------
JSON_SERIALIZER = "orjson"  # "orjson" | "json"
JSON_ENSURE_ASCII = False

DEFAULT_FPS = 30.0
WINDOW_NAME = "Live Pipeline"

# -----------------------------
# Default runtime toggles
# -----------------------------
PEOPLE_ON_DEFAULT = True
FIRE_ON_DEFAULT = False
RECORDING_ENABLED_DEFAULT = False

TRACK_ID_OFFSET_PEOPLE = 0
TRACK_ID_OFFSET_FIRE = 1_000_000
DRAW_TRACK_IDS = True
DRAW_DETECTIONS_DEFAULT = True

# -----------------------------
# Robust mitigation of Object-ID flicker in UDP JSON streams
# -----------------------------
ID_FLICKER_APPLY_CLASSES = ("person",)
ID_FLICKER_EMA_ALPHA = 0.45
ID_FLICKER_TAU_ON = 0.80
ID_FLICKER_TAU_OFF = 0.55
ID_FLICKER_COAST_FRAMES = 6
ID_FLICKER_DROP_FRAMES = 45
ID_FLICKER_COAST_CONF_DECAY = 0.985

# -----------------------------
# Mask rendering
# -----------------------------
MASK_ALPHA = 0.35
MASK_TEXT_SCALE = 0.6
MASK_TEXT_THICKNESS = 2

COLORS = {
    "person": (255, 255, 0),
}

# -----------------------------
# DJI menu overlay (PNG on top of video)
# -----------------------------
DJI_MENU_OVERLAY_PATH = "DJImenu.png"
DJI_MENU_OVERLAY_ENABLED_DEFAULT = False

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
