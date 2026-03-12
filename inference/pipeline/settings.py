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
VIDEO_PATH = "/Users/troy/Desktop/XRDrone/inference/pipeline/ArUco test.mp4"
VIDEO_SOURCE = 0

INPUT_MODE = "camera"  # "camera" | "file"

CAMERA_SOURCE_DEFAULT = "webcam"  # "webcam" | "capture_card"
WEBCAM_INDEX = 0
CAPTURE_CARD_INDEX = 1

CAPTURE_BACKEND = "auto"

SAVE_OUTPUT = False
OUTPUT_VIDEO = "/Users/troy/Desktop/XRDrone/inference/pipeline/ArUco Output.mp4"
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

UDP_SEND_CLASSES = (
    "person",
)

# -----------------------------
# Logging
# -----------------------------
DETECTION_LOG_PATH = "detections_log.json"

# -----------------------------
# Models
# -----------------------------
# NOTE: In your repo structure:
#   inference/models/*.pt
#   inference/pipeline/*.py
# So paths from pipeline/ should be ../models/<file>.pt
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
# Pose estimation (camera pose via ArUco)
# -----------------------------
# Adds a top-level "pose" object to the UDP JSON packet.
# If pose cannot be computed, pose_valid will be False and markers_used will be 0.
POSE_ENABLED_DEFAULT = True

# Horizontal field-of-view used to approximate intrinsics (demo-quality).
POSE_HFOV_DEG = 84.0

# Physical marker size (meters). Must match your printed ArUco markers.
POSE_MARKER_SIZE_M = 0.1645

# OpenCV ArUco dictionary name (string constant under cv2.aruco).
POSE_ARUCO_DICT = "DICT_4X4_50"

# Marker world positions in meters (origin at marker id 0 by default).
# Each value is (x, y, z). The pose solver assumes markers lie on the Y=0 plane.
POSE_MARKER_WORLD_POSITIONS = {
    0: (0.0, 0.0, 0.0)
}

# Solver policy:
#   - "auto": use single-marker when only one known marker is visible,
#              otherwise use the multi-marker board solve.
#   - "single_marker": always use the single-marker path.
#   - "multi_marker_board": always try the joint multi-marker path.
POSE_USE_CASE = "auto"  # "auto" | "single_marker" | "multi_marker_board"

# Initial pose solver for one visible marker.
POSE_SINGLE_INIT_SOLVER = "ippe_square"  # "ippe_square" | "iterative" | "ransac"

# Initial pose solver when multiple fixed markers are visible together.
POSE_MULTI_INIT_SOLVER = "sqpnp"  # "sqpnp" | "ransac" | "iterative" | "ippe_square"

# Nonlinear refinement run after the initializer.
POSE_REFINER = "vvs"  # "vvs" | "lm" | "none"
POSE_ENABLE_REFINEMENT = True

# Minimum known visible markers required to enter the multi-marker board path.
POSE_MIN_MARKERS_FOR_MULTI = 2

# Optional ArUco detector corner refinement.
POSE_CORNER_REFINEMENT = "none"  # "none" | "subpix" | "contour" | "apriltag"

# RANSAC tuning for the multi-marker initializer (and optional single-marker fallback).
POSE_RANSAC_REPROJ_THRESHOLD_PX = 4.0
POSE_RANSAC_CONFIDENCE = 0.99
POSE_RANSAC_ITERATIONS = 100

# If True, draw detected ArUco markers on the output frame.
POSE_DRAW_ARUCO = False

# If True, draw a small status label showing whether the frame currently has
# no known markers, a single marker, or multiple markers.
POSE_MODE_OVERLAY_ENABLED_DEFAULT = True
POSE_MODE_OVERLAY_ORIGIN = (20, 40)
POSE_MODE_OVERLAY_TEXT_SCALE = 0.9
POSE_MODE_OVERLAY_TEXT_THICKNESS = 2

# -----------------------------
# Mask rendering
# -----------------------------
MASK_ALPHA = 0.35
MASK_TEXT_SCALE = 0.6
MASK_TEXT_THICKNESS = 2

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
KEY_TOGGLE_POSE_MODE_OVERLAY = (ord("m"), ord("M"))

# -----------------------------
# Test mode
# -----------------------------
TEST_IMAGE_PATH = "/Users/troy/Desktop/XRDrone/models/yolo_people_fire_smoke/people_furniture.avif"