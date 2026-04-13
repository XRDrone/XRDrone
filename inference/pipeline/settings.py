"""
XRDrone local pipeline settings.

This file centralizes all runtime configuration for the local YOLO
inference demo (video source, model paths, thresholds, HUD, logging, keys,
and network streaming).
Edit tunable values here; fixed optimized pipeline policy is intentionally hard-coded in main.py.
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

UDP_SEND_CLASSES = ("person",)

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

# -----------------------------
# Serialization / transport
# -----------------------------
# Keep the UDP JSON schema identical, but allow a faster encoder path.
# "orjson" -> use orjson when installed, with stdlib compact JSON fallback.
# "json"   -> always use stdlib compact JSON.
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

# Tracking is now fixed in the pipeline to use the optimized Ultralytics
# tracking path. These remaining values only affect downstream formatting.
TRACK_ID_OFFSET_PEOPLE = 0
TRACK_ID_OFFSET_FIRE = 1_000_000
DRAW_TRACK_IDS = True

DRAW_DETECTIONS_DEFAULT = True

# -----------------------------
# Robust mitigation of Object-ID flicker in UDP JSON streams
# -----------------------------
# The continuity layer itself is fixed on in the pipeline.
# These remain as tuning controls for the hard-coded path.
ID_FLICKER_APPLY_CLASSES = ("person",)

# Confidence gate:
#   new IDs must reach tau_on
#   existing emitted IDs can persist down to tau_off
ID_FLICKER_EMA_ALPHA = 0.45
ID_FLICKER_TAU_ON = 0.80
ID_FLICKER_TAU_OFF = 0.55

# Keep emitting a recently seen ID for a short time even when detections dip
# or one/few frames are missed.
ID_FLICKER_COAST_FRAMES = 6
ID_FLICKER_DROP_FRAMES = 45
ID_FLICKER_COAST_CONF_DECAY = 0.985

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
POSE_MARKER_WORLD_POSITIONS = {0: (0.0, 0.0, 0.0)}

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

# Minimum known visible markers required to enter the multi-marker board path.
POSE_MIN_MARKERS_FOR_MULTI = 2

# Optional ArUco detector corner refinement.
POSE_CORNER_REFINEMENT = "apriltag"  # "none" | "subpix" | "contour" | "apriltag"

# RANSAC tuning for the multi-marker initializer (and optional single-marker fallback).
POSE_RANSAC_REPROJ_THRESHOLD_PX = 4.0
POSE_RANSAC_CONFIDENCE = 0.99
POSE_RANSAC_ITERATIONS = 100

# Explicit pose-loss fallback when no known marker corners are visible is fixed on in the pipeline.
# During the short hold timeout, the last valid pose numbers are preserved,
# but pose_valid remains False and registration stays unavailable.
POSE_LOSS_HOLD_TIMEOUT_S = 0.35
POSE_LOSS_PRESERVE_LAST_NUMBERS_DURING_HOLD = True
POSE_LOSS_CLEAR_NUMBERS_AFTER_TIMEOUT = True

# If True, draw detected ArUco markers on the output frame.
POSE_DRAW_ARUCO = False

# If True, draw a small status label showing whether the frame currently has
# no known markers, a single marker, or multiple markers.
POSE_MODE_OVERLAY_ENABLED_DEFAULT = True
POSE_MODE_OVERLAY_ORIGIN = (20, 40)
POSE_MODE_OVERLAY_TEXT_SCALE = 0.9
POSE_MODE_OVERLAY_TEXT_THICKNESS = 2


# -----------------------------
# ORB-SLAM fusion middle-man
# -----------------------------
# When enabled, the live runtime listens for externally generated ORB-SLAM
# pose packets over UDP, aligns them to detector frames, projects foot points
# onto a ground plane, and publishes the fused result to Unity.
ORBSLAM_FUSION_ENABLED = True
ORBSLAM_UDP_LISTEN_IP = "127.0.0.1"
ORBSLAM_UDP_PORT = 5010
ORBSLAM_UDP_MAX_PACKET_BYTES = 65535
ORBSLAM_PACKET_STALE_TIMEOUT_S = 0.50
ORBSLAM_MATCH_TIME_TOLERANCE_S = 0.10
ORBSLAM_POSE_BUFFER_SIZE = 4096
ORBSLAM_GROUND_PLANE_Y = 0.0
ORBSLAM_STATUS_OVERLAY_ENABLED = True
ORBSLAM_STATUS_OVERLAY_ORIGIN = (20, 72)
ORBSLAM_STATUS_OVERLAY_TEXT_SCALE = 0.65
ORBSLAM_STATUS_OVERLAY_TEXT_THICKNESS = 2

# -----------------------------
# Motion smoothing (ArUco-based object registration)
# -----------------------------
# Single 0..1 slider used by both layers:
#   0.0 = raw / most responsive
#   1.0 = smoothest / most damped
MOTION_SMOOTHING = 0.50
MOTION_SMOOTHING_STEP = 0.05
MOTION_SMOOTHING_DERIVATIVE_CUTOFF_HZ = 1.0
MOTION_SMOOTHING_RESET_TIMEOUT_S = 0.75

# Motion smoothing is fixed on in the pipeline for both pose and world-space filtering.
WORLD_MOTION_SMOOTHING_MAX_TRACK_AGE_S = 1.50

# -----------------------------
# Adaptive runtime tuning
# -----------------------------
# Bounded runtime adaptation for smoothing + ID flicker mitigation only.
# This does NOT change physical marker layout, Unity class mapping, the UDP schema,
# or the structural ArUco solver-selection policy.
ADAPTIVE_TUNING_ENABLED = True
ADAPTIVE_TUNING_LOG_UPDATES = True
ADAPTIVE_TUNING_TARGET_CLASSES = ("person",)
ADAPTIVE_TUNING_WINDOW_FRAMES = 45
ADAPTIVE_TUNING_UPDATE_INTERVAL_FRAMES = 15
ADAPTIVE_TUNING_COOLDOWN_FRAMES = 30
ADAPTIVE_TUNING_IOU_MATCH_THRESHOLD = 0.35

ADAPTIVE_MOTION_SMOOTHING_MIN = 0.30
ADAPTIVE_MOTION_SMOOTHING_MAX = 0.85
ADAPTIVE_MOTION_SMOOTHING_STEP = 0.05

ADAPTIVE_ID_TAU_ON_MIN = 0.75
ADAPTIVE_ID_TAU_ON_MAX = 0.90
ADAPTIVE_ID_TAU_OFF_MIN = 0.45
ADAPTIVE_ID_TAU_OFF_MAX = 0.65
ADAPTIVE_ID_TAU_STEP = 0.02
ADAPTIVE_ID_COAST_FRAMES_MIN = 3
ADAPTIVE_ID_COAST_FRAMES_MAX = 10
ADAPTIVE_ID_COAST_STEP = 1

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
KEY_TOGGLE_POSE_MODE_OVERLAY = (ord("m"), ord("M"))
KEY_TOGGLE_MOTION_SMOOTHING = (ord("g"), ord("G"))
KEY_DECREASE_MOTION_SMOOTHING = (ord("["), ord("{"))
KEY_INCREASE_MOTION_SMOOTHING = (ord("]"), ord("}"))

# -----------------------------
# Test mode
# -----------------------------
TEST_IMAGE_PATH = "/Users/troy/Desktop/XRDrone/models/yolo_people_fire_smoke/people_furniture.avif"
