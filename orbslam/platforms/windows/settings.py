"""
settings.py

Runtime constants for the Windows-side XRDrone ORB-SLAM3 fusion pipeline.
Edit this file for local machine paths, ports, RTSP source, model path, and
logging defaults. Command-line flags in rtsp_yolo_orbslam_fusion.py can still
override these values at runtime.
"""

from __future__ import annotations

# =============================================================================
# Input stream and model
# =============================================================================
RTSP_URL = "rtsp://192.168.1.4:8554/dji"
YOLO_MODEL_PATH = r"C:\Users\students\Desktop\pose_receiver\yolov5nu.pt"

# =============================================================================
# ORB-SLAM3 pose UDP input
# =============================================================================
# Use 0.0.0.0 so Windows can listen even if its Wi-Fi IP changes.
POSE_LISTEN_IP = "0.0.0.0"
POSE_PORT = 5005
POSE_FORMAT = "orbslam-text"  # "auto" | "orbslam-text" | "json"

MAX_POSE_AGE_SECONDS = 1.0
POSE_SCALE = 1.0
SWAP_YZ = False
INVERT_X = False
INVERT_Y = False
INVERT_Z = False

# =============================================================================
# Unity UDP output
# =============================================================================
UNITY_OUTPUT_HOST = "127.0.0.1"
UNITY_OUTPUT_PORT = 6000

# =============================================================================
# Display and inference
# =============================================================================
SHOW_WINDOW = True
DEVICE = 0  # Use 0 for CUDA GPU 0, or "cpu" for CPU-only.
IMG_SIZE = 640
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.70
PERSON_CLASS_ID = 0
USE_YOLO_TRACKING = True

# =============================================================================
# Projection settings
# =============================================================================
DEFAULT_HFOV_DEG = 70.0
DEFAULT_CAMERA_HEIGHT_M = 1.5
GROUND_Y = 0.0
POSE_ANGLES_IN_DEGREES = False

YAW_OFFSET_DEG = 0.0
PITCH_OFFSET_DEG = 0.0
ROLL_OFFSET_DEG = 0.0

# =============================================================================
# RTSP latency and reconnect behavior
# =============================================================================
RECONNECT_DELAY_SECONDS = 2.0
DROP_STALE_GRABS = 3
RTSP_CAPTURE_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

# =============================================================================
# Runtime logging
# =============================================================================
LOGS_ENABLED_DEFAULT = False
WAIT_FOR_VIDEO_DEFAULT = False
LOG_ROOT = "logs"
