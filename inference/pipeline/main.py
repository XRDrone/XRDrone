"""
main.py

CLI entrypoint for the pure ArUco XRDrone local inference pipeline.
"""

from __future__ import annotations

import argparse

import settings as S
from live_runner import run_live


def parse_args():
    parser = argparse.ArgumentParser(description="XRDrone pure ArUco prerecorded-video pipeline")
    parser.add_argument(
        "-test",
        "--test",
        action="store_true",
        help="Run the existing single-image detector test path.",
    )
    parser.add_argument(
        "--test-image",
        default=S.TEST_IMAGE_PATH,
        help="Path to the image used with -test.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Path to a prerecorded video file. Example: .\\2026_05_18_15_28_04_Cache_Trimmed.mp4",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional YOLO model path. Defaults to settings.PEOPLE_MODEL_PATH.",
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Disable YOLO detections and run ArUco pose only.",
    )
    parser.add_argument(
        "--logs",
        "--log",
        action="store_true",
        help="Enable per-run log folder creation. By default, python main.py runs normally without logs.",
    )
    parser.add_argument(
        "--log-root",
        default=S.LOG_ROOT,
        help="Directory where per-run log folders are created when --logs is used.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for smoke tests.",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save annotated output video.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output video path when --save-output is used.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable OpenCV imshow window. Use this for headless/log-only runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.test:
        from test_runner import run_test

        return run_test(args)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
