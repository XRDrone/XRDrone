"""
main.py

Thin CLI entrypoint for the XRDrone local inference pipeline.
"""

from __future__ import annotations

import argparse

import settings as S
from live_runner import run_live
from test_runner import run_test


def parse_args():
    parser = argparse.ArgumentParser(description="XRDrone local YOLO pipeline")
    parser.add_argument(
        "-test",
        "--test",
        action="store_true",
        help="Run a single-image test and print the UDP JSON payload.",
    )
    parser.add_argument(
        "--test-image",
        default=S.TEST_IMAGE_PATH,
        help="Path to the image used with -test.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable OpenCV imshow window (useful for headless runs).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.test:
        return run_test(args)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
