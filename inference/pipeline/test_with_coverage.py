"""
test_with_coverage.py

UDP contract tests for the XRDrone pipeline.

What this checks:
  1) Formatter/schema test:
     Builds a sample packet and validates that it matches the README UDP contract.

  2) UDP loopback transport test:
     Sends a real UDP packet through UDPPublisher, receives it on localhost,
     parses it back from JSON, and validates the received packet.

  3) Optional live packet test:
     Listens on a UDP port for packets from a running pipeline (main.py) and
     validates that live runtime packets match the same README contract.

Success behavior:
  - Prints exactly one PASS line on success
  - Prints exactly one FAIL line on failure
  - Exits with code 0 on success, 1 on failure

Usage:
  python test_with_coverage.py
      Runs formatter/schema + real UDP loopback send/receive.

  python test_with_coverage.py --live
      Runs formatter/schema + loopback + live listener validation.
      Start main.py in another terminal before running this mode.

  python test_with_coverage.py --live --packets 5 --timeout 8
      Wait for 5 valid live packets for up to 8 seconds.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from typing import Any, Dict, Iterable, Tuple

import settings as S
from output_formatter import to_unity_udp_packet
from streaming import UDPPublisher


EXPECTED_TOP_LEVEL_KEYS = {
    "frame_id",
    "timestamp",
    "width",
    "height",
    "detections",
    "pose",
}

EXPECTED_DETECTION_KEYS = {
    "id",
    "cls",
    "conf",
    "cx",
    "cy",
    "w",
    "h",
    "foot_x",
    "foot_y",
    "world_valid",
    "world_x",
    "world_y",
    "world_z",
}

EXPECTED_POSE_KEYS = {
    "x",
    "altitude",
    "z",
    "yaw",
    "pitch",
    "roll",
    "hfov",
    "markers_used",
    "pose_valid",
}


def _assert_exact_keys(obj: Dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(obj.keys())
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if extra:
            parts.append(f"extra={extra}")
        raise AssertionError(f"{label} keys mismatch: " + ", ".join(parts))


def _assert_type(value: Any, types: Any, label: str) -> None:
    if not isinstance(value, types):
        raise AssertionError(f"{label} has wrong type: got {type(value).__name__}")


def _assert_number_range(value: Any, lo: float, hi: float, label: str) -> None:
    if not isinstance(value, (int, float)):
        raise AssertionError(f"{label} must be numeric")
    value_f = float(value)
    if value_f < lo or value_f > hi:
        raise AssertionError(f"{label} out of range [{lo}, {hi}]: {value}")


def _validate_top_level(pkt: Dict[str, Any]) -> None:
    _assert_exact_keys(pkt, EXPECTED_TOP_LEVEL_KEYS, "top-level packet")

    _assert_type(pkt["frame_id"], int, "frame_id")
    _assert_type(pkt["timestamp"], (int, float), "timestamp")
    _assert_type(pkt["width"], int, "width")
    _assert_type(pkt["height"], int, "height")
    _assert_type(pkt["detections"], list, "detections")
    _assert_type(pkt["pose"], dict, "pose")

    if pkt["width"] <= 0 or pkt["height"] <= 0:
        raise AssertionError("width and height must be positive")


def _validate_detection(det: Dict[str, Any], index: int) -> None:
    prefix = f"detections[{index}]"
    _assert_exact_keys(det, EXPECTED_DETECTION_KEYS, prefix)

    _assert_type(det["id"], int, f"{prefix}.id")
    _assert_type(det["cls"], int, f"{prefix}.cls")
    _assert_type(det["conf"], (int, float), f"{prefix}.conf")
    _assert_type(det["world_valid"], bool, f"{prefix}.world_valid")

    _assert_number_range(det["conf"], 0.0, 1.0, f"{prefix}.conf")

    for key in ("cx", "cy", "w", "h", "foot_x", "foot_y"):
        _assert_number_range(det[key], 0.0, 1.0, f"{prefix}.{key}")

    for key in ("world_x", "world_y", "world_z"):
        _assert_type(det[key], (int, float), f"{prefix}.{key}")


def _validate_pose(pose: Dict[str, Any]) -> None:
    _assert_exact_keys(pose, EXPECTED_POSE_KEYS, "pose")

    for key in ("x", "altitude", "z", "yaw", "pitch", "roll", "hfov"):
        _assert_type(pose[key], (int, float), f"pose.{key}")

    _assert_type(pose["markers_used"], int, "pose.markers_used")
    _assert_type(pose["pose_valid"], bool, "pose.pose_valid")

    if pose["markers_used"] < 0:
        raise AssertionError("pose.markers_used must be non-negative")


def _validate_packet(pkt: Dict[str, Any]) -> None:
    _validate_top_level(pkt)

    for i, det in enumerate(pkt["detections"]):
        if not isinstance(det, dict):
            raise AssertionError(f"detections[{i}] must be an object")
        _validate_detection(det, i)

    _validate_pose(pkt["pose"])


def _build_sample_packet() -> Dict[str, Any]:
    width = 1920
    height = 1080

    merged_detections = [
        {
            "class": "person",
            "confidence": 0.91,
            "bbox_xyxy": [100.0, 200.0, 300.0, 800.0],
            "track_id": 7,
            "foot_x": 200.0 / width,
            "foot_y": 800.0 / height,
            "world_valid": False,
            "world_x": 0.0,
            "world_y": 0.0,
            "world_z": 0.0,
        }
    ]

    pkt = to_unity_udp_packet(
        merged_detections,
        frame_id=1,
        timestamp=1234567890.123,
        width=width,
        height=height,
        class_map=S.UNITY_CLASS_ID,
        allowed_classes=S.UDP_SEND_CLASSES,
        min_conf=S.UDP_MIN_CONF,
    )

    pkt["pose"] = {
        "x": 0.0,
        "altitude": 0.0,
        "z": 0.0,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "hfov": float(S.POSE_HFOV_DEG),
        "markers_used": 0,
        "pose_valid": False,
    }

    return pkt


def _recv_json_packet(sock: socket.socket, timeout_s: float) -> Dict[str, Any]:
    sock.settimeout(float(timeout_s))
    data, _addr = sock.recvfrom(65535)

    try:
        text = data.decode("utf-8")
    except Exception as e:
        raise AssertionError(f"UDP payload is not valid UTF-8 JSON: {e}")

    try:
        obj = json.loads(text)
    except Exception as e:
        raise AssertionError(f"UDP payload is not valid JSON: {e}")

    if not isinstance(obj, dict):
        raise AssertionError("UDP payload JSON root must be an object")

    return obj


def _test_formatter_schema() -> None:
    pkt = _build_sample_packet()
    _validate_packet(pkt)


def _test_udp_loopback(timeout_s: float) -> None:
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind(("127.0.0.1", 0))
    recv_ip, recv_port = recv_sock.getsockname()

    publisher = UDPPublisher(recv_ip, recv_port)

    try:
        expected_pkt = _build_sample_packet()
        publisher.send_json(expected_pkt)

        received_pkt = _recv_json_packet(recv_sock, timeout_s=timeout_s)
        _validate_packet(received_pkt)

        if received_pkt != expected_pkt:
            raise AssertionError("received UDP packet does not match sent packet")
    finally:
        try:
            publisher.close()
        except Exception:
            pass
        try:
            recv_sock.close()
        except Exception:
            pass


def _iter_valid_live_packets(
    host: str,
    port: int,
    packets_needed: int,
    timeout_s: float,
) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, int(port)))
        sock.settimeout(float(timeout_s))

        valid_count = 0
        while valid_count < packets_needed:
            pkt = _recv_json_packet(sock, timeout_s=timeout_s)
            _validate_packet(pkt)
            valid_count += 1

        return valid_count
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _test_live_udp_packets(
    host: str,
    port: int,
    packets_needed: int,
    timeout_s: float,
) -> None:
    valid_count = _iter_valid_live_packets(
        host=host,
        port=port,
        packets_needed=packets_needed,
        timeout_s=timeout_s,
    )
    if valid_count < packets_needed:
        raise AssertionError(
            f"only received {valid_count} valid live packets, expected {packets_needed}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate XRDrone UDP packet structure and UDP transport."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also listen for live packets from a running main.py process.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/interface to bind for live UDP listening.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(S.UDP_PORT),
        help="UDP port to bind for live UDP listening.",
    )
    parser.add_argument(
        "--packets",
        type=int,
        default=3,
        help="Number of valid live packets required for --live.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Socket timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    try:
        _test_formatter_schema()
        _test_udp_loopback(timeout_s=args.timeout)

        if args.live:
            _test_live_udp_packets(
                host=args.host,
                port=args.port,
                packets_needed=args.packets,
                timeout_s=args.timeout,
            )
            print("PASSED: live UDP transport and README packet structure are valid")
        else:
            print("PASSED: UDP formatter structure and UDP send/receive are valid")

        return 0

    except Exception as e:
        if args.live:
            print(f"FAILED: live UDP transport or README packet structure is invalid: {e}")
        else:
            print(f"FAILED: UDP formatter structure or UDP send/receive is invalid: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())