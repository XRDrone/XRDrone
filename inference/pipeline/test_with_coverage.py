"""
test_with_coverage.py

UDP contract tests for the XRDrone pipeline.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Any

import settings as S
from output_formatter import to_unity_udp_packet
from streaming import UDPPublisher

EXPECTED_TOP_LEVEL_KEYS = {
    "frame_id",
    "timestamp",
    "width",
    "height",
    "detections",
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
}


def _assert_exact_keys(obj: dict[str, Any], expected: set[str], label: str) -> None:
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
    if not isinstance(value, int | float):
        raise AssertionError(f"{label} must be numeric")
    value_f = float(value)
    if value_f < lo or value_f > hi:
        raise AssertionError(f"{label} out of range [{lo}, {hi}]: {value}")


def _validate_top_level(pkt: dict[str, Any]) -> None:
    _assert_exact_keys(pkt, EXPECTED_TOP_LEVEL_KEYS, "top-level packet")

    _assert_type(pkt["frame_id"], int, "frame_id")
    _assert_type(pkt["timestamp"], (int, float), "timestamp")
    _assert_type(pkt["width"], int, "width")
    _assert_type(pkt["height"], int, "height")
    _assert_type(pkt["detections"], list, "detections")

    if pkt["width"] <= 0 or pkt["height"] <= 0:
        raise AssertionError("width and height must be positive")


def _validate_detection(det: dict[str, Any], index: int) -> None:
    prefix = f"detections[{index}]"
    _assert_exact_keys(det, EXPECTED_DETECTION_KEYS, prefix)

    _assert_type(det["id"], int, f"{prefix}.id")
    _assert_type(det["cls"], int, f"{prefix}.cls")
    _assert_type(det["conf"], (int, float), f"{prefix}.conf")

    _assert_number_range(det["conf"], 0.0, 1.0, f"{prefix}.conf")

    for key in ("cx", "cy", "w", "h", "foot_x", "foot_y"):
        _assert_number_range(det[key], 0.0, 1.0, f"{prefix}.{key}")


def _validate_packet(pkt: dict[str, Any]) -> None:
    _validate_top_level(pkt)

    for i, det in enumerate(pkt["detections"]):
        if not isinstance(det, dict):
            raise AssertionError(f"detections[{i}] must be an object")
        _validate_detection(det, i)


def _build_sample_packet() -> dict[str, Any]:
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
        }
    ]

    return to_unity_udp_packet(
        merged_detections,
        frame_id=1,
        timestamp=1234567890.123,
        width=width,
        height=height,
        class_map=S.UNITY_CLASS_ID,
        allowed_classes=S.UDP_SEND_CLASSES,
        min_conf=S.UDP_MIN_CONF,
    )


def _recv_json_packet(sock: socket.socket, timeout_s: float) -> dict[str, Any]:
    pkt, _data, _arrival = _recv_json_packet_with_meta(sock, timeout_s)
    return pkt


def _recv_json_packet_with_meta(
    sock: socket.socket,
    timeout_s: float,
) -> tuple[dict[str, Any], bytes, float]:
    sock.settimeout(float(timeout_s))
    data, _addr = sock.recvfrom(65535)
    arrival_ts = time.time()

    try:
        text = data.decode("utf-8")
    except Exception as e:
        raise AssertionError(f"UDP payload is not valid UTF-8 JSON: {e}") from e

    try:
        obj = json.loads(text)
    except Exception as e:
        raise AssertionError(f"UDP payload is not valid JSON: {e}") from e

    if not isinstance(obj, dict):
        raise AssertionError("UDP payload JSON root must be an object")

    return obj, data, arrival_ts


def _test_formatter_schema() -> None:
    pkt = _build_sample_packet()
    _validate_packet(pkt)


def _test_udp_loopback() -> None:
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind((S.UDP_IP, 0))
    recv_host, recv_port = recv_sock.getsockname()

    pub = UDPPublisher(recv_host, recv_port)
    try:
        pkt = _build_sample_packet()
        pub.send_json(pkt)
        got = _recv_json_packet(recv_sock, timeout_s=2.0)
        _validate_packet(got)
    finally:
        pub.close()
        recv_sock.close()


def _validate_live_packet_stats(pkt: dict[str, Any]) -> tuple[int, int]:
    _validate_packet(pkt)
    num_dets = len(pkt["detections"])
    packet_size = len(json.dumps(pkt, separators=(",", ":")).encode("utf-8"))
    return num_dets, packet_size


def _run_live_listener(
    host: str, port: int, packets: int, timeout_s: float
) -> list[dict[str, Any]]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(0.2)

    deadline = time.time() + float(timeout_s)
    received: list[dict[str, Any]] = []
    try:
        while len(received) < int(packets) and time.time() < deadline:
            try:
                pkt = _recv_json_packet(sock, timeout_s=0.2)
            except TimeoutError:
                continue
            _validate_packet(pkt)
            received.append(pkt)
    finally:
        sock.close()
    return received


def _stats_summary(packets: list[dict[str, Any]]) -> dict[str, Any]:
    sizes = [len(json.dumps(pkt, separators=(",", ":")).encode("utf-8")) for pkt in packets]
    det_counts = [len(pkt["detections"]) for pkt in packets]
    return {
        "packets": len(packets),
        "avg_packet_size": sum(sizes) / len(sizes) if sizes else 0.0,
        "min_packet_size": min(sizes) if sizes else 0,
        "max_packet_size": max(sizes) if sizes else 0,
        "avg_detections": sum(det_counts) / len(det_counts) if det_counts else 0.0,
    }


def _run_pipeline_for_video(video_path: str, timeout_s: float) -> list[dict[str, Any]]:
    import subprocess
    import sys

    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind((S.UDP_IP, 0))
    host, port = recv_sock.getsockname()
    recv_sock.close()

    cmd = [
        sys.executable,
        "main.py",
        "--no-gui",
    ]
    env = None
    process = subprocess.Popen(cmd, env=env)
    try:
        return _run_live_listener(host, port, packets=30, timeout_s=timeout_s)
    finally:
        process.terminate()
        process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the XRDrone UDP contract and transport.")
    parser.add_argument(
        "--live", action="store_true", help="Listen for packets from a running pipeline."
    )
    parser.add_argument(
        "--stats", action="store_true", help="Print summary stats after validation."
    )
    parser.add_argument(
        "--packets", type=int, default=5, help="Number of live packets to validate."
    )
    parser.add_argument(
        "--timeout", type=float, default=8.0, help="Live listener timeout in seconds."
    )
    parser.add_argument("--video", default="", help="Optional video path used with --stats.")
    args = parser.parse_args()

    try:
        _test_formatter_schema()
        _test_udp_loopback()

        live_packets: list[dict[str, Any]] = []
        if args.live:
            live_packets = _run_live_listener(S.UDP_IP, S.UDP_PORT, args.packets, args.timeout)
            if len(live_packets) < args.packets:
                raise AssertionError(
                    f"Only received {len(live_packets)} valid live packets before timeout"
                )

        if args.stats:
            packets = live_packets
            if not packets and args.video:
                packets = _run_pipeline_for_video(args.video, args.timeout)
            summary = _stats_summary(packets)
            print(json.dumps(summary, indent=2, sort_keys=True))

        print("PASS")
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
