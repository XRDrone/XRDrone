"""
test_with_coverage.py

UDP contract tests for the XRDrone pipeline.

What this checks:
  1) Formatter/schema test:
     Builds a sample packet and validates that it matches docs/udp-json.md.

  2) UDP loopback transport test:
     Sends a real UDP packet through UDPPublisher, receives it on localhost,
     parses it back from JSON, and validates the received packet.

  3) Optional live packet test:
     Listens on a UDP port for packets from a running pipeline (main.py) and
     validates that live runtime packets match the same UDP contract.

  4) Optional stats mode:
     Collects simple packet/runtime stats from a running pipeline or from a
     video file by launching the pipeline automatically.

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

  python test_with_coverage.py --stats --packets 120 --timeout 8
      Collect stats from a running pipeline.

  python test_with_coverage.py --video "/path/to/video.mp4" --stats
      Launch the pipeline on the given video and print stats after it finishes.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time
from types import SimpleNamespace
from typing import Any

import cv2
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

    return pkt


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
        last_progress_time = 0.0

        while valid_count < packets_needed:
            pkt = _recv_json_packet(sock, timeout_s=timeout_s)
            _validate_packet(pkt)
            valid_count += 1

            now = time.time()
            if now - last_progress_time >= 0.05 or valid_count == packets_needed:
                _print_progress(
                    current=valid_count,
                    total=packets_needed,
                    prefix="Live packets",
                )
                last_progress_time = now

        _finish_progress()
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


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _safe_mean(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    return float(math.sqrt(var))


def _class_name_from_id(cls_id: Any) -> str:
    try:
        cls_i = int(cls_id)
    except Exception:
        return ""

    for name, mapped_id in getattr(S, "UNITY_CLASS_ID", {}).items():
        try:
            if int(mapped_id) == cls_i:
                return str(name).lower()
        except Exception:
            continue
    return ""


def _person_detections(pkt: dict[str, Any]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    detections = pkt.get("detections", [])
    if not isinstance(detections, list):
        return out

    for det in detections:
        if not isinstance(det, dict):
            continue
        if _class_name_from_id(det.get("cls")) != "person":
            continue
        try:
            out.append(
                {
                    "id": float(det["id"]),
                    "cx": float(det["cx"]),
                    "cy": float(det["cy"]),
                }
            )
        except Exception:
            continue

    return out


def _estimate_id_switches(
    prev_people: list[dict[str, float]],
    cur_people: list[dict[str, float]],
    max_dist: float = 0.08,
) -> int:
    if not prev_people or not cur_people:
        return 0

    used_cur: set[int] = set()
    switches = 0

    for prev in prev_people:
        best_j = -1
        best_dist = float("inf")
        px = float(prev["cx"])
        py = float(prev["cy"])

        for j, cur in enumerate(cur_people):
            if j in used_cur:
                continue
            dx = float(cur["cx"]) - px
            dy = float(cur["cy"]) - py
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j >= 0 and best_dist <= max_dist:
            used_cur.add(best_j)
            if int(cur_people[best_j]["id"]) != int(prev["id"]):
                switches += 1

    return switches


def _estimate_video_total_packets(video_path: str) -> int | None:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            return frame_count
        return None
    finally:
        try:
            cap.release()
        except Exception:
            pass


def _print_progress(
    current: int,
    total: int | None = None,
    *,
    prefix: str = "Progress",
    width: int = 32,
) -> None:
    if total is not None and total > 0:
        current = max(0, min(current, total))
        ratio = float(current) / float(total)
        filled = int(round(ratio * width))
        bar = "#" * filled + "-" * (width - filled)
        pct = ratio * 100.0
        print(f"\r{prefix}: [{bar}] {current}/{total} ({pct:5.1f}%)", end="", flush=True)
    else:
        spinner = "|/-\\"
        ch = spinner[current % len(spinner)]
        print(f"\r{prefix}: {ch} packets={current}", end="", flush=True)


def _finish_progress() -> None:
    print()


def _collect_udp_stats(
    host: str,
    port: int,
    packets_needed: int,
    timeout_s: float,
    stop_event: threading.Event | None = None,
    progress_total: int | None = None,
    progress_prefix: str = "Stats",
) -> dict[str, Any]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, int(port)))
        sock.settimeout(float(timeout_s))

        valid_packets = 0
        invalid_packets = 0
        packet_sizes: list[int] = []
        source_dt_values: list[float] = []
        arrival_dt_values: list[float] = []

        frame_gaps = 0
        duplicate_frames = 0
        out_of_order_frames = 0
        person_id_switches_est = 0

        prev_frame_id: int | None = None
        prev_source_ts: float | None = None
        prev_arrival_ts: float | None = None
        prev_people: list[dict[str, float]] = []

        last_progress_time = 0.0

        while True:
            if packets_needed > 0 and valid_packets >= packets_needed:
                break
            if stop_event is not None and stop_event.is_set():
                break

            try:
                pkt, raw_bytes, arrival_ts = _recv_json_packet_with_meta(sock, timeout_s)
                _validate_packet(pkt)
            except TimeoutError:
                if stop_event is not None and stop_event.is_set():
                    break
                continue
            except Exception:
                invalid_packets += 1
                continue

            valid_packets += 1
            packet_sizes.append(len(raw_bytes))

            frame_id = int(pkt["frame_id"])
            source_ts = float(pkt["timestamp"])

            if prev_frame_id is not None:
                frame_delta = frame_id - prev_frame_id
                if frame_delta > 1:
                    frame_gaps += frame_delta - 1
                elif frame_delta == 0:
                    duplicate_frames += 1
                else:
                    out_of_order_frames += 1

            if prev_source_ts is not None:
                dt = source_ts - prev_source_ts
                if dt > 0:
                    source_dt_values.append(dt)

            if prev_arrival_ts is not None:
                dt = arrival_ts - prev_arrival_ts
                if dt > 0:
                    arrival_dt_values.append(dt)

            cur_people = _person_detections(pkt)
            person_id_switches_est += _estimate_id_switches(prev_people, cur_people)
            prev_people = cur_people

            prev_frame_id = frame_id
            prev_source_ts = source_ts
            prev_arrival_ts = arrival_ts

            now = time.time()
            if now - last_progress_time >= 0.05:
                _print_progress(
                    current=valid_packets,
                    total=progress_total,
                    prefix=progress_prefix,
                )
                last_progress_time = now

        _print_progress(
            current=valid_packets,
            total=progress_total,
            prefix=progress_prefix,
        )
        _finish_progress()

        source_fps_est = 0.0
        arrival_fps_est = 0.0

        if source_dt_values:
            source_fps_est = 1.0 / _safe_mean(source_dt_values)
        if arrival_dt_values:
            arrival_fps_est = 1.0 / _safe_mean(arrival_dt_values)

        return {
            "valid_packets": valid_packets,
            "invalid_packets": invalid_packets,
            "avg_packet_bytes": _safe_mean([float(v) for v in packet_sizes]),
            "min_packet_bytes": min(packet_sizes) if packet_sizes else 0,
            "max_packet_bytes": max(packet_sizes) if packet_sizes else 0,
            "source_fps_est": source_fps_est,
            "arrival_fps_est": arrival_fps_est,
            "source_dt_jitter_s": _safe_std(source_dt_values),
            "arrival_dt_jitter_s": _safe_std(arrival_dt_values),
            "frame_gaps": frame_gaps,
            "duplicate_frames": duplicate_frames,
            "out_of_order_frames": out_of_order_frames,
            "person_id_switches_est": person_id_switches_est,
        }
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _print_stats(stats: dict[str, Any]) -> None:
    print(
        "STATS: "
        f"valid_packets={stats['valid_packets']}, "
        f"invalid_packets={stats['invalid_packets']}, "
        f"avg_packet_bytes={stats['avg_packet_bytes']:.1f}, "
        f"min_packet_bytes={stats['min_packet_bytes']}, "
        f"max_packet_bytes={stats['max_packet_bytes']}, "
        f"source_fps_est={stats['source_fps_est']:.2f}, "
        f"arrival_fps_est={stats['arrival_fps_est']:.2f}, "
        f"source_dt_jitter_s={stats['source_dt_jitter_s']:.6f}, "
        f"arrival_dt_jitter_s={stats['arrival_dt_jitter_s']:.6f}, "
        f"frame_gaps={stats['frame_gaps']}, "
        f"duplicate_frames={stats['duplicate_frames']}, "
        f"out_of_order_frames={stats['out_of_order_frames']}, "
        f"person_id_switches_est={stats['person_id_switches_est']}"
    )


def _run_pipeline_on_video(video_path: str) -> int:
    import main as pipeline_main

    old_input_mode = S.INPUT_MODE
    old_video_path = S.VIDEO_PATH
    old_enable_udp = S.ENABLE_UDP

    try:
        S.INPUT_MODE = "file"
        S.VIDEO_PATH = video_path
        S.ENABLE_UDP = True
        return int(pipeline_main.run_live(SimpleNamespace(no_gui=True)))
    finally:
        S.INPUT_MODE = old_input_mode
        S.VIDEO_PATH = old_video_path
        S.ENABLE_UDP = old_enable_udp


def _collect_video_stats(
    video_path: str,
    host: str,
    port: int,
    timeout_s: float,
) -> dict[str, Any]:
    stop_event = threading.Event()
    pipeline_result: dict[str, Any] = {"code": None, "error": None}

    def _pipeline_target() -> None:
        try:
            pipeline_result["code"] = _run_pipeline_on_video(video_path)
        except Exception as e:
            pipeline_result["error"] = e
        finally:
            stop_event.set()

    expected_packets = _estimate_video_total_packets(video_path)

    thread = threading.Thread(target=_pipeline_target, daemon=True)
    thread.start()

    time.sleep(0.25)

    stats = _collect_udp_stats(
        host=host,
        port=port,
        packets_needed=0,
        timeout_s=timeout_s,
        stop_event=stop_event,
        progress_total=expected_packets,
        progress_prefix="Video stats",
    )

    thread.join()

    if pipeline_result["error"] is not None:
        raise RuntimeError(f"video pipeline failed: {pipeline_result['error']}")

    if int(stats["valid_packets"]) <= 0:
        raise AssertionError("no valid UDP packets were received while running the video")

    return stats


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
        "--stats",
        action="store_true",
        help="Collect UDP stream stats.",
    )
    parser.add_argument(
        "--video",
        default="",
        help="Run the pipeline on this video path and print stats after it finishes.",
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
        help="Number of valid packets required for --live or --stats. Ignored for --video.",
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

        if args.video:
            if not args.stats:
                raise AssertionError("--video requires --stats")
            stats = _collect_video_stats(
                video_path=args.video,
                host=args.host,
                port=args.port,
                timeout_s=args.timeout,
            )
            print(
                "PASSED: UDP formatter structure, UDP send/receive, "
                "and video stats collection are valid"
            )
            _print_stats(stats)

        elif args.stats:
            stats = _collect_udp_stats(
                host=args.host,
                port=args.port,
                packets_needed=args.packets,
                timeout_s=args.timeout,
                progress_total=args.packets,
                progress_prefix="Stats",
            )
            print(
                "PASSED: UDP formatter structure, UDP send/receive, and stats collection are valid"
            )
            _print_stats(stats)

        elif args.live:
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
        if args.video:
            print(f"FAILED: video stats collection or packet structure is invalid: {e}")
        elif args.stats:
            print(f"FAILED: UDP stats collection or packet structure is invalid: {e}")
        elif args.live:
            print(f"FAILED: live UDP transport or README packet structure is invalid: {e}")
        else:
            print(f"FAILED: UDP formatter structure or UDP send/receive is invalid: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
