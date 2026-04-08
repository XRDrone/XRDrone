"""
streaming.py

Provides network streaming utilities for the XRDrone pipeline.

Includes:
  - UDPPublisher:
      Sends JSON detection packets to external consumers (e.g., Unity)
      over UDP sockets.

  - RTSPStreamer:
      Pipes raw video frames into FFmpeg and publishes a live RTSP stream.

Handles:
  - socket lifecycle management
  - JSON serialization and transmission
  - FFmpeg process spawning and restart logic
  - resolution changes during runtime
"""

from __future__ import annotations

import json
import socket
import subprocess
from typing import Any

import numpy as np
import settings as S

try:
    import orjson
except Exception:  # pragma: no cover
    orjson = None  # type: ignore


def _encode_json_bytes(obj: dict[str, Any]) -> bytes:
    """
    Encode a packet to UTF-8 JSON bytes without changing the JSON structure.
    Uses orjson when enabled and available, otherwise falls back to compact stdlib JSON.
    """
    serializer = str(getattr(S, "JSON_SERIALIZER", "json")).strip().lower()
    ensure_ascii = bool(getattr(S, "JSON_ENSURE_ASCII", False))

    if serializer == "orjson" and orjson is not None:
        return orjson.dumps(obj)

    return json.dumps(
        obj,
        separators=(",", ":"),
        ensure_ascii=ensure_ascii,
    ).encode("utf-8")


class UDPPublisher:
    def __init__(self, ip: str, port: int) -> None:
        self.ip = ip
        self.port = int(port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_json(self, obj: dict[str, Any]) -> None:
        msg = _encode_json_bytes(obj)
        self.sock.sendto(msg, (self.ip, self.port))

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


class RTSPStreamer:
    """
    Spawns FFmpeg and pipes raw BGR frames to an RTSP URL.
    Mirrors the approach in test_udp.py: ffmpeg stdin + frame.tobytes().
    """

    def __init__(self, rtsp_url: str, fps: float) -> None:
        self.rtsp_url = rtsp_url
        self.fps = float(fps)
        self.proc: subprocess.Popen | None = None
        self.width: int | None = None
        self.height: int | None = None

    def _start(self, width: int, height: int) -> None:
        # ffmpeg reads raw bgr24 frames from stdin and publishes to RTSP.
        cmd = [
            "ffmpeg",
            "-re",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rtsp",
            self.rtsp_url,
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.width = int(width)
        self.height = int(height)

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None:
            return
        h, w = frame_bgr.shape[:2]

        if self.proc is None or self.proc.stdin is None:
            self._start(w, h)
        elif self.width != w or self.height != h:
            # Restart if resolution changes.
            self.close()
            self._start(w, h)

        try:
            assert self.proc is not None and self.proc.stdin is not None
            self.proc.stdin.write(frame_bgr.tobytes())
        except BrokenPipeError:
            # Downstream server closed or ffmpeg died; attempt restart next frame.
            self.close()
        except Exception:
            # Keep the main loop alive even if streaming fails.
            self.close()

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=1.0)
        except Exception:
            pass
        self.proc = None
        self.width = None
        self.height = None
