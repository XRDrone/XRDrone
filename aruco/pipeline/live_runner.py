"""
live_runner.py

Pure ArUco prerecorded/live-video runtime for XRDrone.
This runner uses only ArUco marker pose estimation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import settings as S
from aruco_pose import ArucoPoseEstimator
from frame_io import format_frame
from runtime_logger import RuntimeLogger
from streaming import UDPPublisher


class OptionalYoloDetector:
    """Small optional Ultralytics wrapper. If the model is missing, detections are empty."""

    def __init__(self, model_path: str, *, enabled: bool = True) -> None:
        self.model_path = str(model_path)
        self.enabled = bool(enabled)
        self.model = None
        self.names: dict[int, str] = {}
        self.error: str | None = None

        if not self.enabled:
            self.error = "disabled"
            return
        if not self.model_path or not Path(self.model_path).exists():
            self.enabled = False
            self.error = f"model not found: {self.model_path}"
            return
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            raw_names = getattr(self.model, "names", {}) or {}
            if isinstance(raw_names, dict):
                self.names = {int(k): str(v) for k, v in raw_names.items()}
            elif isinstance(raw_names, (list, tuple)):
                self.names = {i: str(v) for i, v in enumerate(raw_names)}
            else:
                self.names = {}
        except Exception as exc:
            self.enabled = False
            self.error = repr(exc)

    def _class_ids(self, wanted_names: tuple[str, ...]) -> list[int] | None:
        if not self.names:
            return None
        wanted = {str(name).lower() for name in wanted_names}
        ids = [idx for idx, name in self.names.items() if str(name).lower() in wanted]
        return ids or None

    def detect(self, frame_bgr, *, frame_id: int) -> tuple[list[dict[str, Any]], Any]:
        if not self.enabled or self.model is None:
            return [], None

        pred_kw = dict(
            conf=float(getattr(S, "PEOPLE_CONF", 0.40)),
            imgsz=int(getattr(S, "IMGSZ", 960)),
            verbose=False,
        )
        device = getattr(S, "DEVICE", None)
        if device is not None:
            pred_kw["device"] = device
        class_ids = self._class_ids(tuple(getattr(S, "DETECT_CLASSES", ("person",))))
        if class_ids is not None:
            pred_kw["classes"] = class_ids

        results = None
        try:
            tracker_yaml = str(getattr(S, "ULTRALYTICS_TRACKER_YAML", "botsort_drone.yaml"))
            if bool(getattr(S, "TRACKING_ENABLED", True)) and Path(tracker_yaml).exists():
                results = self.model.track(
                    frame_bgr,
                    persist=True,
                    tracker=tracker_yaml,
                    **pred_kw,
                )
            else:
                results = self.model.predict(frame_bgr, **pred_kw)
        except Exception:
            results = self.model.predict(frame_bgr, **pred_kw)

        detections: list[dict[str, Any]] = []
        height, width = frame_bgr.shape[:2]
        width_f = float(max(1, width))
        height_f = float(max(1, height))

        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                try:
                    cls_id = int(box.cls[0]) if box.cls is not None else -1
                    label = str(self.names.get(cls_id, cls_id)).lower()
                    if label == "item":
                        label = "person"
                    if label not in set(getattr(S, "UDP_SEND_CLASSES", ("person",))):
                        continue
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    track_id = None
                    if getattr(box, "id", None) is not None:
                        try:
                            track_id = int(box.id[0])
                        except Exception:
                            track_id = None
                    detections.append(
                        {
                            "frame_id": int(frame_id),
                            "class": label,
                            "confidence": conf,
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                            "track_id": track_id,
                            "foot_x": max(0.0, min(1.0, ((x1 + x2) * 0.5) / width_f)),
                            "foot_y": max(0.0, min(1.0, y2 / height_f)),
                        }
                    )
                except Exception:
                    continue
        return detections, results


def _build_packet(
    *,
    frame_id: int,
    timestamp: float,
    timestamp_video_s: float | None,
    width: int,
    height: int,
    pose: dict[str, Any],
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "frame_id": int(frame_id),
        "timestamp": float(timestamp),
        "timestamp_video_s": None if timestamp_video_s is None else float(timestamp_video_s),
        "width": int(width),
        "height": int(height),
        "pose": pose,
        "detections": detections,
        "counts": {
            "detections": int(len(detections)),
            "markers_detected": int(pose.get("markers_detected", 0) or 0),
            "markers_used": int(pose.get("markers_used", 0) or 0),
        },
    }


def _draw_detections(frame, detections: list[dict[str, Any]]) -> None:
    for det in detections:
        try:
            x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
            label = str(det.get("class", "obj"))
            conf = float(det.get("confidence", 0.0))
            tid = det.get("track_id")
            text = f"{label} {conf:.2f}" if tid is None else f"{label}#{tid} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                frame,
                text,
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        except Exception:
            continue


def _draw_pose_status(frame, pose: dict[str, Any], log_dir: Path | None = None) -> None:
    status = "POSE OK" if pose.get("pose_valid") else "POSE MISSING"
    text = (
        f"{status} | markers {pose.get('markers_used', 0)}/"
        f"{pose.get('markers_detected', 0)} | frame {pose.get('frame_id')}"
    )
    cv2.putText(
        frame,
        text,
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if log_dir is not None:
        cv2.putText(
            frame,
            f"logs: {log_dir}",
            (20, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _open_capture(video_path: str | None):
    if video_path:
        cap = cv2.VideoCapture(str(video_path))
        desc = f"file: {video_path}"
    elif str(getattr(S, "INPUT_MODE", "file")).lower() == "file":
        cap = cv2.VideoCapture(str(S.VIDEO_PATH))
        desc = f"file: {S.VIDEO_PATH}"
    else:
        cap = cv2.VideoCapture(int(getattr(S, "VIDEO_SOURCE", 0)))
        desc = f"camera: {S.VIDEO_SOURCE}"
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input {desc}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1.0:
        fps = float(getattr(S, "DEFAULT_FPS", 30.0))
    return cap, desc, fps


def run_live(args) -> int:
    video_path = getattr(args, "video", None)
    cap, input_desc, fps = _open_capture(video_path)
    logs_enabled = bool(getattr(args, "logs", False))
    logger = (
        RuntimeLogger(getattr(args, "log_root", None) or getattr(S, "LOG_ROOT", "logs"))
        if logs_enabled
        else None
    )

    pose_estimator = ArucoPoseEstimator(
        marker_world_positions=S.POSE_MARKER_WORLD_POSITIONS,
        marker_size_m=float(getattr(S, "POSE_MARKER_SIZE_M", 0.1645)),
        aruco_dict_name=str(getattr(S, "POSE_ARUCO_DICT", "DICT_4X4_50")),
        hfov_deg=float(getattr(S, "POSE_HFOV_DEG", 84.0)),
        corner_refinement=str(getattr(S, "POSE_CORNER_REFINEMENT", "subpix")),
    )
    detector = OptionalYoloDetector(
        getattr(args, "model", None) or getattr(S, "PEOPLE_MODEL_PATH", ""),
        enabled=not bool(getattr(args, "no_detect", False)),
    )
    udp = UDPPublisher(S.UDP_IP, int(S.UDP_PORT)) if bool(getattr(S, "ENABLE_UDP", True)) else None

    metadata = {
        "input": input_desc,
        "fps": fps,
        "log_dir": str(logger.run_dir) if logger is not None else None,
        "logs_enabled": bool(logs_enabled),
        "pure_aruco": True,
        "marker_world_positions": {
            str(k): list(v) for k, v in S.POSE_MARKER_WORLD_POSITIONS.items()
        },
        "marker_size_m": float(getattr(S, "POSE_MARKER_SIZE_M", 0.1645)),
        "aruco_dict": str(getattr(S, "POSE_ARUCO_DICT", "DICT_4X4_50")),
        "hfov_deg": float(getattr(S, "POSE_HFOV_DEG", 84.0)),
        "detection_model_path": detector.model_path,
        "detection_enabled": bool(detector.enabled),
        "detection_status": detector.error or "loaded",
    }
    if logger is not None:
        logger.write_metadata(metadata)

    print("Input:", input_desc)
    if logger is not None:
        print("Logs:", logger.run_dir)
    else:
        print("Logs: disabled (use --logs to create a per-run log folder)")
    if detector.error:
        print("Detection model:", detector.error)
    else:
        print("Detection model loaded:", detector.model_path)

    frame_id = 0
    pose_valid_count = 0
    total_detections = 0
    started = time.time()
    out = None
    writer_fps = fps
    max_frames = getattr(args, "max_frames", None)
    save_output = bool(getattr(args, "save_output", False) or getattr(S, "SAVE_OUTPUT", False))

    try:
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                break
            frame_start = time.time()
            next_frame_id = frame_id + 1
            if max_frames is not None and int(max_frames) > 0 and next_frame_id > int(max_frames):
                break
            frame_id = next_frame_id

            timestamp_video_s = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            timestamp = time.time()
            frame = format_frame(raw_frame)
            height, width = frame.shape[:2]

            pose_result = pose_estimator.estimate(
                frame,
                frame_id=frame_id,
                timestamp=timestamp,
                timestamp_video_s=timestamp_video_s,
                draw=bool(getattr(S, "POSE_DRAW_ARUCO", True)),
            )
            pose = pose_result.pose_packet
            if pose.get("pose_valid"):
                pose_valid_count += 1

            detections, _raw_results = detector.detect(frame, frame_id=frame_id)
            total_detections += len(detections)

            detection_record = {
                "frame_id": int(frame_id),
                "timestamp": float(timestamp),
                "timestamp_video_s": float(timestamp_video_s),
                "width": int(width),
                "height": int(height),
                "detections": detections,
                "detection_count": int(len(detections)),
            }
            packet = _build_packet(
                frame_id=frame_id,
                timestamp=timestamp,
                timestamp_video_s=timestamp_video_s,
                width=width,
                height=height,
                pose=pose,
                detections=detections,
            )

            if logger is not None:
                logger.log_pose(pose)
                logger.log_markers(pose_result.marker_packet)
                logger.log_detections(detection_record)
                logger.log_packet(packet)
                logger.log_frame_row(
                    {
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "timestamp_video_s": timestamp_video_s,
                        "width": width,
                        "height": height,
                        "pose_valid": bool(pose.get("pose_valid", False)),
                        "markers_detected": int(pose.get("markers_detected", 0) or 0),
                        "markers_used": int(pose.get("markers_used", 0) or 0),
                        "detection_count": len(detections),
                        "processing_ms": round((time.time() - frame_start) * 1000.0, 3),
                    }
                )

            if udp is not None:
                try:
                    udp.send_json(packet)
                except Exception as exc:
                    if logger is not None:
                        logger.log_error(
                            {
                                "frame_id": int(frame_id),
                                "timestamp": float(timestamp),
                                "stage": "udp_send",
                                "error": repr(exc),
                            }
                        )
                    else:
                        print(f"UDP send error on frame {frame_id}: {exc!r}")

            if bool(getattr(S, "DRAW_DETECTIONS_DEFAULT", True)):
                _draw_detections(frame, detections)
            _draw_pose_status(frame, pose, logger.run_dir if logger is not None else None)

            if save_output:
                if out is None:
                    output_path = Path(
                        getattr(args, "output", None)
                        or getattr(S, "OUTPUT_VIDEO", "aruco_output.mp4")
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*str(getattr(S, "OUTPUT_CODEC", "mp4v")))
                    out = cv2.VideoWriter(str(output_path), fourcc, writer_fps, (width, height))
                out.write(frame)

            if not bool(getattr(args, "no_gui", False)):
                cv2.imshow(str(getattr(S, "WINDOW_NAME", "XRDrone ArUco")), frame)
                key = cv2.waitKey(1) & 0xFF
                if key == int(getattr(S, "KEY_ESC", 27)):
                    break

            if frame_id % int(getattr(S, "PRINT_EVERY_N_FRAMES", 30)) == 0:
                print(
                    f"frame={frame_id} pose_valid={pose.get('pose_valid')} "
                    f"markers={pose.get('markers_used')}/{pose.get('markers_detected')} "
                    f"detections={len(detections)}"
                )
    except Exception as exc:
        if logger is not None:
            logger.log_error({"timestamp": time.time(), "stage": "run_live", "error": repr(exc)})
        raise
    finally:
        elapsed = max(0.001, time.time() - started)
        summary = {
            "input": input_desc,
            "log_dir": str(logger.run_dir) if logger is not None else None,
            "logs_enabled": bool(logs_enabled),
            "frames_processed": int(frame_id),
            "elapsed_s": elapsed,
            "avg_runtime_fps": float(frame_id) / elapsed,
            "pose_valid_frames": int(pose_valid_count),
            "pose_valid_ratio": float(pose_valid_count / frame_id) if frame_id else 0.0,
            "total_detections_logged": int(total_detections),
        }
        if logger is not None:
            logger.write_summary(summary)
            logger.close()
        try:
            cap.release()
        except Exception:
            pass
        if out is not None:
            try:
                out.release()
            except Exception:
                pass
        if udp is not None:
            udp.close()
        if not bool(getattr(args, "no_gui", False)):
            cv2.destroyAllWindows()

    print("Done.")
    print(json.dumps(summary, indent=2))
    return 0
