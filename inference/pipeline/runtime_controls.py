"""
runtime_controls.py

Runtime state and keyboard-control helpers for the live pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import settings as S
from frame_io import open_capture
from motion_smoothing import PoseMotionSmoother, WorldTrackSmoother


@dataclass
class LiveRuntimeState:
    people_on: bool
    fire_on: bool
    recording_enabled: bool
    draw_detections: bool
    tracking_enabled: bool
    draw_track_ids: bool
    pose_mode_overlay_on: bool
    dji_overlay_on: bool
    active_camera_source: str
    motion_smoothing_value: float


def format_adaptive_metrics(metrics) -> str:
    return (
        "pose_valid={:.0%}, markers={:.2f}, pos_jitter={:.3f}m, rot_jitter={:.2f}deg, "
        "coast={:.0%}, id_switch={:.0%}, fps={:.1f}, drops={:.2f}"
    ).format(
        float(getattr(metrics, "pose_valid_ratio", 0.0)),
        float(getattr(metrics, "avg_markers_used", 0.0)),
        float(getattr(metrics, "pose_position_jitter_m", 0.0)),
        float(getattr(metrics, "pose_rotation_jitter_deg", 0.0)),
        float(getattr(metrics, "coast_ratio", 0.0)),
        float(getattr(metrics, "id_switch_rate", 0.0)),
        float(getattr(metrics, "avg_fps", 0.0)),
        float(getattr(metrics, "avg_drop_frames", 0.0)),
    )


def update_motion_smoothing_value(
    pose_smoother: PoseMotionSmoother | None,
    world_smoother: WorldTrackSmoother | None,
    value: float,
) -> float:
    value = max(0.0, min(1.0, float(value)))
    if pose_smoother is not None:
        pose_smoother.set_smoothness(value)
    if world_smoother is not None:
        world_smoother.set_smoothness(value)
    return value


def _reset_runtime_filters(pose_smoother, world_smoother, id_flicker_mitigator, adaptive_tuner):
    if pose_smoother is not None:
        pose_smoother.reset()
    if world_smoother is not None:
        world_smoother.reset()
    if id_flicker_mitigator is not None:
        id_flicker_mitigator.reset()
    if adaptive_tuner is not None:
        adaptive_tuner.reset()


def handle_runtime_key(
    *,
    key: int,
    state: LiveRuntimeState,
    pose_smoother,
    world_smoother,
    id_flicker_mitigator,
    adaptive_tuner,
    cap,
    is_file_source: bool,
    target_fps: float,
    video_start_wall: float,
    fps_hist,
    drop_hist,
    t_prev: float,
    window_frames: int,
    window_start: float,
):
    if key == S.KEY_ESC:
        return {
            "should_exit": True,
            "cap": cap,
            "is_file_source": is_file_source,
            "target_fps": target_fps,
            "video_start_wall": video_start_wall,
            "t_prev": t_prev,
            "window_frames": window_frames,
            "window_start": window_start,
        }

    if key in S.KEY_TOGGLE_RECORDING:
        if not state.recording_enabled:
            print("Recording ENABLED at", time.strftime("%Y-%m-%d %H:%M:%S"))
            state.recording_enabled = True
        else:
            print("Recording DISABLED at", time.strftime("%Y-%m-%d %H:%M:%S"))
            state.recording_enabled = False

    elif key in S.KEY_TOGGLE_PEOPLE:
        state.people_on = not state.people_on
        id_flicker_mitigator.reset()
        adaptive_tuner.reset()

    elif key in S.KEY_TOGGLE_FIRE:
        state.fire_on = not state.fire_on
        id_flicker_mitigator.reset()
        adaptive_tuner.reset()

    elif key in S.KEY_TOGGLE_DRAW:
        state.draw_detections = not state.draw_detections

    elif key in S.KEY_TOGGLE_DJI_OVERLAY:
        state.dji_overlay_on = not state.dji_overlay_on

    elif key in getattr(S, "KEY_TOGGLE_TRACKING", (ord("t"), ord("T"))):
        state.tracking_enabled = not state.tracking_enabled
        id_flicker_mitigator.reset()
        adaptive_tuner.reset()

    elif key in getattr(S, "KEY_TOGGLE_POSE_MODE_OVERLAY", (ord("m"), ord("M"))):
        state.pose_mode_overlay_on = not state.pose_mode_overlay_on

    elif key in S.KEY_TOGGLE_INPUT and S.INPUT_MODE.lower() == "camera":
        prev_source = state.active_camera_source
        state.active_camera_source = "capture_card" if prev_source == "webcam" else "webcam"

        try:
            cap.release()
        except Exception:
            pass

        try:
            cap, is_file_source, target_fps, video_start_wall, _input_desc = open_capture(
                "camera", state.active_camera_source
            )

            fps_hist.clear()
            drop_hist.clear()
            t_prev = time.time()
            window_frames = 0
            window_start = time.time()
            _reset_runtime_filters(
                pose_smoother,
                world_smoother,
                id_flicker_mitigator,
                adaptive_tuner,
            )

        except Exception as exc:
            print("Toggle input failed:", exc)
            state.active_camera_source = prev_source
            cap, is_file_source, target_fps, video_start_wall, _input_desc = open_capture(
                "camera", state.active_camera_source
            )
            _reset_runtime_filters(
                pose_smoother,
                world_smoother,
                id_flicker_mitigator,
                adaptive_tuner,
            )

    elif key in getattr(S, "KEY_TOGGLE_MOTION_SMOOTHING", (ord("g"), ord("G"))):
        if pose_smoother is None and world_smoother is None:
            print("Motion smoothing unavailable in the current runtime mode")
        else:
            current_enabled = True
            if pose_smoother is not None:
                current_enabled = bool(pose_smoother.enabled)
            elif world_smoother is not None:
                current_enabled = bool(world_smoother.enabled)
            new_enabled = not current_enabled
            if pose_smoother is not None:
                pose_smoother.set_enabled(new_enabled)
            if world_smoother is not None:
                world_smoother.set_enabled(new_enabled)
            status = "ENABLED" if new_enabled else "DISABLED"
            print(f"Motion smoothing {status} | value={state.motion_smoothing_value:.2f}")

    elif key in getattr(S, "KEY_DECREASE_MOTION_SMOOTHING", (ord("["), ord("{"))):
        state.motion_smoothing_value = update_motion_smoothing_value(
            pose_smoother,
            world_smoother,
            state.motion_smoothing_value - float(getattr(S, "MOTION_SMOOTHING_STEP", 0.05)),
        )
        print(f"Motion smoothing: {state.motion_smoothing_value:.2f}")

    elif key in getattr(S, "KEY_INCREASE_MOTION_SMOOTHING", (ord("]"), ord("}"))):
        state.motion_smoothing_value = update_motion_smoothing_value(
            pose_smoother,
            world_smoother,
            state.motion_smoothing_value + float(getattr(S, "MOTION_SMOOTHING_STEP", 0.05)),
        )
        print(f"Motion smoothing: {state.motion_smoothing_value:.2f}")

    return {
        "should_exit": False,
        "cap": cap,
        "is_file_source": is_file_source,
        "target_fps": target_fps,
        "video_start_wall": video_start_wall,
        "t_prev": t_prev,
        "window_frames": window_frames,
        "window_start": window_start,
    }
