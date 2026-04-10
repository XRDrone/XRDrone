"""
runtime_controls.py

Runtime state and keyboard-control helpers for the live pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import settings as S
from frame_io import open_capture


@dataclass
class LiveRuntimeState:
    people_on: bool
    fire_on: bool
    recording_enabled: bool
    draw_detections: bool
    tracking_enabled: bool
    draw_track_ids: bool
    dji_overlay_on: bool
    active_camera_source: str


def _reset_runtime_filters(id_flicker_mitigator):
    id_flicker_mitigator.reset()


def handle_runtime_key(
    *,
    key: int,
    state: LiveRuntimeState,
    id_flicker_mitigator,
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

    elif key in S.KEY_TOGGLE_FIRE:
        state.fire_on = not state.fire_on
        id_flicker_mitigator.reset()

    elif key in S.KEY_TOGGLE_DRAW:
        state.draw_detections = not state.draw_detections

    elif key in S.KEY_TOGGLE_DJI_OVERLAY:
        state.dji_overlay_on = not state.dji_overlay_on

    elif key in getattr(S, "KEY_TOGGLE_TRACKING", (ord("t"), ord("T"))):
        state.tracking_enabled = not state.tracking_enabled
        id_flicker_mitigator.reset()

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
            _reset_runtime_filters(id_flicker_mitigator)

        except Exception as exc:
            print("Toggle input failed:", exc)
            state.active_camera_source = prev_source
            cap, is_file_source, target_fps, video_start_wall, _input_desc = open_capture(
                "camera", state.active_camera_source
            )
            _reset_runtime_filters(id_flicker_mitigator)

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
