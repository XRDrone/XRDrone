# Runtime UI and Terminal Reference

This document explains the text you may see **on the video/output window** and the text that may be **printed to the terminal** during normal XRDrone pipeline use.

**Scope:** This document covers runtime behavior from the main pipeline and related modules. It intentionally **does not** cover `test_with_coverage.py`.

---

## 1. On-screen text in the video/output window

## ArUco pose-mode overlay

When the pose-mode overlay is enabled, the pipeline draws a small status label near the top-left of the frame. Its position and styling are controlled by:

- `POSE_MODE_OVERLAY_ENABLED_DEFAULT`
- `POSE_MODE_OVERLAY_ORIGIN`
- `POSE_MODE_OVERLAY_TEXT_SCALE`
- `POSE_MODE_OVERLAY_TEXT_THICKNESS`

Possible text values include:

### `ArUco: no known markers`
No configured ArUco marker from `POSE_MARKER_WORLD_POSITIONS` is currently visible.

### `ArUco: single marker | mode: single`
Exactly one known marker is visible and the single-marker pose path is being used.

### `ArUco: multiple markers (N) | mode: multi`
Two or more known markers are visible and the multi-marker board path is being used. `N` is the visible known-marker count.

### `ArUco: multiple markers (N) | mode: single`
Two or more known markers are visible, but the pipeline ended up using the single-marker path instead of the multi-marker solve. This can happen when the auto policy falls back to a single-marker solve.

### `ArUco: pose lost | holding`
The marker-based pose was lost, but the short pose-loss hold window is still active. During this period, the system preserves the last valid numeric pose values internally for a short time, while still marking pose as invalid.

### `ArUco: pose lost`
The marker-based pose is currently unavailable and the hold window is not active anymore.

### `... | pose unavailable`
This suffix can be appended when markers were seen but a valid pose could not be produced. For example:

- `ArUco: single marker | mode: single | pose unavailable`
- `ArUco: multiple markers (2) | mode: multi | pose unavailable`

This means the system detected known marker visibility status, but pose solving did not succeed for that frame.

---

## Detection and tracking labels

The pipeline can draw object labels directly on the frame.

### When tracked-box display is **not** active
The label format is:

```text
<class> <confidence>%
```

Examples:

```text
person 91.3%
fire 84.7%
smoke 76.1%
```

This is the standard per-detection class/confidence label.

### When tracked-box display **is** active
The label format becomes:

```text
<class> #<track_id> <confidence>%
```

Example:

```text
person #12 88.4%
```

This means the detection is currently associated with a persistent track ID.

### `[hold]` suffix
If the sender-side continuity layer is currently **coasting** a tracked object forward, the label gets this suffix:

```text
[hold]
```

Example:

```text
person #12 83.1% [hold]
```

This does **not** mean a fresh detection was seen in that frame. It means the continuity layer is temporarily forwarding the last stable tracked object through a short hold/coast window to reduce flicker.

---

## ArUco marker drawing

If `POSE_DRAW_ARUCO` is enabled, detected ArUco markers can be drawn on the frame. This is a visual drawing of marker outlines/corners, not a text label by itself.

---

## DJI overlay

If `DJI_MENU_OVERLAY_ENABLED_DEFAULT` is on, the pipeline can draw a pre-made RGBA overlay image across the frame. This is not generated from code text in `main.py`; it is an image asset loaded from `DJI_MENU_OVERLAY_PATH`.

---

## Window title

The OpenCV window title is:

```text
Live Pipeline
```

That is the window name, not an in-frame overlay.

---

## 2. Terminal output during normal live runs

## Startup prints

When `run_live()` starts, the pipeline prints basic Torch/CUDA information.

### `torch: ...`
Example:

```text
torch: 2.10.0
```

This shows the installed PyTorch version.

### `cuda available: ...`
Example:

```text
cuda available: True
```

This tells you whether PyTorch currently sees a usable CUDA GPU.

### `gpu: ...`
Example:

```text
gpu: NVIDIA GeForce RTX 4090
```

This only prints when CUDA is available.

---

## Adaptive tuning log line

If adaptive runtime tuning is enabled and logging is on, the pipeline can print lines such as:

```text
Adaptive tuning [low_trust] -> smooth=0.55, tau_on=0.82, tau_off=0.57, coast=5 | pose_valid=0%, markers=0.00, pos_jitter=0.000m, rot_jitter=0.00deg, coast=0%, id_switch=0%, fps=3.1, drops=14.00
```

This line means the adaptive controller changed runtime stabilization settings.

### Mode names

The mode inside brackets can be:

- `stable`
- `jittery_visible`
- `low_trust`
- `recovering`

### Left side of the line

- `smooth` = current motion-smoothing value being applied
- `tau_on` = confidence threshold required for a new tracked object to begin being emitted
- `tau_off` = lower threshold an already-emitted tracked object can fall to before it drops out
- `coast` = how many frames the continuity layer is allowed to hold/coast a tracked object

### Right side of the line

- `pose_valid` = fraction of recent frames with a valid pose
- `markers` = average number of markers used recently
- `pos_jitter` = average recent position jitter in meters
- `rot_jitter` = average recent rotation jitter in degrees
- `coast` = fraction of recent target detections that were being coasted rather than freshly observed
- `id_switch` = estimated recent track-ID switch rate
- `fps` = average recent runtime FPS
- `drops` = average recent dropped-frame estimate

### What the modes generally mean

- `stable`: pose and tracked output look stable; tuner is easing back toward the base settings.
- `jittery_visible`: pose is still visible, but motion/IDs look noisy; tuner is increasing stabilization.
- `low_trust`: the current runtime state is unreliable, usually because pose validity is low, marker support is weak, or frame drops are high.
- `recovering`: the system is between bad and stable; tuner moves back toward the base settings gradually.

---

## Recording toggle prints

Pressing the recording toggle key prints one of these:

### `Recording ENABLED at ...`
Example:

```text
Recording ENABLED at 2026-03-16 09:15:42
```

The pipeline has started saving the output video.

### `Recording DISABLED at ...`
Example:

```text
Recording DISABLED at 2026-03-16 09:18:07
```

The pipeline has stopped saving the output video.

---

## Input-toggle failure print

If switching between webcam and capture card fails, the pipeline prints:

```text
Toggle input failed: <error>
```

Example:

```text
Toggle input failed: Could not open camera index=1 backend=auto
```

This means the requested alternate camera source could not be opened, and the pipeline falls back to the previous source.

---

## Motion-smoothing prints

### `Motion smoothing ENABLED | value=...`
Example:

```text
Motion smoothing ENABLED | value=0.50
```

Motion smoothing was turned on.

### `Motion smoothing DISABLED | value=...`
Example:

```text
Motion smoothing DISABLED | value=0.50
```

Motion smoothing was turned off.

### `Motion smoothing: ...`
Example:

```text
Motion smoothing: 0.45
```

The smoothing value was manually decreased or increased.

---

## 3. Terminal output in `--test` mode

If you run the pipeline in test mode, it prints the UDP packet in two forms.

### `[UDP] JSON payload (one-line):`
The next line is the full packet serialized on one line.

### `[UDP] JSON payload (pretty):`
The next block is the same packet formatted with indentation for easier reading.

This is useful for inspecting the exact packet structure being produced by the pipeline.

---

## 4. Important terms that may confuse readers

## `holding`
When you see `ArUco: pose lost | holding`, it means the pose system just lost marker-based visibility, but it is still inside the short configurable hold window. The pose remains invalid, but the last valid numeric pose values may be temporarily preserved internally for continuity.

## `[hold]`
When you see `[hold]` appended to a tracked object label, it means the ID-flicker continuity layer is **coasting** the last stable detection for a short number of frames instead of showing a brand-new observation.

## `coasted`
Internally, the continuity layer uses the term `coasted` for a temporarily held-forward tracked detection. The visible on-screen sign of this is the `[hold]` suffix.

## `observed`
Internally, a tracked detection marked as `observed` is a fresh accepted detection rather than a coasted one. This internal state is not printed directly on the video frame.

---

## 5. Things that change behavior but do not print their own terminal message

Some runtime toggles change behavior without printing a dedicated terminal line:

- toggling people detection
- toggling fire detection
- toggling drawing on/off
- toggling the DJI overlay
- toggling tracked-box display
- toggling the pose-mode overlay
- successfully switching input source

You will usually notice these changes from the video/output window rather than the terminal.

---

## 6. Quick summary

If you are scanning a live run, the most important runtime text usually means:

- `ArUco: ...` = current marker visibility / pose-mode status
- `#<id>` = tracked object ID
- `[hold]` = track is being coasted to reduce flicker
- `Adaptive tuning [...]` = runtime tuner changed smoothing / hysteresis / coast settings
- `Motion smoothing ...` = smoothing was toggled or adjusted
- `Recording ENABLED/DISABLED ...` = output recording changed state

