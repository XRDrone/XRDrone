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

These mode decisions come from the native `AdaptiveRuntimeTuner::choose_mode()` implementation in `src/adaptive_tuning.rs`.
The tuner checks modes in this priority order:

1. `low_trust`
2. `stable`
3. `jittery_visible`
4. `recovering`

That order matters. For example, if the system is noisy enough to qualify as `jittery_visible` but also has very low pose validity, it will be reported as `low_trust` because that check runs first.

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

### Exact conditions for switching modes

All thresholds below come from `src/adaptive_tuning.rs`.

#### `low_trust`
The tuner switches to `low_trust` if **any one** of these is true:

- `pose_valid_ratio < 0.35`
- `avg_markers_used < 0.75`
- `avg_drop_frames > 3.0`

This is the most conservative mode. It wins immediately if any of the above conditions are met.

#### `stable`
The tuner switches to `stable` only if it did **not** already enter `low_trust`, and **all** of these are true:

- `pose_valid_ratio >= 0.85`
- `pose_position_jitter_m <= 0.05`
- `pose_rotation_jitter_deg <= 2.5`
- `coast_ratio <= 0.12`
- `id_switch_rate <= 0.05`
- `avg_drop_frames <= 1.0`

This is the cleanest runtime state. The controller uses it to relax stabilization back toward the base settings.

#### `jittery_visible`
The tuner switches to `jittery_visible` only if it did **not** already enter `low_trust` or `stable`, and:

- `pose_valid_ratio >= 0.35`

and **at least one** of these is true:

- `pose_position_jitter_m >= 0.08`
- `pose_rotation_jitter_deg >= 4.0`
- `coast_ratio >= 0.18`
- `id_switch_rate >= 0.08`

This means pose is still present often enough to trust visibility, but the motion or track continuity is noisy enough that the tuner increases stabilization.

#### `recovering`
The tuner switches to `recovering` when the frame history is **not bad enough** for `low_trust`, **not clean enough** for `stable`, and **not noisy enough** for `jittery_visible`.

Operationally, this is the in-between state where the controller steps values back toward the configured base settings.

### What each mode changes

These target adjustments come from `AdaptiveRuntimeTuner::propose_adjustment()` in `src/adaptive_tuning.rs`.

#### `stable`
- `smooth` target: `max(motion_smoothing_min, base_motion_smoothing - 0.10)`
- `tau_on` target: `base_tau_on`
- `tau_off` target: `min(id_tau_off_max, base_tau_off + 0.02)`
- `coast` target: `max(id_coast_frames_min, base_coast_frames - 1)`

#### `jittery_visible`
- `smooth` target: `min(motion_smoothing_max, base_motion_smoothing + 0.15)`
- `tau_on` target: `base_tau_on`
- `tau_off` target: `max(id_tau_off_min, base_tau_off - 0.06)`
- `coast` target: `min(id_coast_frames_max, base_coast_frames + 2)`

#### `low_trust`
- `smooth` target: `min(motion_smoothing_max, base_motion_smoothing + 0.10)`
- `tau_on` target: `min(id_tau_on_max, base_tau_on + 0.04)`
- `tau_off` target: `min(id_tau_off_max, base_tau_off + 0.04)`
- `coast` target: `max(id_coast_frames_min, base_coast_frames - 1)`

#### `recovering`
- `smooth` target: `base_motion_smoothing`
- `tau_on` target: `base_tau_on`
- `tau_off` target: `base_tau_off`
- `coast` target: `base_coast_frames`

### How quickly a mode change can take effect

The mode logic is not evaluated continuously every frame without delay. The tuner has three built-in timing gates in `src/adaptive_tuning.rs`:

- `window_frames`: rolling-history length used for the recent metrics
- `update_interval_frames`: the controller only proposes a change when `frame_count % update_interval_frames == 0`
- `cooldown_frames`: after a change is applied, the tuner will not apply another one until this many frames pass

That means a mode switch has two kinds of delay:

1. **History delay**: the metrics need enough recent data to be meaningful.
2. **Application delay**: even after the metrics indicate a new mode, the tuner waits until the next update boundary and may also be blocked by cooldown.

So the controller reacts on a rolling, bounded cadence rather than instantaneously.

### Estimated motion-smoothing lag by mode

The tuner-side smoothing value still comes from `src/adaptive_tuning.rs`, but the actual smoothing implementation is now visible in `src/smoothing.rs` and the smoothness-to-filter-parameter mapping lives in `src/geometry.rs`.

The pose smoother uses One Euro filters, not a simple fixed-frame carryover. The relevant formulas are:

```text
position min_cutoff_hz = 3.5 - 3.1 * smoothness
position beta          = 0.03 + 0.32 * smoothness
rotation min_cutoff_hz = 4.5 - 3.8 * smoothness
rotation beta          = 0.04 + 0.42 * smoothness
alpha(cutoff, dt)      = 1 / (1 + tau / dt), where tau = 1 / (2π * cutoff)
```

That means higher `smooth` lowers the minimum cutoff and raises `beta`. In practice, that makes the filter more willing to damp slow motion and more willing to open up when motion speed increases. So the lag is **state dependent** and changes with runtime motion, not just with the configured mode.

Even with that limitation, the mode ordering is still clear from the code:

- `stable`: lowest expected lag because it targets the lowest smoothing value
- `recovering`: baseline lag because it returns to `base_motion_smoothing`
- `low_trust`: moderately higher lag because it raises smoothing above base
- `jittery_visible`: highest expected lag because it raises smoothing the most

Example only, assuming `base_motion_smoothing = 0.50`:

- `stable` -> target `smooth = 0.40` -> position params `min_cutoff_hz = 2.26`, `beta = 0.158`; rotation params `min_cutoff_hz = 2.98`, `beta = 0.208`
- `recovering` -> target `smooth = 0.50` -> position params `min_cutoff_hz = 1.95`, `beta = 0.190`; rotation params `min_cutoff_hz = 2.60`, `beta = 0.250`
- `low_trust` -> target `smooth = 0.60` -> position params `min_cutoff_hz = 1.64`, `beta = 0.222`; rotation params `min_cutoff_hz = 2.22`, `beta = 0.292`
- `jittery_visible` -> target `smooth = 0.65` -> position params `min_cutoff_hz = 1.49`, `beta = 0.238`; rotation params `min_cutoff_hz = 2.03`, `beta = 0.313`

Because One Euro filters adapt to signal speed, exact lag in milliseconds is not a single constant that can be read directly from the config alone. The honest interpretation is that higher modes increase damping and therefore usually increase apparent lag, but the actual delay depends on frame rate and how quickly the pose is moving at that moment.
Also note that the tuner does not jump straight to the target in one step unless the current value is already close. It moves by `motion_smoothing_step`, so the full transition into a new mode can take several update cycles.

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
When you see `ArUco: pose lost | holding`, it means the pose system just lost marker-based visibility, but it is still inside the pose-loss hold window. During that window, the system can keep the last valid numeric pose values internally for continuity even though the pose is already marked invalid for the current frame.

### How long `holding` lasts

From the uploaded files, this duration is described as a **short configurable hold window**, but the exact parameter name and source file that define that pose-loss hold duration are **not present in the uploaded files**. In other words:

- the behavior is documented here
- the exact symbol that configures it was not available to verify
- the exact duration in frames or seconds therefore cannot be stated with confidence from the provided files alone

So `holding` lasts **until that pose-loss hold window expires**, but this document cannot honestly name the exact parameter unless the pose-loss implementation file is provided.

## `[hold]`
When you see `[hold]` appended to a tracked object label, it means the ID-flicker continuity layer is currently forwarding the last stable tracked object instead of using a fresh accepted observation from the current frame.

### How long `[hold]` lasts

This visible state is tied to the continuity layer's coast duration.

The runtime value is printed as `coast=<N>` in the adaptive tuning log line, and the tuner-side parameters visible in `src/adaptive_tuning.rs` are:

- `current_coast_frames`
- `base_coast_frames`
- `id_coast_frames_min`
- `id_coast_frames_max`
- `id_coast_step`

So `[hold]` can last for **up to the current `coast_frames` budget**, measured in frames after the last accepted observation, unless one of these happens sooner:

- a fresh detection is accepted again
- the object is dropped entirely
- the coast budget is exhausted

To convert that to time:

```text
hold time (seconds) ≈ coast_frames / runtime_fps
```

Examples:

- `coast=5` at `30 FPS` -> about `0.17 s`
- `coast=5` at `10 FPS` -> about `0.50 s`

## `coasted`
Internally, `coasted` is the continuity-state name for a held-forward track. The visible on-screen sign of this state is the `[hold]` suffix.

### How long `coasted` lasts

`coasted` lasts for the **same duration** as `[hold]`, because `[hold]` is just the visible rendering of that internal state.

So, from `src/adaptive_tuning.rs`, the duration is controlled indirectly by the current continuity coast budget, with the relevant tuner-side parameters being:

- `current_coast_frames`
- `base_coast_frames`
- `id_coast_frames_min`
- `id_coast_frames_max`
- `id_coast_step`

The actual time in seconds still depends on current runtime FPS.

## `observed`
Internally, a tracked detection marked as `observed` is a fresh accepted detection rather than a coasted one. This internal state is not printed directly on the video frame.

### How long `observed` lasts

From the uploaded files, `observed` should be read as a **current emission-state label**, not as a separate timed hold window.

So the safest interpretation is:

- `observed` applies to the current accepted detection output
- on the next frame, that same track may again be `observed`, may switch to `coasted`, or may disappear
- no separate `observed_duration` parameter is visible in the uploaded files

In practical terms, `observed` lasts for the current accepted output step, then gets recomputed again on the next frame/update.

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



## ORB-SLAM fusion status block

When `ORBSLAM_STATUS_OVERLAY_ENABLED` is on, the runtime draws a multi-line block in the top-left corner of the frame. This is the failure-handling overlay for the middle-man.

Typical lines are:

```text
SLAM: OK
Match: FRAME_ID
Projection: OK (1/1)
```

or, during failure conditions:

```text
SLAM: MISSING
Match: NONE
Projection: UNAVAILABLE (0/1)
no aligned ORB-SLAM pose; waiting for ORB-SLAM UDP packets
```

Interpretation:

- `SLAM`: whether the middle-man found a usable aligned ORB-SLAM pose, or whether the UDP feed is missing or stale
- `Match`: whether that alignment came from exact `frame_id`, timestamp fallback, latest-sample fallback, or none
- `Projection`: whether eligible detections could be projected to the configured ground plane
- final line: short failure reason surfaced directly from `fusion_status.reason`
