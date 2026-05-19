# XRDrone Pure ArUco Video Runner

This version runs the prerecorded-video ArUco marker pipeline only. It does not use ORB-SLAM.

## Marker Layout

The default marker world coordinates in `settings.py` are:

| Marker ID | X | Y | Z |
|---:|---:|---:|---:|
| 0 | 0.00 | 0.00 | 0.00 |
| 1 | -3.03 | 0.00 | 0.00 |
| 2 | -5.00 | 0.00 | 0.00 |

The pose solver assumes the markers lie on the `Y=0` plane and that coordinates are in meters.

## Normal Run

Put `2026_05_18_15_28_04_Cache_Trimmed.mp4` in the same folder as `main.py`, then run:

```powershell
python main.py
```

This runs the ArUco pipeline normally and does **not** create a log folder.

## Run With Logs

Use `--logs` when you want the per-frame logs:

```powershell
python main.py --logs
```

Headless/logging run:

```powershell
python main.py --logs --no-gui
```

Use a different video path if needed:

```powershell
python main.py --video ".\2026_05_18_15_28_04_Cache_Trimmed.mp4" --logs
```

Run only ArUco pose without YOLO detections:

```powershell
python main.py --no-detect
```

Run only ArUco pose with logs:

```powershell
python main.py --no-detect --logs
```

## Output Logs

Logs are created only when `--logs` or `--log` is passed.

Each logging run creates a folder like:

```text
logs/run_YYYYMMDD_HHMMSS/
```

Files written per logging run:

- `run_metadata.json` — input path, marker layout, model status, and run configuration.
- `summary.json` — total frames, runtime FPS, pose-valid ratio, and total detections.
- `pose_log.jsonl` — one pose record per frame with `frame_id`, timestamps, markers used, camera position, quaternion, rvec/tvec, and reprojection error.
- `marker_log.jsonl` — detected marker IDs and marker image corners per frame.
- `detections_log.jsonl` — one detection record per frame with timestamps, frame size, bounding boxes, confidence, class, track ID, and foot-point coordinates.
- `packets_log.jsonl` — full packet written/sent per frame.
- `frames_log.csv` — compact frame-by-frame summary.
- `errors_log.jsonl` — UDP/runtime errors if any occur.

## Detection Model Behavior

The runner tries to load `../models/yolo26n-seg.pt` by default. If the model is not present, the program still runs and uses empty detection arrays. This keeps ArUco pose processing usable even when the YOLO weights are missing.

Use a different model path with:

```powershell
python main.py --model "..\models\your_model.pt"
```

## Smoke Tests

Normal run, no logs:

```powershell
python main.py --no-gui --no-detect --max-frames 30
```

Logging run:

```powershell
python main.py --logs --no-gui --no-detect --max-frames 30
```
