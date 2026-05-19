# UDP Validation and Testing

This document explains how to validate the detector + ORB-SLAM middle-man using `test_with_coverage.py`.

The test script now checks that:

- the combined UDP packet schema matches the documented fusion contract
- detector fields, compatibility `pose`, raw `slam`, and `fusion_status` are all present
- world projection fields can be populated through the ORB-SLAM middle-man path
- UDP send/receive still works correctly over localhost

## Core Test Modes

### Default mode

```bash
python test_with_coverage.py
```

This mode performs two checks:

1. **Formatter + fusion schema test**
   - Builds a sample detector packet.
   - Builds a sample ORB-SLAM UDP pose packet in the same JSON shape expected from the external sender.
   - Runs the ORB-SLAM projection helper to populate `foot_*` and `world_*`.
   - Adds `pose`, `slam`, and `fusion_status`.
   - Validates the exact packet structure.

2. **UDP loopback transport test**
   - Sends the resulting JSON packet over UDP to localhost.
   - Receives it back.
   - Re-validates the full packet structure after transport.

Success output remains one `PASSED:` line and exit code `0`.

### Live mode

```bash
python test_with_coverage.py --live --packets 5 --timeout 8
```

Use this while the pipeline is running against your real MediaMTX stream. The listener validates that each received packet still contains the fusion keys and valid types.

### Stats mode

```bash
python test_with_coverage.py --stats --packets 120 --timeout 8
```

This mode still computes packet sizes, timing jitter, frame gaps, and estimated ID switches. The packet validation step now covers the fusion fields as well.



## Runtime Log Testing

The runtime log feature is enabled with the `--logs` flag. Logs are optional so that normal runs do not create extra output folders.

### Prerecorded video with logs

Use this mode when validating a saved MP4 test video:

```bash
python main.py --video "2026_05_18_15_28_04_Cache_Trimmed.mp4" --logs
```

For a full absolute path on macOS:

```bash
python main.py --video "/Users/troy/Desktop/XRDrone/inference/pipeline/2026_05_18_15_28_04_Cache_Trimmed.mp4" --logs
```

For a headless run that only writes logs and does not open the OpenCV display window:

```bash
python main.py --video "/Users/troy/Desktop/XRDrone/inference/pipeline/2026_05_18_15_28_04_Cache_Trimmed.mp4" --logs --no-gui
```

### Live video with logs

Use this mode when the pipeline is reading from the configured live camera or capture-card input:

```bash
python main.py --logs
```

For headless live logging:

```bash
python main.py --logs --no-gui
```

The live input source is controlled by `settings.py`:

```python
INPUT_MODE = "camera"
VIDEO_SOURCE = 0
```

If the capture card is not camera `0`, test camera indices and update `VIDEO_SOURCE` to the index that opens successfully.

### Normal run without logs

Running without `--logs` should process video normally without creating a new runtime log folder:

```bash
python main.py
```

For a prerecorded video without logs:

```bash
python main.py --video "2026_05_18_15_28_04_Cache_Trimmed.mp4"
```

### Expected log folder

When `--logs` is enabled, each run creates a timestamped folder under:

```text
logs/run_YYYYMMDD_HHMMSS/
```

Expected files include:

- `run_metadata.json` — input source, marker layout, model status, and run configuration.
- `summary.json` — processed frame count, runtime FPS, pose-valid ratio, and detection totals.
- `pose_log.jsonl` — per-frame pose records with timestamps, frame IDs, marker usage, camera position, quaternion, rvec/tvec, and reprojection error.
- `marker_log.jsonl` — detected marker IDs, image corners, known/unknown marker status, and rejected marker count.
- `detections_log.jsonl` — per-frame detection records with frame IDs, timestamps, bounding boxes, confidence, class labels, track IDs, and foot-point coordinates.
- `packets_log.jsonl` — the complete packet generated for each processed frame.
- `frames_log.csv` — compact per-frame summary with pose validity, marker counts, detection count, and processing time.
- `errors_log.jsonl` — runtime or UDP errors, if any occur.

### Quick log verification

After running with `--logs`, check that the newest log folder exists:

```bash
ls -lt logs | head
```

Inspect the summary:

```bash
cat logs/run_*/summary.json
```

Inspect the first few pose and detection records:

```bash
head -5 logs/run_*/pose_log.jsonl
head -5 logs/run_*/detections_log.jsonl
```

A successful logged run should show one pose record and one detection record for each processed frame. If detections are disabled or the model is unavailable, `detections_log.jsonl` should still exist, but each frame may contain an empty `detections` list.

## What to Verify During a Real Fusion Run

When ORB-SLAM is connected, a healthy run should show:

- `slam.pose_valid = true` for aligned frames
- `fusion_status.match_mode = "frame_id"` most of the time
- `fusion_status.projection_state = "ok"` or `"partial"`
- `world_valid = true` on projected person detections

When the ORB-SLAM sender is stale or missing, expected fallback behavior is:

- `slam.pose_valid = false`
- `fusion_status.slam_tracking = "missing"`
- `fusion_status.projection_state = "unavailable"`
- detections still transmit, but `world_valid = false`

## MediaMTX + ORB-SLAM Workflow

A typical end-to-end test setup is:

1. Start MediaMTX and publish the camera or video stream.
2. Run the detector branch against that stream.
3. Run ORB-SLAM against the same stream and have it send UDP pose packets to `ORBSLAM_UDP_LISTEN_IP:ORBSLAM_UDP_PORT`.
4. Run `python test_with_coverage.py --live --packets 5 --timeout 8` to confirm that the middle-man is producing the expected combined packets.

The live test does not need direct access to the ORB-SLAM sender. It validates the final UDP contract that Unity will receive.
