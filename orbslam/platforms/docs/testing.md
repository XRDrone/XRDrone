# ORB-SLAM3 Fusion Testing and Runtime Logs

This document explains how to test the Windows-side ORB-SLAM3 fusion runtime and how to collect logs that can be compared with the ArUco-side runtime logs.

The relevant files are:

```text
rtsp_yolo_orbslam_fusion.py
orbslam_fusion_testing.py
```

`rtsp_yolo_orbslam_fusion.py` runs the actual RTSP + YOLO + ORB-SLAM pose fusion process. `orbslam_fusion_testing.py` is imported by the runtime only when logging is enabled with `--logs`.

---

## 1. What the ORB-SLAM3 fusion runtime does

The Windows-side fusion runtime:

1. reads the RTSP stream published from the Linux machine,
2. receives ORB-SLAM3 pose packets over UDP,
3. runs YOLO person detection/tracking,
4. projects detected person foot-points onto the configured ground plane when a fresh pose is available,
5. sends the final fused UDP JSON packet to Unity,
6. optionally writes testing logs when `--logs` is enabled.

Expected ORB-SLAM3 text pose packet format:

```text
frame timestamp x y z qx qy qz qw
```

---

## 2. Live logging test

Use this when the RTSP stream is already available or when you expect the stream to be available immediately.

```powershell
python .\rtsp_yolo_orbslam_fusion.py --logs
```

Recommended explicit version:

```powershell
python .\rtsp_yolo_orbslam_fusion.py `
  --rtsp-url "rtsp://127.0.0.1:8554/dji" `
  --model ".\yolov5nu.pt" `
  --pose-listen-ip "0.0.0.0" `
  --pose-port 9000 `
  --pose-format "orbslam-text" `
  --output-host "127.0.0.1" `
  --output-port 9002 `
  --device 0 `
  --logs `
  --show
```

For CPU-only testing:

```powershell
--device cpu
```

The logger starts when the first valid frame is processed and continues until the program exits.

---

## 3. Wait-for-video logging test

Use this when the Windows process may start before the Linux machine begins publishing the RTSP stream.

```powershell
python .\rtsp_yolo_orbslam_fusion.py --logs --wait-for-video
```

Recommended explicit version:

```powershell
python .\rtsp_yolo_orbslam_fusion.py `
  --rtsp-url "rtsp://127.0.0.1:8554/dji" `
  --model ".\yolov5nu.pt" `
  --pose-listen-ip "0.0.0.0" `
  --pose-port 9000 `
  --pose-format "orbslam-text" `
  --output-host "127.0.0.1" `
  --output-port 9002 `
  --device 0 `
  --logs `
  --wait-for-video `
  --show
```

By default, `--wait-for-video` waits forever. To stop waiting after a fixed amount of time:

```powershell
--wait-timeout-s 60
```

The logger does not finalize a run until at least one frame has been processed. This keeps empty pre-stream waiting time out of the runtime statistics.

---

## 4. Headless logging

To run without the OpenCV preview window:

```powershell
python .\rtsp_yolo_orbslam_fusion.py --logs --no-show
```

Wait-for-video plus headless mode:

```powershell
python .\rtsp_yolo_orbslam_fusion.py --logs --wait-for-video --no-show
```

---

## 5. Expected log folder

When `--logs` is enabled, the runtime creates a timestamped folder:

```text
logs/run_YYYYMMDD_HHMMSS/
```

Expected files:

```text
run_metadata.json
summary.json
pose_log.jsonl
detections_log.jsonl
packets_log.jsonl
frames_log.csv
errors_log.jsonl
```

There is no `marker_log.jsonl` for the ORB-SLAM3 path because ORB-SLAM3 does not send ArUco marker IDs, marker corners, rejected marker counts, or ArUco reprojection error.

---

## 6. Log file descriptions

| File | Purpose |
|---|---|
| `run_metadata.json` | Run configuration, RTSP input source, model settings, UDP ports, and log schema information. |
| `summary.json` | Aggregate statistics for runtime FPS, pose availability, detections, world projection, UDP packets, RTSP reconnects, and errors. |
| `pose_log.jsonl` | One record per processed frame describing the latest ORB-SLAM3 pose used by that frame. |
| `detections_log.jsonl` | One record per processed frame with YOLO detections, tracking IDs, foot-points, and world-projection results. |
| `packets_log.jsonl` | One record per processed frame containing the exact JSON packet sent to Unity. |
| `frames_log.csv` | Compact per-frame CSV summary for direct comparison across test runs. |
| `errors_log.jsonl` | Runtime errors, malformed pose packets, RTSP failures, UDP send failures, YOLO errors, and projection errors. |

---

## 7. Main summary statistics

`summary.json` includes:

- processed frame count,
- runtime duration,
- average runtime FPS,
- average/min/median/max frame processing time,
- pose-valid frame count,
- pose-valid ratio,
- ORB-SLAM3 pose packets received,
- ORB-SLAM3 pose packets parsed,
- malformed ORB-SLAM3 pose packets,
- average and maximum pose age,
- total detections,
- frames with detections,
- average detections per frame,
- unique track IDs,
- world-projection attempts,
- world-projection successes,
- world-projection success ratio,
- UDP packet count,
- UDP send error count,
- packet-size statistics,
- RTSP read failures,
- RTSP reconnect count,
- total runtime error count.

---

## 8. Quick verification commands

After a logged run:

```powershell
Get-ChildItem .\logs | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

Inspect the newest summary:

```powershell
Get-Content .\logs\run_*\summary.json
```

Inspect the first few pose, detection, and packet records:

```powershell
Get-Content .\logs\run_*\pose_log.jsonl -TotalCount 5
Get-Content .\logs\run_*\detections_log.jsonl -TotalCount 5
Get-Content .\logs\run_*\packets_log.jsonl -TotalCount 5
```

On macOS/Linux-style shells:

```bash
ls -lt logs | head
cat logs/run_*/summary.json
head -5 logs/run_*/pose_log.jsonl
head -5 logs/run_*/detections_log.jsonl
head -5 logs/run_*/packets_log.jsonl
```

---

## 9. Healthy run indicators

A healthy run should generally show:

- `pose.pose_valid = true` for frames with fresh ORB-SLAM3 pose,
- `slam.tracking_state = "ok"`,
- `fusion_status.slam_tracking = "ok"`,
- `fusion_status.projection_state = "ok"` or `"partial"` when people are detected,
- `world_valid = true` on detections that successfully project to the ground plane,
- stable runtime FPS,
- low RTSP reconnect count,
- low malformed pose packet count.

If pose packets are missing or stale, detections still send to Unity, but `world_valid` remains `false` and `fusion_status.reason` explains why projection was unavailable.
