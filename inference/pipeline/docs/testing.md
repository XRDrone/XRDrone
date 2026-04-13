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
