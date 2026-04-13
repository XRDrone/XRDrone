# UDP JSON Structure

This document describes the detector + ORB-SLAM fusion packet that is sent to Unity. The middle-man aligns detector output and ORB-SLAM UDP pose packets by `frame_id` first and timestamp second, then projects foot points onto the configured ground plane before publishing a combined UDP packet.

## Packet Sample

```json
{
  "frame_id": 50,
  "timestamp": 1772154677.198175,
  "width": 1920,
  "height": 1080,
  "detections": [
    {
      "id": 1,
      "cls": 0,
      "conf": 0.91,
      "cx": 0.46,
      "cy": 0.63,
      "w": 0.12,
      "h": 0.38,
      "foot_x": 0.46,
      "foot_y": 0.82,
      "world_valid": true,
      "world_x": 1.38,
      "world_y": 0.0,
      "world_z": -2.41
    }
  ],
  "pose": {
    "x": 1.22,
    "altitude": 1.63,
    "z": -2.05,
    "yaw": 4.8,
    "pitch": -1.2,
    "roll": 0.5,
    "hfov": 84.0,
    "markers_used": 0,
    "pose_valid": true
  },
  "slam": {
    "tracking_state": "ok",
    "match_mode": "frame_id",
    "pose_valid": true,
    "frame_id": 50,
    "timestamp": 1772154677.198175,
    "x": 1.22,
    "y": 1.63,
    "z": -2.05,
    "qx": 0.0,
    "qy": 0.02,
    "qz": 0.0,
    "qw": 0.9998
  },
  "fusion_status": {
    "source": "orbslam",
    "slam_tracking": "ok",
    "match_mode": "frame_id",
    "projection_state": "ok",
    "pose_valid": true,
    "projection_attempted": 1,
    "projection_projected": 1,
    "reason": ""
  }
}
```

## Top-Level Fields

### `frame_id`
Sequential detector frame counter used as the primary alignment key.

### `timestamp`
Detector-frame processing time in Unix epoch seconds. This is the secondary alignment key when an exact ORB-SLAM `frame_id` is not available.

### `width`, `height`
Frame resolution in pixels. These values are required to interpret normalized box and foot-point coordinates and to reconstruct camera intrinsics from `hfov`.

### `detections`
Detector-branch objects after fusion. Each detection still carries image-space geometry plus `foot_*` and `world_*` fields. The `world_*` coordinates are now produced by the ORB-SLAM middle-man rather than the old marker-based path.

### `pose`
Compatibility camera-pose object for existing Unity consumers. It is derived from the matched ORB-SLAM pose and keeps the previous field names (`x`, `altitude`, `z`, `yaw`, `pitch`, `roll`, `hfov`, `markers_used`, `pose_valid`). `markers_used` remains `0` because ORB-SLAM, not ArUco, is driving the pose.

### `slam`
Raw ORB-SLAM transport object used by the middle-man and available to Unity.

- `tracking_state`: sender or fusion status such as `ok`, `stale`, `missing`, or `lost`
- `match_mode`: `frame_id`, `timestamp`, `latest`, or `none`
- `pose_valid`: whether a pose was matched
- `frame_id`, `timestamp`: the matched ORB-SLAM sample identifiers, or `null` when unavailable
- `x`, `y`, `z`, `qx`, `qy`, `qz`, `qw`: raw ORB-SLAM camera pose values

### `fusion_status`
Middle-man status block for runtime diagnostics and UI overlays.

- `source`: currently `orbslam`
- `slam_tracking`: `ok` or `missing`
- `match_mode`: same value exposed in `slam.match_mode`
- `projection_state`: `ok`, `partial`, `idle`, or `unavailable`
- `pose_valid`: whether the projection step had a usable aligned pose
- `projection_attempted`: detections that were eligible for world projection
- `projection_projected`: detections that successfully intersected the configured ground plane
- `reason`: failure text surfaced in the on-screen status block

## Detection Object Fields

Each `detections[i]` entry contains:

- `id`: track ID
- `cls`: Unity class ID
- `conf`: confidence in `[0, 1]`
- `cx`, `cy`, `w`, `h`: normalized image-space box geometry
- `foot_x`, `foot_y`: normalized bottom-center point used for projection
- `world_valid`: `true` only when the middle-man had an aligned ORB-SLAM pose and the back-projected foot ray intersected the configured ground plane
- `world_x`, `world_y`, `world_z`: fused world coordinates in the ORB-SLAM world frame

## Fusion Logic

1. The detector branch processes the current MediaMTX-backed frame and produces detections.
2. The ORB-SLAM branch sends UDP JSON packets with `frame_id`, `timestamp`, `pose_valid`, `tracking_state`, `x`, `y`, `z`, `qx`, `qy`, `qz`, and `qw`.
3. The middle-man listens on `ORBSLAM_UDP_LISTEN_IP:ORBSLAM_UDP_PORT`, stores recent ORB-SLAM packets in memory, looks for an exact `frame_id` match, and falls back to the nearest timestamp within `ORBSLAM_MATCH_TIME_TOLERANCE_S`.
4. If a valid pose is available, the middle-man reconstructs intrinsics from `POSE_HFOV_DEG`, turns each foot point into a camera ray, rotates that ray into the ORB-SLAM world frame, intersects it with `ORBSLAM_GROUND_PLANE_Y`, and writes `world_*` back into the detection.
5. If no pose is aligned, detections still ship, but `world_valid` stays `false` and `fusion_status.reason` explains why.

## Failure Handling

The runtime draws the `fusion_status` block in the top-left corner of the output frame. Typical states are:

- `SLAM: MISSING` when no ORB-SLAM packets have arrived yet
- `Match: TIMESTAMP` when the middle-man had to fall back to time alignment instead of exact `frame_id`
- `SLAM: STALE` when packets stopped arriving within `ORBSLAM_PACKET_STALE_TIMEOUT_S`
- `Projection: UNAVAILABLE` when no valid pose exists for the frame
- `Projection: PARTIAL (n/m)` when only some eligible detections intersected the configured ground plane

This overlay is informational only. It does not block packet emission.
