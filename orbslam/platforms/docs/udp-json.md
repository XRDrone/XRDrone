# ORB-SLAM3 Fusion UDP JSON Structure

This document describes the final JSON packet produced by `rtsp_yolo_orbslam_fusion.py` before it is sent to Unity.

The packet combines:

1. detector frame metadata,
2. YOLO person detections,
3. a compatibility `pose` object,
4. a raw ORB-SLAM3 `slam` object,
5. a diagnostic `fusion_status` object.

---

## 1. ORB-SLAM3 input pose format

The Windows fusion script expects ORB-SLAM3 pose packets over UDP in this default text format:

```text
frame timestamp x y z qx qy qz qw
```

Field meanings:

| Field | Meaning |
|---|---|
| `frame` | ORB-SLAM3 frame ID. |
| `timestamp` | ORB-SLAM3 pose timestamp. |
| `x`, `y`, `z` | ORB-SLAM3 camera position after optional coordinate mapping. |
| `qx`, `qy`, `qz`, `qw` | ORB-SLAM3 camera orientation as a quaternion. |

The script can also parse JSON pose packets when `--pose-format json` is used, but the default is:

```text
--pose-format orbslam-text
```

---

## 2. Final Unity packet sample

```json
{
  "frame_id": 50,
  "timestamp": 1772154677.198175,
  "width": 1920,
  "height": 1080,
  "detections": [
    {
      "id": 12,
      "cls": 0,
      "conf": 0.91,
      "x1": 811.2,
      "y1": 492.4,
      "x2": 1037.6,
      "y2": 902.1,
      "cx": 924.4,
      "cy": 697.25,
      "w": 226.4,
      "h": 409.7,
      "foot_x": 924.4,
      "foot_y": 902.1,
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
    "yaw": 0.084,
    "pitch": -0.021,
    "roll": 0.009,
    "hfov": 70.0,
    "markers_used": 0,
    "pose_valid": true
  },
  "slam": {
    "tracking_state": "ok",
    "match_mode": "latest",
    "pose_valid": true,
    "frame_id": 140,
    "timestamp": 1772154677.151,
    "pose_age_s": 0.047,
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
    "match_mode": "latest",
    "projection_state": "ok",
    "pose_valid": true,
    "projection_attempted": 1,
    "projection_projected": 1,
    "reason": ""
  }
}
```

---

## 3. Top-level fields

| Field | Type | Meaning |
|---|---|---|
| `frame_id` | integer | Windows detector/fusion frame counter. |
| `timestamp` | float | Windows processing time in Unix epoch seconds. |
| `width` | integer | Frame width in pixels. |
| `height` | integer | Frame height in pixels. |
| `detections` | array | YOLO person detections after optional ORB-SLAM3 world projection. |
| `pose` | object | Compatibility camera-pose object for Unity receivers that already expect a `pose` block. |
| `slam` | object | Raw ORB-SLAM3 pose state exposed to Unity and logs. |
| `fusion_status` | object | Diagnostic state for pose availability and projection status. |

---

## 4. Detection object

Each entry in `detections` contains:

| Field | Type | Meaning |
|---|---|---|
| `id` | integer | YOLO track ID when available, otherwise a generated per-frame fallback ID. |
| `cls` | integer | Class ID. For person-only runs this is usually `0`. |
| `conf` | float | YOLO confidence in `[0, 1]`. |
| `x1`, `y1`, `x2`, `y2` | float | Pixel-space bounding-box corners. |
| `cx`, `cy` | float | Pixel-space bounding-box center. |
| `w`, `h` | float | Pixel-space bounding-box width and height. |
| `foot_x`, `foot_y` | float | Pixel-space bottom-center point used for ground-plane projection. |
| `world_valid` | boolean | `true` when the detection was successfully projected into ORB-SLAM3 world space. |
| `world_x`, `world_y`, `world_z` | float or null | Projected ground-plane world position. These are `null` when `world_valid` is `false`. |

---

## 5. Compatibility `pose` object

The `pose` object keeps a simple camera-pose interface for Unity-side compatibility.

| Field | Type | Meaning |
|---|---|---|
| `x` | float or null | Camera x position. |
| `altitude` | float or null | Camera y position exposed under the older altitude-style name. |
| `z` | float or null | Camera z position. |
| `yaw`, `pitch`, `roll` | float or null | Euler orientation derived from the ORB-SLAM3 quaternion. Values are radians unless `--pose-angles-in-degrees` changes projection interpretation. |
| `hfov` | float | Horizontal field of view used for projection. |
| `markers_used` | integer | Always `0` for ORB-SLAM3 because this path does not use ArUco markers. |
| `pose_valid` | boolean | Whether a fresh ORB-SLAM3 pose was available for this detector frame. |

---

## 6. Raw `slam` object

The `slam` object exposes the newest ORB-SLAM3 pose sample used by the fusion script.

| Field | Type | Meaning |
|---|---|---|
| `tracking_state` | string | `ok`, `stale`, or `missing`. |
| `match_mode` | string | `latest` when the newest fresh pose is used, otherwise `none`. |
| `pose_valid` | boolean | Whether the pose is fresh and usable for projection. |
| `frame_id` | integer or null | ORB-SLAM3 frame ID from the UDP pose packet. |
| `timestamp` | float or null | ORB-SLAM3 timestamp from the UDP pose packet. |
| `pose_age_s` | float or null | Age of the pose relative to the Windows processing time. |
| `x`, `y`, `z` | float or null | ORB-SLAM3 camera position. |
| `qx`, `qy`, `qz`, `qw` | float or null | ORB-SLAM3 quaternion orientation. |

---

## 7. `fusion_status` object

The `fusion_status` object explains whether pose and projection were usable for the current frame.

| Field | Type | Meaning |
|---|---|---|
| `source` | string | Always `orbslam` for this path. |
| `slam_tracking` | string | `ok`, `stale`, or `missing`. |
| `match_mode` | string | `latest` or `none`. |
| `projection_state` | string | `ok`, `partial`, `idle`, or `unavailable`. |
| `pose_valid` | boolean | Whether a fresh ORB-SLAM3 pose was available. |
| `projection_attempted` | integer | Number of detections eligible for projection. |
| `projection_projected` | integer | Number of detections successfully projected to world space. |
| `reason` | string | Empty when the frame is healthy, otherwise a short explanation. |

---

## 8. Common runtime states

### Healthy pose and projection

```json
{
  "slam_tracking": "ok",
  "match_mode": "latest",
  "projection_state": "ok",
  "pose_valid": true,
  "reason": ""
}
```

### No detections, but pose is healthy

```json
{
  "slam_tracking": "ok",
  "match_mode": "latest",
  "projection_state": "idle",
  "pose_valid": true,
  "reason": "no eligible detections to project"
}
```

### Pose missing or stale

```json
{
  "slam_tracking": "missing",
  "match_mode": "none",
  "projection_state": "unavailable",
  "pose_valid": false,
  "reason": "no fresh ORB-SLAM pose; waiting for ORB-SLAM UDP packets"
}
```

---

## 9. Notes for Unity

- The detection geometry fields are pixel-space values, not normalized values.
- `world_valid` should be checked before using `world_x`, `world_y`, or `world_z`.
- Unity can continue rendering detections when pose is missing, but world-space placement should be disabled or marked unavailable.
- `pose.markers_used` is always `0` because ORB-SLAM3 is markerless in this path.
- `packets_log.jsonl` records the exact packet sent to Unity for each processed frame when `--logs` is enabled.
