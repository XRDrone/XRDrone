# UDP JSON Structure

This document describes the UDP JSON packet structure used by the pipeline, including the top-level packet fields, detection object fields, pose fields, and how related fields should be interpreted.

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
      "conf": 0.8389231562614441,
      "cx": 0.4600933710734049,
      "cy": 0.6266968056007668,
      "w": 0.6039250691731771,
      "h": 0.7393832171404803,
      "foot_x": 0.0,
      "foot_y": 0.0,
      "world_valid": false,
      "world_x": 0.0,
      "world_y": 0.0,
      "world_z": 0.0
    }
  ],
  "pose": {
    "x": 0.0,
    "altitude": 0.0,
    "z": 0.0,
    "yaw": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
    "hfov": 84.0,
    "markers_used": 0,
    "pose_valid": false
  }
}
```

## Top-Level Fields

### `frame_id`
Sequential frame counter from the pipeline loop. Increments by `1` per processed frame.

### `timestamp`
Wall-clock time in Unix epoch seconds, including fractional precision, when the frame was processed.

Used to align detections and pose with video time.

### `width`, `height`
Frame resolution in pixels.

These values are used to interpret normalized image-space quantities such as `cx`, `cy`, `w`, `h`, `foot_x`, and `foot_y`.

### `detections`
Array of detected objects for the frame.

Each entry contains object tracking, classification, confidence, image-space box geometry, optional foot-point information, and optional world projection.

### `pose`
Camera or drone pose estimate for the frame, derived from markers.

This object is always included, but it may be invalid when `pose_valid` is `false`.

## Detection Object Fields

Each element in `detections[i]` has the following fields.

### `id`
Persistent object ID, also referred to as the track ID.

When tracking is stable, the same physical object should keep the same `id` across frames.

### `cls`
Integer class ID for the object.

This is intended for the consumer-side class mapping. For example, `0` may represent `person` depending on the configured mapping.

### `conf`
Detection confidence score in the range `[0, 1]`.

### `cx`, `cy`
Normalized bounding-box center in image coordinates:

- `cx = center_x / width`
- `cy = center_y / height`

Interpretation:

- `0.0` is the left or top edge
- `1.0` is the right or bottom edge

### `w`, `h`
Normalized bounding-box size:

- `w = box_width / width`
- `h = box_height / height`

These values are typically in the range `[0, 1]`.

### Example interpretation

If:

- `cx = 0.46`
- `cy = 0.63`
- `w = 0.60`
- `h = 0.74`

then the object center is about `46%` across the image and `63%` down the image, while the bounding box covers about `60%` of the image width and `74%` of the image height.

### `foot_x`, `foot_y`
Normalized bottom-center point of the bounding box, also called the foot point.

This is typically used as the point where the object touches the ground.

Definitions:

- `foot_x = foot_x_px / width`
- `foot_y = foot_y_px / height`

where:

- `foot_x_px = (x1 + x2) / 2`
- `foot_y_px = y2`

If `foot_x` and `foot_y` are unavailable, a consumer can derive the bottom-center point from `cx`, `cy`, `w`, and `h`.

### `world_valid`
Boolean indicating whether the pipeline successfully projected the detection foot point into world space for the current frame.

- `true`: world projection is available
- `false`: world projection is unavailable or unreliable

### `world_x`, `world_y`, `world_z`
World-space coordinates of the detection foot point in the ArUco or world reference frame.

These are computed by back-projecting the foot pixel into a world-space ray using the camera pose and intrinsics, then intersecting that ray with the ground plane.

When `world_valid` is `false`, these values are set to `0.0`.

## Cross-Field Relationship

`world_valid` depends on both of the following:

1. `pose.pose_valid == true` for the frame
2. A successful ray-plane intersection for the detection foot point

## Pose Object Fields

### `x`, `altitude`, `z`
Camera position in the world coordinate system.

Units depend on the marker scale and are typically meters.

- `x`: world X position
- `altitude`: world Y position, or height
- `z`: world Z position

### `yaw`, `pitch`, `roll`
Camera orientation angles in degrees.

The exact convention depends on the implementation, but typically:

- `yaw`: rotation around the vertical axis
- `pitch`: tilt up or down
- `roll`: tilt left or right

### `hfov`
Horizontal field of view, in degrees.

Used to approximate camera intrinsics for pose solving.

### `markers_used`
Number of detected markers that were actually used to compute pose for the current frame.

### `pose_valid`
Boolean indicating whether pose estimation succeeded for the frame.

- `true`: a reliable pose was computed
- `false`: pose estimation failed or was not reliable, commonly because no markers were detected or pose solving failed

## Notes for Consumers

- Bounding-box geometry fields are normalized and must be interpreted using the packet `width` and `height`.
- The `pose` object is always present, but consumers must check `pose_valid` before trusting pose values.
- World coordinates must only be used when `world_valid` is `true`.
- Object identity across frames depends on tracker stability, so `id` should be treated as a persistent track identifier rather than a per-frame index.
