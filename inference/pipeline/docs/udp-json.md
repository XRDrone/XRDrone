# UDP JSON Structure

This document describes the UDP JSON packet structure used by the human-detection branch.

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
      "foot_x": 0.4600933710734049,
      "foot_y": 0.9963888888888889
    }
  ]
}
```

## Top-Level Fields

### `frame_id`
Sequential frame counter from the pipeline loop. Increments by `1` per processed frame.

### `timestamp`
Wall-clock time in Unix epoch seconds, including fractional precision, when the frame was processed.

### `width`, `height`
Frame resolution in pixels.

These values are used to interpret normalized image-space quantities such as `cx`, `cy`, `w`, `h`, `foot_x`, and `foot_y`.

### `detections`
Array of detected objects for the frame.

Each entry contains tracking, classification, confidence, image-space box geometry, and a normalized foot point.

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

### `w`, `h`
Normalized bounding-box size:

- `w = box_width / width`
- `h = box_height / height`

### `foot_x`, `foot_y`
Normalized bottom-center point of the bounding box, also called the foot point.

Definitions:

- `foot_x = foot_x_px / width`
- `foot_y = foot_y_px / height`

where:

- `foot_x_px = (x1 + x2) / 2`
- `foot_y_px = y2`

If `foot_x` and `foot_y` are unavailable, a consumer can derive the bottom-center point from `cx`, `cy`, `w`, and `h`.

## Notes for Consumers

- Bounding-box geometry fields are normalized and must be interpreted using the packet `width` and `height`.
- The packet does not include pose or world-coordinate data.
- Object identity across frames depends on tracker stability, so `id` should be treated as a persistent track identifier rather than a per-frame index.
