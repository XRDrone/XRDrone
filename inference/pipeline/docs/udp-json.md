# UDP JSON Structure

This document describes the UDP JSON packet structure used by the human-detection branch of the pipeline. The packet contains frame metadata and image-space detection data only. It does not include pose data or world-coordinate projection fields.

## Top-Level Packet Example

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
      "foot_y": 0.0
    }
  ]
}
```

## Top-Level Fields

### `frame_id`
Integer frame counter for the emitted packet.

Consumers can use this to align detections with the source frame sequence.

### `timestamp`
Floating-point UNIX timestamp in seconds for the frame.

Used to align detections with video time.

### `width`, `height`
Integer dimensions of the frame used to generate the packet.

These values are used to interpret normalized image-space quantities such as `cx`, `cy`, `w`, `h`, `foot_x`, and `foot_y`.

### `detections`
Array of detection objects that passed the UDP emission filters.

The array may be empty when no qualifying detections are present.

## Detection Object Fields

Each element in `detections` represents one emitted tracked object or detection.

### `id`
Integer identifier for the detection in the UDP stream.

When tracking is enabled, this is the persistent track ID. When tracking is unavailable, the formatter falls back to a per-packet index.

### `cls`
Integer Unity-facing class ID.

This is derived from the internal class name through the configured class map.

### `conf`
Floating-point confidence score in the range `[0, 1]`.

This is the confidence value selected for UDP output after any class filtering or continuity handling.

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

Definitions:

- `foot_x = foot_x_px / width`
- `foot_y = foot_y_px / height`

where:

- `foot_x_px = (x1 + x2) / 2`
- `foot_y_px = y2`

If `foot_x` and `foot_y` are unavailable on an input detection, the UDP formatter derives them from the bounding box.

## Notes for Consumers

- Bounding-box geometry fields are normalized and must be interpreted using the packet `width` and `height`.
- The packet contains image-space detection data only. Pose fields and world-coordinate fields are intentionally omitted from this branch.
- Object identity across frames depends on tracker stability, so `id` should be treated as a persistent track identifier rather than a per-frame index.
