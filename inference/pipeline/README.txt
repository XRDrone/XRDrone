UDP JSON Packet Sample
=================================
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
      "h": 0.7393832171404803
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

UDP JSON Packet Field Explanations
=================================

Top-level fields
----------------
frame_id
  Sequential frame counter from the pipeline loop. Increments by 1 per processed frame.

timestamp
  Wall-clock time (Unix epoch seconds, fractional) when the frame was processed.
  Used to align detections/pose with video time.

width, height
  Frame resolution (pixels) used to normalize bounding boxes and for the consumer (Unity)
  to interpret cx/cy/w/h correctly.

detections
  Array of detected objects for this frame. Each entry is one object with normalized
  bounding box + metadata.

pose
  Camera/drone pose estimate for this frame, derived from markers. Always included;
  may be invalid (pose_valid=false).


Detection object fields (detections[i])
---------------------------------------
id
  Persistent object ID (track ID). Same object should keep the same id across frames
  when tracking is working.

cls
  Integer class ID for the object (Unity mapping). Example: 0 usually means "person"
  if your mapping is {"person":0, "fire":1, ...}.

conf
  Detection confidence score in [0,1] from the model.

cx, cy
  Normalized center of the bounding box in image coordinates:
    cx = center_x / width
    cy = center_y / height
  0.0 is left/top edge, 1.0 is right/bottom edge.

w, h
  Normalized bounding box size:
    w = box_width / width
    h = box_height / height
  Values are in [0,1].

Example interpretation:
  cx=0.46, cy=0.63 means the object center is ~46% across the image and ~63% down.
  w=0.60, h=0.74 means the box covers ~60% of the image width and ~74% of the image height.


Pose object fields (pose)
-------------------------
x, altitude, z
  Camera position in a world coordinate system (units depend on your marker scale;
  typically meters).
    x        = world X position
    altitude = world Y position (height)
    z        = world Z position

yaw, pitch, roll
  Camera orientation angles (degrees). Convention depends on implementation, but typically:
    yaw   = rotation around vertical axis
    pitch = tilt up/down
    roll  = tilt left/right

hfov
  Horizontal field-of-view (degrees) used to approximate camera intrinsics for pose solving.

markers_used
  Number of detected markers that were actually used to compute pose this frame.

pose_valid
  Whether pose estimation succeeded this frame.
    false means the system did not compute a reliable pose (often because no markers were
    detected, or solvePnP failed).
