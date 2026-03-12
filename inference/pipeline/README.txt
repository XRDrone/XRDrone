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

foot_x, foot_y
  Normalized bottom-center point of the bounding box (the “foot point”), in image coordinates.
  This is typically used as the point on the ground where the object is standing.
    foot_x = foot_x_px / width
    foot_y = foot_y_px / height
  where:
    foot_x_px = (x1 + x2) / 2
    foot_y_px = y2
  If foot_x/foot_y are not available, consumers can fall back to bottom-center derived
  from (cx, cy, w, h).

world_valid
  True if the pipeline successfully projected the detection’s foot point into world space
  for this frame. False means the world projection is unavailable or unreliable.

world_x, world_y, world_z
  World-space coordinates for the detection’s foot point in the ArUco/world reference frame.
  These are computed by back-projecting the foot pixel into a world-space ray using the
  camera pose/intrinsics, then intersecting that ray with the ground plane.
  When world_valid is false, these are set to 0.0.

Cross-field relationship
  world_valid depends on both:
    1) pose.pose_valid == true for the frame, and
    2) a successful ray-plane intersection for the detection’s foot point.


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

UDP Validation Tests
====================

The file `test_with_coverage.py` provides simple validation checks for the UDP packet format
described above. Its purpose is to confirm that the pipeline produces packets that match this
README structure and that UDP transport works as expected.

File overview
-------------
`test_with_coverage.py` is a small validation script for the UDP packet contract.
It is not part of the live pipeline itself. Instead, it is used to verify that the
formatted packet and UDP send/receive behavior are correct.

Command: python test_with_coverage.py
-------------------------------------
This runs the non-live validation checks.

What it does:
  1) Builds a sample UDP packet using the same formatter used by the pipeline.
  2) Checks that the packet structure matches this README exactly.
  3) Sends that packet over UDP on localhost.
  4) Receives the same packet back.
  5) Validates the received packet again.

This confirms:
  - the UDP packet structure is correct
  - the formatter output matches this README
  - UDP send/receive works on localhost

Expected success output:
  PASSED: UDP formatter structure and UDP send/receive are valid

If this test fails, the issue is usually one of:
  - missing required fields
  - extra unexpected fields
  - wrong field types
  - invalid UDP loopback send/receive behavior


Command: python test_with_coverage.py --live
--------------------------------------------
This runs the live UDP validation mode.

What it does:
  1) Runs the same formatter/schema checks as the default mode.
  2) Runs the same localhost UDP loopback test as the default mode.
  3) Opens a UDP listener on the configured port.
  4) Waits for real packets from a running pipeline process.
  5) Validates each live packet against this README structure.

This confirms:
  - the live pipeline is actively sending packets
  - the packets received over UDP match this README
  - runtime packet structure remains valid outside of the synthetic sample test

Expected success output:
  PASSED: live UDP transport and README packet structure are valid

Important:
  `--live` does not generate packets by itself. A separate pipeline process must already
  be running and sending UDP packets to the configured host/port.

Typical usage:
  Terminal 1:
    python main.py

  Terminal 2:
    python test_with_coverage.py --live

If the live test fails with a timeout, that usually means:
  - `main.py` is not running
  - UDP output is disabled
  - the test is listening on the wrong port
  - another program is already using the port


How the validation works
------------------------
The validation script checks that every UDP packet contains exactly these top-level fields:
  - frame_id
  - timestamp
  - width
  - height
  - detections
  - pose

For each detection object, it checks:
  - id
  - cls
  - conf
  - cx
  - cy
  - w
  - h
  - foot_x
  - foot_y
  - world_valid
  - world_x
  - world_y
  - world_z

For the pose object, it checks:
  - x
  - altitude
  - z
  - yaw
  - pitch
  - roll
  - hfov
  - markers_used
  - pose_valid

The test fails if:
  - a required field is missing
  - an extra field is present
  - a field has the wrong type
  - normalized fields are outside the expected range


Summary
-------
Use `python test_with_coverage.py` to verify the UDP formatter and localhost UDP transport.
Use `python test_with_coverage.py --live` to verify packets from the real running pipeline.
Both tests are intended to confirm that the UDP packet structure matches this README.