# XRDrone

**Sponsor:** Prof. Raffaele De Amicis  
**Course:** CS 461–463 (Capstone 2025–2026)  
**Institution:** Oregon State University  
**Organization:** XRDrone Team

---

## Project Overview

**XRDrone** is a real-time drone-to-XR perception pipeline that connects video capture, ArUco-based camera pose estimation, human detection, UDP communication, and Unity scene updates.

The backend can process prerecorded video or live capture-card video. It estimates camera pose from visible ArUco fiducial markers, detects and tracks people in the video stream, prepares structured scene-state packets, and sends those packets to Unity. Unity then updates the XR scene by representing the estimated camera position as a drone model and detected humans as scene objects.

The project is centered on three goals:

- reliable ArUco-based camera pose estimation from drone video
- stable human detection and tracking for Unity scene updates
- compact UDP JSON output that Unity can consume in real time

---

## System Architecture

### 1. Video Source and Streaming

- The pipeline supports both prerecorded video files and live video capture.
- For live drone input, the DJI Neo sends video to the DJI RC 2 controller.
- The controller output is routed through a USB-C-to-HDMI adapter, HDMI capture card, and USB connection into the backend computer.
- The backend treats the incoming video as a local video source for frame processing.
- MediaMTX can be used when the video stream needs to be routed or viewed separately from the structured Unity packet stream.

### 2. Human Detection Branch

- The Python perception pipeline reads frames from the selected video source.
- YOLO-based detection produces image-space human detections, confidence values, bounding boxes, and tracking metadata.
- The detection branch is evaluated by runtime behavior, tracking continuity, and whether detections remain stable enough for Unity visualization.

### 3. ArUco Pose Estimation Branch

- OpenCV detects visible ArUco markers in the video frame.
- Known marker IDs and marker coordinates define the spatial reference for pose estimation.
- The estimated camera pose is used to place the virtual drone/camera representation in Unity.
- Pose quality depends on marker visibility, marker size, viewing angle, calibration, and image clarity.

### 4. Middle-Man Fusion Layer

- The backend combines ArUco pose estimates, human detections, tracking metadata, timestamps, and frame information.
- Python coordinates frame capture, pose estimation, detection, logging, and UDP communication.
- Rust helper code supports selected packet formatting, projection, smoothing, adaptive tuning, and ID-continuity operations.
- The output is a structured UDP JSON scene-state packet for Unity.
- When logging is enabled, the runtime also saves pose, detection, packet, and frame-level logs for later review.

### 5. Unity XR Scene

- Unity receives UDP JSON packets from the backend.
- Unity parses pose, detection, tracking, and frame metadata to update the XR scene in real time.
- The estimated camera position is rendered as a drone model.
- Detected humans are represented as persistent scene objects.
- The Meta Quest 2 is used as a PC-linked XR display target.

---

## Current Capabilities

- Prerecorded video testing with optional runtime logs
- Live video capture support with optional runtime logs
- ArUco marker detection and camera pose estimation
- YOLO-based human detection and tracking
- Structured UDP JSON output for Unity integration
- Frame-level logging for pose, detections, markers, packets, and runtime summaries
- Python runtime orchestration with Rust helper support
- Unity-facing scene-state updates for XR visualization

---

## Current Workflow

A typical ArUco-based run follows this flow:

```text
DJI Neo / prerecorded video
        -> backend video capture
        -> ArUco pose estimation
        -> human detection and tracking
        -> Python/Rust scene-state preparation
        -> UDP JSON packets
        -> Unity XR scene
        -> Meta Quest 2 display
```

Run on the default video path:

```bash
python main.py
```

Run on a specific prerecorded video:

```bash
python main.py --video "path/to/video.mp4"
```

Run with logs enabled:

```bash
python main.py --logs
```

Run headless with logs:

```bash
python main.py --logs --no-gui
```

At a high level:

- the backend reads video frames
- ArUco markers provide the camera pose reference
- the detection branch identifies and tracks people
- the backend packages frame, pose, and detection data into UDP JSON
- Unity receives the packets and updates the XR scene

---

## Tech Stack

### Hardware

- **Drone:** DJI Neo
- **Controller:** DJI RC 2
- **Capture path:** USB-C-to-HDMI adapter, HDMI capture card, USB-A-to-USB-A USB 3.0 cable
- **XR Headset:** Meta Quest 2

### Perception and Backend

- **Languages:** Python, Rust
- **Computer Vision:** OpenCV ArUco
- **Detection / Tracking:** Ultralytics YOLO-based pipeline
- **Numerics / Data Processing:** NumPy
- **Runtime:** PyTorch
- **Native acceleration:** PyO3-based Rust extension (`xrdrone_native`)

### Streaming and Communication

- **Video routing:** MediaMTX when needed
- **Transport:** UDP
- **Data format:** Structured JSON packets
- **Supporting tools:** FFmpeg, socket-based networking, runtime logging tools

### XR Application

- **Engine:** Unity
- **Language:** C#
- **Role:** Receive structured perception output and update the XR scene

---

## Repository Documentation

Detailed documentation lives in `docs/`.

Main references:

- `docs/setup.md` – environment setup and runtime setup
- `docs/settings.md` – runtime configuration settings
- `docs/udp-json.md` – Unity-facing UDP packet schema
- `docs/runtime-ui-and-terminal-reference.md` – runtime overlay and terminal reference
- `docs/testing.md` – validation and testing workflow

---

## Team Members

| Name | Focus Areas |
|------|------|
| **William Brennan** | XR/VR interaction, Unity systems, and visualization |
| **Troy Diaz** | GIS, machine learning, spatial perception, backend fusion, and Python-to-Unity pipeline development |
| **Balakrishna Thirumavalavan** | XR display systems, HUD integration, and runtime interaction design |
| **Guillermo Morales** | AR/VR, machine learning, perception, and spatial registration workflows |

---

## Communication and Cadence

- **Primary Channel:** Microsoft Teams (OSU Capstone workspace)
- **Team Meetings:** Fridays 10–11 AM in person
- **TA Meetings:** Fridays 11 AM on Zoom
- **Sponsor Meetings:** Fridays 12 PM in person
- **Response-Time Expectation:** Pull requests and messages reviewed within 24 hours
- **Stand-ups:** Mondays and Thursdays, 10-minute check-ins on Teams

---

## Branching and Reviews

- **Default branch:** `main`
- **Branch flow:** `feature/* -> pull request -> review -> merge`
- **PR template:** `.github/PULL_REQUEST_TEMPLATE.md`
- **Expectation:** implementation changes should update documentation when behavior, schema, or setup changes

---

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

---

## Acknowledgements

Special thanks to **Prof. Raffaele De Amicis** for project sponsorship and guidance, and to Oregon State University’s School of EECS for supporting the XRDrone Capstone.
