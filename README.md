# XRDrone

**Sponsor:** Prof. Raffaele De Amicis  
**Course:** CS 461–463 (Capstone 2025–2026)  
**Institution:** Oregon State University  
**Organization:** XRDrone Team

---

## Project Overview

**XRDrone** is a real-time perception-to-XR pipeline that combines computer vision, structured networking, and Unity scene updates. The current backend focuses on detecting and tracking people from video, receiving ORB-SLAM pose data through a dedicated middle-man receiver, and packaging the fused result into structured UDP JSON for Unity.

Rather than treating each frame independently, the pipeline is designed for continuity across frames. The detection side produces stable tracked objects, the ORB-SLAM side provides camera pose input, and the middle-man aligns both branches before sending a unified packet downstream. This allows Unity to consume a more coherent scene update instead of isolated per-frame detections.

The project is currently centered on three goals:
- reliable human detection and tracking from live video
- ORB-SLAM pose ingestion and fusion through a UDP middle-man
- consistent Unity-side consumption of the fused JSON output

---

## System Architecture

### 1. Video Source and Streaming
- A shared video source is distributed through MediaMTX.
- The detection branch and the ORB-SLAM branch are expected to read from the same stream.
- This keeps both branches aligned to the same source video during live runs.

### 2. Human Detection Branch
- The Python perception pipeline reads the stream and runs the current detection and tracking logic.
- It produces image-space detections, track IDs, and related metadata.
- The detection branch does not independently define the final world pose for Unity.

### 3. ORB-SLAM Branch
- ORB-SLAM runs separately on the same video stream.
- It sends pose packets over UDP to the backend middle-man.
- These packets include frame or timing information plus pose values used for fusion.

### 4. Middle-Man Fusion Layer
- The backend listens for incoming ORB-SLAM UDP pose packets.
- It aligns ORB-SLAM data with the detection branch by `frame_id` first and timestamp second.
- When a usable pose is available, it projects detection foot points into world space and builds a fused packet.
- It also exposes runtime status information for missing, stale, or invalid ORB-SLAM input.

### 5. Unity XR Scene
- Unity receives the fused UDP JSON output from the backend.
- It parses detection, tracking, and pose-related fields to update scene content in real time.
- The long-term goal is stable spatial scene updates driven by the fused perception output.

---

## Current Capabilities

- Python-based human detection and tracking pipeline
- Structured UDP JSON output for Unity integration
- ORB-SLAM middle-man UDP receiver in the backend
- Detection-to-pose fusion path for combining tracked humans with ORB-SLAM camera pose
- Runtime overlay support for fusion status and failure handling
- Documentation and test updates for the ORB-SLAM middle-man workflow

---

## Current Workflow

A typical live setup is:

```text
Video source -> MediaMTX
                  ├──> XRDrone detection pipeline
                  └──> ORB-SLAM
                           └──> UDP pose sender -> XRDrone middle-man
                                                     └──> fused UDP JSON -> Unity
```

At a high level:
- MediaMTX distributes the video stream
- the XRDrone backend performs detection and listens for ORB-SLAM pose packets
- the middle-man fuses both inputs
- Unity receives the final fused output

---

## Tech Stack

### Hardware
- **Drone:** DJI Neo
- **XR Headset:** Meta Quest 2

### Perception and Backend
- **Languages:** Python, Rust
- **Computer Vision:** OpenCV
- **Detection / Tracking:** Ultralytics YOLO-based pipeline
- **Numerics / Data Processing:** NumPy
- **Runtime:** PyTorch
- **Native acceleration:** PyO3-based Rust extension (`xrdrone_native`)

### Streaming and Communication
- **Video distribution:** MediaMTX
- **Transport:** UDP
- **Data format:** Structured JSON packets
- **Supporting tools:** FFmpeg, socket-based networking, subprocess-driven components

### XR Application
- **Engine:** Unity
- **Language:** C#
- **Role:** Receive fused perception output and update the XR scene

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
