<div align="center">

# XRDrone

### Real-Time Drone-to-XR Perception Fusion and Immersive Visualization

![Python](https://img.shields.io/badge/Python-Backend-blue)
![Rust](https://img.shields.io/badge/Rust-Native%20Helpers-orange)
![Unity](https://img.shields.io/badge/Unity%206-XR%20Visualization-black)
![ORB--SLAM3](https://img.shields.io/badge/ORB--SLAM3-Markerless%20Pose-purple)
![ArUco](https://img.shields.io/badge/ArUco-Fiducial%20Pose-green)
![UDP JSON](https://img.shields.io/badge/UDP-JSON%20Scene%20State-lightgrey)

**XRDrone** is an edge-assisted drone-to-XR pipeline that connects live video capture, camera-pose estimation, person-aware perception, UDP scene-state transport, and Unity-based immersive visualization.

</div>

---

## Project

XRDrone compares two camera-pose architectures inside one shared drone-to-XR workflow. Both paths convert drone or prerecorded video into structured scene-state updates that Unity can use to render camera pose, tracked people, and optional world-space positions in an XR scene.

```text
Drone / prerecorded video
        ↓
Video capture and routing
        ↓
Camera-pose estimation + human detection
        ↓
Scene-state fusion
        ↓
UDP JSON packets
        ↓
Unity XR visualization
        ↓
Meta Quest 2 display
```

---

## Architectures

| Architecture | Purpose |
|---|---|
| **ArUco-based pipeline** | Fiducial-marker pose estimation for controlled marker-referenced scenes. Uses OpenCV ArUco, YOLO-based detection/tracking, Python/Rust helper logic, runtime logging, and UDP JSON output to Unity. |
| **ORB-SLAM3-based pipeline** | Markerless SLAM pose estimation using a split Linux/Windows workflow. Linux handles capture, virtual camera creation, FFmpeg stream splitting, and ORB-SLAM3 pose output. Windows receives RTSP through MediaMTX, runs YOLOv5nu fusion, and forwards UDP JSON to Unity. |

---

## Setup Entry Points

### ArUco path

```bash
cd aruco/pipeline
```

Start with the documentation in [`aruco/pipeline/docs/`](./aruco/pipeline/docs/) to get started.

### ORB-SLAM3 path

```bash
cd orbslam/platforms
```

Start with the documentation in [`orbslam/platforms/docs/`](./orbslam/platforms/docs/) to get started.

---

## Tech Stack

### Hardware and Capture

| Component | Details |
|---|---|
| Drone | DJI Neo |
| Controller | DJI RC 2 |
| Capture chain | DJI RC 2 → USB-C-to-HDMI adapter → HDMI capture card → USB-A-to-USB-A USB 3.0 → backend machine |
| XR headset | Meta Quest 2 as a PC-linked XR display |
| ORB-SLAM3 Linux side | Ubuntu Linux workstation for capture-card acquisition, virtual-camera creation, FFmpeg splitting, and ORB-SLAM3 execution |
| Windows backend | Windows machine for MediaMTX, YOLOv5nu detection/fusion, UDP JSON output, and Unity 6 visualization |

### Perception, Pose, and Fusion

| Layer | Tools / Models |
|---|---|
| ArUco pose path | Python, OpenCV ArUco, camera calibration, known marker IDs/coordinates |
| ORB-SLAM3 pose path | ORB-SLAM3 Docker submodule, v4l2loopback virtual camera, EuRoC validation workflow |
| Human detection | Ultralytics YOLO models, YOLO26 models for the ArUco path, `yolov5nu.pt` for the ORB-SLAM3 Windows fusion path |
| Tracking / continuity | YOLO tracking metadata, ID-continuity handling, smoothing, and runtime state management |
| Backend fusion | Python orchestration with NumPy, OpenCV, PyTorch, and JSON packet generation |
| Native helpers | Rust helper modules built through PyO3 / maturin for selected projection, smoothing, UDP, and ID-continuity operations |

### Streaming, Networking, and XR

| Layer | Tools / Protocols |
|---|---|
| Video routing | FFmpeg, RTSP, MediaMTX |
| Linux virtual camera | v4l2loopback and v4l-utils |
| Pose transport | UDP pose packets from ORB-SLAM3 to Windows |
| Unity transport | UDP JSON scene-state packets from backend to Unity |
| XR visualization | Unity 6, C#, XR scene receiver scripts, Meta/Oculus PC-linked headset workflow |

---

## Team

**Sponsor:** Prof. Raffaele De Amicis  
**Course:** CS 461–463 Capstone 2025–2026  
**Institution:** Oregon State University  
**Organization:** XRDrone Team

| Name | Focus Areas |
|---|---|
| **William Brennan** | XR/VR interaction, Unity systems, and visualization |
| **Troy Diaz** | GIS, machine learning, spatial perception, backend fusion, and Python-to-Unity pipeline development |
| **Balakrishna Thirumavalavan** | XR display systems, HUD integration, and runtime interaction design |
| **Guillermo Morales** | AR/VR, machine learning, perception, and spatial registration workflows |

---

## License

This project is licensed under the MIT License. See [`LICENSE`](./LICENSE) for details.

---

## Acknowledgments

XRDrone was developed as an Oregon State University EECS capstone project. The authors thank Dr. Raffaele De Amicis, Kirsten Winters, Alexander Ulbrich, and Oregon State University's School of Electrical Engineering and Computer Science for their sponsorship, instruction, guidance, and support.
