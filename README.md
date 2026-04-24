# XRDrone

**Sponsor:** Prof. Raffaele De Amicis  
**Course:** CS 461–463 (Capstone 2025–2026)  
**Institution:** Oregon State University  
**Organization:** XRDrone Team

---

## 🛰️ Project Overview

**XRDrone** is a Python-to-Unity XR perception pipeline that processes video, runs the current detection and tracking logic, and packages the results into structured UDP JSON for Unity. The Python side is responsible for producing stable object information and the spatial data needed downstream, including the pose and registration components that keep detections consistent within the scene. Rather than treating each frame independently, the method emphasizes continuity across frames so Unity can work with persistent, meaningful scene updates instead of noisy per-frame outputs.

On the Unity side, the system consumes the incoming JSON stream and uses it to drive the XR scene in real time. Detected and tracked objects are updated in a shared spatial context, allowing the scene to reflect where things are and how they move with consistent alignment. In practice, the method is the integration of three parts working together: computer-vision inference in Python, structured UDP communication between systems, and Unity-based spatial rendering that uses tracking and pose information to represent the environment coherently in XR.

XRDrone is focused on building a coherent end-to-end prototype for real-time XR scene updates from perception output. The current emphasis is reliable Python-side inference, stable identity and spatial consistency across frames, and structured Unity-side rendering of those updates in a shared scene.

---

## 🧭 System Architecture

### 1. Python Perception Pipeline
- Ingests video input and runs the current detection and tracking pipeline.
- Produces stable object-level outputs instead of isolated frame-by-frame detections.
- Computes downstream spatial data, including pose and registration information used for Unity-side alignment.
- Packages the result into structured UDP JSON for transport.

### 2. UDP JSON Interface
- Sends structured scene updates from Python to Unity.
- Carries detection, tracking, and spatial fields needed for downstream visualization.
- Preserves continuity across frames so Unity can apply updates to persistent scene objects.

### 3. Unity XR Scene
- Receives and parses the incoming JSON stream.
- Updates detected and tracked objects in real time within a shared spatial context.
- Uses tracking and pose-aware updates to keep XR scene behavior more stable and meaningful than raw per-frame overlays.

---

## ✅ Current Capabilities

- Python-based video perception pipeline for detection and tracking.
- Structured UDP JSON output for Unity integration.
- Persistent object updates designed for continuity across frames.
- Pose-aware and registration-aware scene updates for downstream spatial consistency.
- Unity-side real-time consumption of perception data to drive XR scene behavior.
- Ongoing prototype work related to HUD improvements, scene interaction, and broader spatial/XR experiments.

---

## 🧰 Tech Stack

### 🛸 Hardware
- **Drone:** DJI Neo
- **XR Headset:** Meta Quest 2

### 🧠 Perception Pipeline
- **Language:** Python 3
- **Computer Vision:** OpenCV
- **Detection / Tracking:** Ultralytics YOLO-based pipeline
- **Numerics / Data Processing:** NumPy
- **Runtime:** PyTorch

### 🌐 Communication
- **Transport:** UDP
- **Data Format:** Structured JSON packets from Python to Unity
- **Streaming / Processing Tools:** FFmpeg, socket-based networking, subprocess-driven pipeline components

### 🥽 XR Application
- **Engine:** Unity
- **Language:** C#
- **Role:** Receive structured perception output and render scene updates in XR

---

## 👥 Team Members

| Name | Focus Areas |
|------|------|
| **William Brennan** | XR/VR interaction, Unity systems, and visualization. |
| **Troy Diaz** | GIS, machine learning, spatial perception, and Python-to-Unity pipeline development. |
| **Balakrishna Thirumavalavan** | XR display systems, HUD integration, and runtime interaction design. |
| **Guillermo Morales** | AR/VR, machine learning, perception, and spatial registration workflows. |

---

## 💬 Communication & Cadence
- **Primary Channel:** Microsoft Teams (OSU Capstone workspace)
- **Team Meetings:** Fridays 10–11 AM in person
- **TA Meetings:** Fridays 11 AM on Zoom
- **Sponsor Meetings:** Fridays 12 PM in person
- **Response-Time Expectation:** Pull requests and messages are reviewed within 24 hours
- **Stand-ups:** Mondays and Thursdays – 10-minute check-ins on Teams

---

## 🔀 Branching & Reviews
- **Default branch:** `main`
- **Branch flow:** `feature/* → pull request → ≥1 review → merge`
- **Review cadence:** All PRs reviewed within 24 hours via Teams notification
- **PR template:** `.github/PULL_REQUEST_TEMPLATE.md`

---

## ⚖️ License
This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgements
Special thanks to **Prof. Raffaele De Amicis** for project sponsorship and guidance, and to Oregon State University’s School of EECS for supporting the XRDrone Capstone.
