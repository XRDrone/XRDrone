# XRDrone

**Sponsor:** Prof. Raffaele De Amicis  
**Course:** CS 461–463 (Capstone 2025–2026)  
**Institution:** Oregon State University  
**Organization:** XRDrone Team

---

## 🛰️ Project Overview

Emergency response and disaster assessment often require situational awareness in environments that are dangerous, time-sensitive, or inaccessible to first responders. Small aerial drones can rapidly collect visual information, but operators are typically limited to 2D video feeds on handheld displays, which can increase cognitive load and reduce spatial understanding in high-stress scenarios.

**XRDrone** addresses this problem by integrating a lightweight aerial drone with a standalone XR headset to provide an immersive, real-time visualization of drone sensor data. The system streams live video from the drone to the headset, performs on-device computer vision inference to detect hazards such as fire, smoke, and people, and renders these detections as stable spatial overlays within a virtual reality environment. This allows users to perceive detections in context, improving situational awareness without relying on cloud services or external infrastructure.

The project scope is intentionally constrained to support **on-device, privacy-preserving operation** suitable for emergency contexts. XRDrone performs inference locally, avoids storing personally identifiable information by default, and prioritizes low latency, stable frame rates, and sustained operation on standalone XR hardware. Advanced capabilities such as natural-language voice commands are included to enable hands-free interaction, but are strictly bounded to a whitelisted set of safe actions to prevent unintended behavior.

XRDrone is not intended to provide autonomous navigation, global mapping, or persistent world anchoring. Instead, it focuses on delivering a credible end-to-end prototype that demonstrates real-time drone-to-XR streaming, on-device hazard detection, and immersive visualization under realistic performance and ethical constraints.

### 🎯 Core Objectives

- **Live Video Streaming:** Stream DJI Neo video to the headset at **≥720p**.
- **Real-Time VR Performance:** Maintain **≥24 FPS** with **≤2% dropped frames** over **3 minutes**, and achieve **≤300 ms median glass-to-glass latency**.
- **On-Device Hazard Detection:** Run **fire/smoke/human** detection on-device at **≥15 FPS** and achieve **F1 ≥ 0.5** on **≥150** evaluation frames.
- **Spatial Visualization (No Geo-Anchoring):** Render detections as **stable spatial VR overlays** with **depth/altitude-like cues**, without **world-relative or geo-referenced** positioning.
- **Operator HUD Telemetry:** Display **FPS, model latency, detection count, and dropped frames** continuously during live operation.
- **Confidence Communication:** **Visually distinguish low-confidence detections** (e.g., alternate styling below a confidence threshold).
- **Stability:** Operate continuously for **≥5 minutes** with **no crashes/freezes** and FPS **never dropping below 20 FPS**.
- **Privacy-by-Default:** **Do not store PII by default**; recording is **only allowed when explicitly enabled** by the user.
- **Speech + LLM HUD Control:** Provide **LLM-backed natural-language voice commands** for HUD control with:
  - **Safe, whitelisted actions/parameters** (reject and log out-of-scope commands)
  - **≤2.5 s median voice-to-action latency** (20 commands) and **graceful failure** with on-screen errors (no interruption to video/overlays)

---

## 📊 Requirements 

| Priority | ID | Type | Requirement | Acceptance Criteria (High Priority Only) | Dependencies | PR/Issue Link | Status |
|---|---|---|---|---|---|---|---|
| P0 | REQ-001 | Functional | Livestream DJI Neo video to the headset at ≥720p. | Verified via headset capture resolution metadata. | This is a foundational requirement because the rest of the system needs a working live video stream before performance, overlays, detection, or voice features can be implemented or validated. | https://github.com/XRDrone/XRDrone/issues/140 | Done |
| P0 | REQ-002 | Non-functional | Maintain ≥24 FPS video with ≤2% dropped frames over 3 min. | Log average FPS & drop rate via profiler. | This depends on REQ-001 because FPS and dropped frames are only meaningful to measure once video is successfully streaming to the headset. | https://github.com/XRDrone/XRDrone/issues/141 | Done |
| P0 | REQ-003 | Non-functional | Achieve ≤300 ms median glass-to-glass latency. | Measured via timestamp overlay + external camera. | This depends on REQ-001 because glass-to-glass latency cannot be measured unless a live end-to-end video pipeline exists. | https://github.com/XRDrone/XRDrone/issues/142 | In progress |
| P0 | REQ-004 | Functional | Run fire/smoke/human detection on-device at ≥15 FPS | Profiling on Quest showing ≥15 inference FPS sustained | This depends on REQ-001 because on-device detection requires a steady stream of video frames to run inference and profile inference speed. | https://github.com/XRDrone/XRDrone/issues/143 | In Progress |
| P1 | REQ-005 | Non-functional | Achieve F1 ≥ 0.5 on ≥150 evaluation frames. | Standard precision/recall evaluation report. | This depends on REQ-004 because you cannot compute F1 score unless the detection model is producing predictions that can be compared against ground truth. | https://github.com/XRDrone/XRDrone/issues/144 | In progress |
| P0 | REQ-006 | Non-Functional | Do not store PII by default; recording only if explicitly enabled by the user. | During runtime testing, no user-identifiable data (name, face, voice ID, location coordinates tied to an identity. | No dependencies | https://github.com/XRDrone/XRDrone/issues/147 | Not Started |
| P0 | REQ-007 | Functional | Render detections as stable spatial VR overlays with depth cues; no world/geo anchoring. | During a live VR demo, detected objects must appear as stable and clearly visible spatial overlays within the VR scene | This depends on REQ-004 because spatial overlays are generated from detection outputs, so there is nothing to render if detections are not being produced. | https://github.com/XRDrone/XRDrone/issues/149 | In progress |
| P1 | REQ-008 | Functional | Show FPS, model latency, detection count, and dropped frames. | HUD elements must continuously display real-time FPS, model latency (ms), the number of objects detected. | This depends on REQ-001 and REQ-004 because HUD metrics require access to the live video pipeline for FPS and drops, and access to the detection pipeline for model latency and detection counts. | https://github.com/XRDrone/XRDrone/issues/150 | In progress |
| P2 | REQ-009 | Functional | Visually distinguish low-confidence detections. | Detections with confidence < threshold (default 0.5 unless user-adjusted) must be visually styled differently—e.g., dashed bounding box or faded color. | This depends on REQ-004 and REQ-007 because confidence values come from the detector and visual differentiation requires an overlay rendering system where styling can be changed. | https://github.com/XRDrone/XRDrone/issues/152 | In progress |
| P0 | REQ-010 | Non-functional | Run continuously for ≥5 minutes without crashes or significant FPS drop. | System must run for ≥5 minutes continuously with FPS never dropping below 20 FPS and no crashes or freezes. fatal errors. | This depends on REQ-001 because continuous stability cannot be demonstrated without continuous streaming, and it is best validated alongside REQ-008 since logs are needed to prove there was no major FPS degradation. | https://github.com/XRDrone/XRDrone/issues/151 | Not started |
| P1 | REQ-011 | Functional | Support LLM-backed natural-language voice commands in Unity with real-time actions. | PASS if, during Play Mode, the user speaks at least 5 distinct commands, the system transcribes them, produces the correct structured action, and applies it in the same session. | This depends on the existence of controllable runtime features like HUD and overlay toggles because voice commands only matter if they can trigger structured actions that change the Unity scene in real time. | https://github.com/XRDrone/XRDrone/issues/148 | In-progress |
| P0 | REQ-012 | Non-functional | Restrict voice/LLM control to a whitelisted action/parameter set; reject others. | PASS if the system rejects (and logs) 10/10 malformed/out-of-scope voice commands (unknown action, invalid parameter types/ranges, unknown target IDs) and the app continues running normally. | This depends on REQ-011 because safety and bounding can only be evaluated once the system is already capable of producing actions from voice and LLM output. | https://github.com/XRDrone/XRDrone/issues/146 | To-do |
| P2 | REQ-013 | Non-functional | ≤2.5 s median voice-to-action latency (20 commands); fail gracefully with on-screen errors. | PASS if median end-to-end latency is <=2.5 s across 20 voice commands, and any STT/LLM outage results in a user-visible error with no crash and no interruption to video streaming/overlays. | This depends on REQ-011 because end-to-end voice command latency can only be measured after the voice pipeline is implemented and able to apply actions, and it also relies on uninterrupted streaming behavior which assumes REQ-001 is stable. | https://github.com/XRDrone/XRDrone/issues/145 | To-do |

### 🏷️ Priority Labels

- **P0 (Must-have / Critical path)**: Required for a credible end-to-end demo; blocking if missing.
- **P1 (Should-have)**: Strongly expected; improves usability, safety, or quality, but the demo can still run.
- **P2 (Nice-to-have)**: Valuable polish or robustness; not essential for the baseline demo.

---

## 👥 Team Members

| Name | Bio |
|------|------|
| **William Brennan** | Interested in VR and has prior experience coding in Unreal Engine. |
| **Troy Diaz** | Worked on a machine learning project with large image datasets and aims to apply that experience to XRDrone’s real-time detection. |
| **Balakrishna Thirumavalavan** | Drawn to the project’s focus on VR display overlays and intrigued by its real-time detection functionality. |
| **Guillermo Morales** | Interested in AR/VR and machine learning for human–computer interaction. |

---

## 💬 Communication & Cadence
- **Primary Channel:** Microsoft Teams (OSU Capstone workspace)  
- **Team Meetings:** Fridays 10–11 AM in Person
- **TA Meetings:** Fridays 11 AM on Zoom  
- **Sponsor Meetings:** Fridays 12 PM in-person
- **Response-Time Expectation:** Pull requests and messages are reviewed within 24 hours
- **Stand-ups:** Mondays and Thursdays – 10-minute check-ins on Teams  

---

## 🔀 Branching & Reviews
- **Default branch:** `main`
- **Branch flow:** `feature/* → pull request → ≥1 review → merge`
- **Review cadence:** All PRs reviewed within 24 hours via Teams notification.
- **PR template:** `.github/PULL_REQUEST_TEMPLATE.md` created to guide contributions.

---

## 🧰 Tech Stack

### 🛸 Hardware
- **Drone:** DJI Neo
- **XR Headset:** Meta Quest 2 (standalone VR)

### 🥽 XR Application
- **Engine:** Unity (C#) targeting Quest
- **Features:** VR visualization of detections + HUD, plus voice/LLM-driven HUD controls

### 🧠 On-Device Computer Vision Pipeline
- **Language:** Python 3
- **CV Inference:** Ultralytics YOLO (detection + segmentation)
- **ML Runtime:** PyTorch (CUDA when available)
- **Video & Rendering:** OpenCV
- **Numerics:** NumPy

### 🌐 Streaming & Networking
- **UDP Telemetry:** JSON packets (detections + metadata) published from Python to Unity
- **Video Streaming:** RTSP pipeline via FFmpeg (x264 / low-latency settings)
- **Transport/IO:** Python `socket` (UDP) + `subprocess` (FFmpeg)

### 🔒 Responsible/Safe Operation
- **Privacy-by-default:** No PII stored unless recording is explicitly enabled
- **Voice Action Bounding:** Whitelisted actions/parameters; reject out-of-scope commands

---

## 📚 Citations

### **Ultralytics YOLO11**
```
@inproceedings{BroylesHaynerEtAl2022,
  author = {Broyles, D.* and Hayner, C.* and Leung, K.},
  booktitle = {{IEEE/RSJ Int.\ Conf.\ on Intelligent Robots \& Systems}},
  title = {{WiSARD}: A Labeled Visual and Thermal Image Dataset for Wilderness Search and Rescue},
  year = {2022},
}
```

---

## 🧩 Repository Layout

```
📦 XRDrone
├─ cesium
│  └─ reformat_json.py
├─ models
│  ├─ yolo_people_fire_smoke
│  │  ├─ detection_log_loader.py
│  │  ├─ hud.py
│  │  ├─ main.py
│  │  ├─ merger.py
│  │  ├─ README.txt
│  │  └─ test_with_coverage.py
│  └─ yolo11_models
│     ├─ yolo11n-seg.pt
│     └─ yolo11n.pt
├─ XRDroneApp
│  ├─ Assets
│  ├─ Packages
│  └─ ProjectSettings
├─ .gitignore
├─ CONTRIBUTING.md
├─ LICENSE
└─ README.md
```
©generated by [Project Tree Generator](https://woochanleee.github.io/project-tree-generator)

---

## ⚖️ License
This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgements
Special thanks to **Prof. Raffaele De Amicis** for project sponsorship and guidance,  
and to Oregon State University’s School of EECS for supporting the XRDrone Capstone.

---