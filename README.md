# XRDrone

**Sponsor:** Prof. Raffaele De Amicis  
**Course:** CS 461–463 (Capstone 2025–2026)  
**Institution:** Oregon State University  
**Organization:** XRDrone Team

---

## 🛰️ Project Overview

XRDrone is a real-time drone-to-VR system for fire-rescue scenarios that streams live video from a DJI Neo into a Meta Quest headset, performs on-device detection of **fire, smoke, and humans**, and renders results as **stable, legible spatial overlays** inside a VR scene. These overlays use **depth-like and altitude-like cues** rather than true geo-referenced or world-relative positioning, consistent with the telemetry available from the DJI Neo.

The system provides a cockpit-style HUD showing **FPS, model latency, detection counts (people/fire/smoke), and dropped frames**, updating at least once per second without obstructing the operator’s view.

In addition to live operation, XRDrone supports **post-flight telemetry review** by parsing DJI Assistant 2 flight logs and generating a **3D Cesium visualization** of the drone’s flight path and key telemetry values for after-action analysis.

### Core Objectives

- **Video → Quest (Performance & Latency):**  
  Stream **≥720p** at **≥24 FPS** with **≤2% dropped frames** over a 3-minute test, achieve **≤300 ms median glass-to-glass latency**, and render on a **curved surface** to minimize distortion inside VR.
- **On-Device Vision (Classes & Quality):**  
  Run **on-headset inference** at **≥15 FPS** for **fire, smoke, and humans** and achieve **F1 ≥ 0.5** on **≥150** evaluation frames relevant to forest-fire scenarios.
- **Spatial Overlays & HUD (UX):**  
  Render **stable, legible spatial overlays** using **depth-like and altitude-like cues** (no world-relative or geo-referenced positioning).  
  HUD displays **FPS, model latency, detection counts (people/fire/smoke), and dropped frames**, all updating at least once per second and remaining unobtrusive.
- **Reliability & Safety:**  
  **Recover from forced disconnects** without requiring an app restart and **operate ≥5 minutes** continuously with no crashes and no severe FPS degradation.
- **Privacy & Ethics:**  
  **Store no PII by default**; any recording requires **explicit opt-in** with a confirmation dialog and consent timestamping, respecting dataset licensing and privacy constraints.
- **Post-Flight Telemetry Review:**  
  Parse **DJI Assistant 2 log files** and generate a **3D Cesium-based post-flight visualization** of the drone’s trajectory and key telemetry for after-action review.
- **Engineering Deliverables:**  
  Provide clear **architecture documentation**, **measurement methods** (FPS/latency/inference), runtime **profiling & logs**, and a **one-command reproducible build**.


---

## 📊 Acceptance Criteria — Consolidated Requirements

| ID | Requirement | Priority | Acceptance Evidence |
|----|------------|----------|---------------------|
| **REQ-001** | Stream live DJI Neo video to the headset at **≥ 720p** | Must | Verified via headset capture resolution metadata |
| **REQ-002 (NFR-Performance)** | Maintain **≥ 24 FPS** with **≤ 2% dropped frames** over **3 min** | Must | Log average FPS & drop rate via profiler |
| **REQ-003 (NFR-Performance)** | Achieve **≤ 300 ms median** glass-to-glass latency | Must | Measured via timestamp overlay + external camera |
| **REQ-004 (NFR-Performance)** | Run fire/smoke/human detection **on-device** at **≥ 15 FPS** | Must | On-device profiling shows sustained ≥ 15 inference FPS |
| **REQ-005 (NFR-Quality)** | Achieve **F1 ≥ 0.5** on **≥ 150** evaluation frames | Must | Standard precision/recall evaluation report |
| **REQ-006 (NFR-Privacy/Ethics)** | Store **no PII by default**; recording must be **explicitly user-enabled** | Must | Recording off by default; enabling requires UI toggle + confirmation dialog + consent timestamp; no PII written unless consented |
| **REQ-007** | Render detected objects as **stable spatial overlays** using **depth/altitude cues**, without world-relative or geo-referenced positioning | Must | Live VR demo shows overlays are stable, legible, depth-cued; ≥3 object types; evaluated qualitatively (no ground-truth required) |
| **REQ-008** | HUD displays **FPS, model latency, detection counts (people/fire/smoke), and dropped frames** | Should | Each value updates ≥1/s, remains visible, and does not obstruct the video feed |
| **REQ-009** | Visually differentiate **low-confidence detections** | Should | Detections below threshold use alternate styling (e.g., dashed/faded); at least two confidence levels shown |
| **REQ-010 (NFR-Reliability/Stability)** | Operate **≥ 5 minutes** continuously with no crashes or major FPS degradation | Should | System runs ≥5 minutes with FPS never <20; logs show stable CPU/GPU and no fatal errors |
| **REQ-011** | Support **post-flight telemetry review** using DJI Assistant 2 log files | Should | Parse `.txt` logs, upload telemetry to Cesium, and generate a 3D post-flight visualization of trajectory and key metrics |


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

### 🛰️ Drone + Streaming
- DJI Neo + RC 2  
- FFmpeg (USB capture → WHIP)  
- MediaMTX (WebRTC server)  
- Unity WebRTC Plugin  

### 🤖 AI Detection
- Ultralytics YOLO11 (detection + segmentation)
- Custom YOLO datasets (people, fire, smoke)
- WiSARD Dataset — Visual + Thermal SAR imagery (see citation below)
- AeroScapes Dataset — Outdoor human segmentation dataset (see citations below)

### 🗺️ Telemetry + Mapping
- dji-log-parser — Rust-based DJI flight log decoder (Luc Vauvillier)
- Python telemetry converter  
- Cesium for Unity  

### 🎮 Unity + VR
- Unity  
- Meta Quest 2   

### 🛠️ DevOps + Training
- GitHub + GitHub Actions  
- TensorBoard  
- RTX GPU laptops (training)  
- Custom YOLO dataset YAML configs  

---

## 📚 Dataset Citations

### **WiSARD Dataset**
@inproceedings{BroylesHaynerEtAl2022,
  author = {Broyles, D.* and Hayner, C.* and Leung, K.},
  booktitle = {{IEEE/RSJ Int.\ Conf.\ on Intelligent Robots \& Systems}},
  title = {{WiSARD}: A Labeled Visual and Thermal Image Dataset for Wilderness Search and Rescue},
  year = {2022},
}

---

### **AeroScapes Dataset**
**Primary Research Paper:**  
Ensemble Knowledge Transfer for Semantic Segmentation
Ishan Nigam, Chen Huang, Deva Ramanan
Proceedings of the 2018 IEEE Winter Conference on Applications of Computer Vision

**Dataset Ninja Tools Reference:**  
@misc{ visualization-tools-for-aeroscapes-dataset,
  title = { Visualization Tools for AeroScapes Dataset },
  type = { Computer Vision Tools },
  author = { Dataset Ninja },
  howpublished = { \url{ https://datasetninja.com/aeroscapes } },
  url = { https://datasetninja.com/aeroscapes },
  journal = { Dataset Ninja },
  publisher = { Dataset Ninja },
  year = { 2025 },
  month = { nov },
  note = { visited on 2025-11-28 },
}

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

