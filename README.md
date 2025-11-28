# XRDrone

**Sponsor:** Prof. Raffaele De Amicis  
**Course:** CS 461–463 (Capstone 2025–2026)  
**Institution:** Oregon State University  
**Organization:** XRDrone Team

---

## 🛰️ Project Overview
XRDrone is a real-time drone-to-VR system for fire-rescue scenarios that streams live video from a DJI Neo into a Meta Quest headset, runs on-device detection for **fire, smoke, and humans**, and renders results as **3D spatial overlays** with a cockpit-style HUD. The goal is to improve situational awareness by showing **where** detections are in space, not just **that** they exist. The app targets **single-operator** use on standalone VR hardware and excludes autonomous flight control and long-term cloud storage by default.

### Core Objectives
- **Video → Quest (Performance & Latency):**  
  Stream **≥720p** at **≥24 FPS** with **≤2% dropped frames** (3-min test), **≤300 ms median glass-to-glass latency**, and render on a **curved surface** to minimize distortion.
- **On-Device Vision (Classes & Quality):**  
  Run **edge inference on Quest** at **≥15 FPS** for **fire/smoke/human**; achieve **F1 ≥ 0.5** on **≥150** evaluation frames relevant to forest-fire scenes.
- **Spatial Overlays & HUD (UX):**  
  Show **3D-aligned overlays** that convey **altitude** and **lateral position**; HUD displays **FPS, latency, detection confidence, and drone battery**; visually **de-emphasize low-confidence** detections.
- **Reliability & Safety:**  
  **Recover from forced disconnect** without app restart and **operate ≥5 minutes** without crashes or severe FPS drops.
- **Privacy & Ethics:**  
  **No PII stored by default**; any recording is **explicit opt-in** with a confirmation dialog and consent timestamping.
- **Engineering Deliverables:**  
  Clear **architecture**, **measurement methods** (latency/FPS/inference), **profiling & logs**, and a **reproducible build**.

---

## 📊 Acceptance Criteria — Consolidated Requirements
| ID | Requirement | Priority | Acceptance Evidence |
|---|---|---|---|
| REQ-001 | Stream live DJI Neo video to the headset at **≥ 720p** | Must | Verified via headset capture resolution metadata |
| REQ-002 (NFR-Performance) | Maintain **≥ 24 FPS** with **≤ 2% dropped frames** over **3 min** | Must | Log average FPS & drop rate via profiler |
| REQ-003 (NFR-Performance) | **≤ 300 ms median** glass-to-glass latency (**≤ 400 ms p95**) | Must | Measured via timestamp overlay + external camera; report median & p95 |
| REQ-004 (NFR-Performance) | On-device detection (fire/smoke/human) at **≥ 15 FPS** on Quest | Must | On-device profiling shows sustained ≥ 15 inference FPS |
| REQ-005 (NFR-Quality) | **F1 ≥ 0.5** on **≥ 150** labeled evaluation frames (fire-scene relevant) | Must | Standard precision/recall evaluation report |
| REQ-006 (NFR-Privacy/Ethics) | **No PII stored by default**; any recording is **explicit opt-in** and **license-compliant** | Must | Runtime shows recording **off** by default; enabling requires clear UI toggle + confirmation dialog; consent **timestamped**; no user-identifiable data logged unless consent; note dataset-license compliance (Fire & Smoke, CrowdHuman) |
| REQ-007 | Render **spatial overlays** that convey **altitude** and **lateral position** | Must | Live VR demo shows correct 3D placement with visible altitude/lateral indicators; ≥ 3 object types; verified by reference marker/external ground truth |
| REQ-008 | HUD shows **FPS, latency, detection confidence, drone battery** | Should | Each value updates ≥ 1/s, remains visible without obstructing video |
| REQ-009 | **Low-confidence** detections are visually differentiated | Should | Detections with confidence < threshold (default 0.5 unless user-adjusted) use distinct styling (e.g., dashed/faded); demonstrate ≥ 2 levels |
| REQ-010 (NFR-Reliability/Stability) | **≥ 5 min** continuous operation; **no crashes**; FPS never **< 20** | Should | Performance log shows stable CPU/GPU, zero fatal errors |
| REQ-011 (Robustness & Recovery) | Auto-reconnect after stream drop in **≤ 5 s**; overlays resume without app restart | Should | Induce forced disconnect; verify reconnect ≤ 5 s and overlay continuity |
| REQ-012 (Build & Reproducibility) | One-command build; runtime metrics logging (FPS, latency samples, inference FPS); short architecture & measurement doc | Should | Fresh clone builds with one command; logs emitted at runtime; doc reproduces metrics methodology |

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
- **Sponsor Meetings:** Time TBD with Prof. Raffaele De Amicis
- **Response-Time Expectation:** Pull requests and messages are reviewed within 24 hours
- **Stand-ups:** Mondays and Thursdays – 10-minute check-ins on Teams  

---

## 🔀 Branching & Reviews
- **Default branch:** `main`
- **Branch flow:** `feature/* → pull request → ≥1 review → merge`
- **Review cadence:** All PRs reviewed within 24 hours via Teams notification.
- **PR template:** `.github/PULL_REQUEST_TEMPLATE.md` created to guide contributions.

---

## 🖥️ Tech Stack 

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

## 📚 Citations

### **Ultralytics YOLO11**
```
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
```

### **WiSARD Dataset**
```
@inproceedings{BroylesHaynerEtAl2022,
  author = {Broyles, D.* and Hayner, C.* and Leung, K.},
  booktitle = {{IEEE/RSJ Int.\ Conf.\ on Intelligent Robots \& Systems}},
  title = {{WiSARD}: A Labeled Visual and Thermal Image Dataset for Wilderness Search and Rescue},
  year = {2022},
}
```

---

### **AeroScapes Dataset**
**Primary Research Paper:** 
``` 
Ensemble Knowledge Transfer for Semantic Segmentation
Ishan Nigam, Chen Huang, Deva Ramanan
Proceedings of the 2018 IEEE Winter Conference on Applications of Computer Vision
```

**Dataset Ninja Tools Reference:**  
```
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

