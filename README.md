# XRDrone

**Sponsor:** Prof. Raffaele De Amicis  
**Course:** CS 461–463 (Capstone 2025–2026)  
**Institution:** Oregon State University  
**Organization:** XRDrone Team

---

## 🛰️ Project Overview
XRDrone is a real-time drone-to-VR streaming and object recognition system. The project aims to build a **Unity-based application** (target: Android/Meta Quest 2) that receives **live video from a DJI Neo drone**, renders it in **immersive 3D space**, and performs **on-device object detection** to overlay visual cues such as bounding boxes, labels, or shader-based highlights.

### **Core Objectives**
- **Drone Video to Quest 2:**  
  Stream ≥ 720p video at ≥ 24 FPS with ≤ 300 ms glass-to-glass latency, rendered on a hemispherical surface to reduce distortion.
- **On-Device Vision:**  
  Detect at least two object classes (e.g., “tree,” “person,” “vehicle”) in real time (≥ 15 FPS) using optimized ML models.
- **VR User Experience:**  
  Provide a cockpit-style HUD showing stream status, detection confidence, and toggles for overlays or recentering.
- **Engineering Deliverables:**  
  Document architecture, streaming protocol rationale, model choice, benchmarking results, and safety/ethical notes.

---

## 📊 Acceptance Criteria
| Category | Target Metric |
|-----------|----------------|
| Streaming Reliability | ≥ 24 FPS, < 2 % dropped frames, recover from forced disconnect |
| Latency | ≤ 300 ms median glass-to-glass |
| Vision Accuracy & Speed | ≥ 0.5 F1 on labeled dataset (≥ 150 frames), ≥ 15 FPS inference |
| VR UX & Effects | Readable overlays, unobtrusive HUD, ≥ 1 shader/VFX linked to detections |
| Engineering Quality | Clear architecture, reproducible build, configuration, and logging |

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

---

## 🧰 Tech Stack
- TBD

---

## 🧩 Repository Layout
