# Contributing

Thank you for contributing to XRDrone. This guide describes the workflow for development, testing, documentation updates, and pull requests across the two XRDrone architecture paths:

- `aruco/pipeline/` — ArUco-based pose, detection, logging, Rust helpers, and UDP JSON output.
- `orbslam/platforms/` — ORB-SLAM3 split-machine setup, Linux capture/pose generation, Windows RTSP/YOLO fusion, and Unity UDP output.

XRDrone uses a docs-first structure. When code, setup steps, runtime behavior, packet formats, or folder layout change, update the matching documentation in the same pull request.

---

## 1. Repository Areas

| Area | Purpose | Primary docs |
|---|---|---|
| `aruco/pipeline/` | Pure ArUco video pipeline with Python, OpenCV ArUco, YOLO detection/tracking, Rust helper modules, runtime logging, and UDP JSON output. | `aruco/pipeline/docs/` |
| `orbslam/platforms/` | ORB-SLAM3 split-machine workflow. Linux handles capture-card input, virtual camera creation, FFmpeg stream splitting, RTSP publishing, and ORB-SLAM3 pose output. Windows handles MediaMTX, YOLOv5nu fusion, and Unity-facing UDP JSON. | `orbslam/platforms/docs/` |
| Unity project files | XR scene rendering and UDP receiver behavior. | Update the relevant Unity notes or setup docs when receiver behavior changes. |

Avoid mixing unrelated ArUco and ORB-SLAM3 changes in the same pull request unless the change intentionally affects both architectures.

---

## 2. Development Environment

Set up the environment for the architecture you are changing.

### ArUco pipeline

From the ArUco pipeline folder:

```bash
cd aruco/pipeline
python3 -m venv yolovenv
source yolovenv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
bash build_native.sh
python -c "import xrdrone_native; print('ok')"
```

On Windows with CUDA, install the appropriate PyTorch CUDA wheel for the machine before running GPU tests.

See:

```text
aruco/pipeline/docs/setup.md
```

### ORB-SLAM3 platform workflow

Start from:

```bash
cd orbslam/platforms
```

Then read the platform setup docs in order:

```text
orbslam/platforms/docs/setup_linux.md
orbslam/platforms/docs/setup_windows.md
```

The ORB-SLAM3 path is split across two machines:

1. Linux receives the HDMI capture-card feed, creates the virtual camera, splits the video stream with FFmpeg, publishes RTSP to Windows, and runs ORB-SLAM3 pose output.
2. Windows runs MediaMTX, reads the RTSP stream, runs YOLOv5nu fusion, receives ORB-SLAM3 pose packets, and sends UDP JSON scene-state packets to Unity.

Use placeholders in committed documentation for machine-specific values:

```text
<LINUX_IP>
<WINDOWS_IP>
<POSE_UDP_PORT>
<UNITY_UDP_PORT>
```

Do not commit private IP addresses, usernames, local absolute paths, tokens, credentials, or machine-specific secrets.

---

## 3. Documentation Map

### ArUco documentation

| File | Purpose |
|---|---|
| `aruco/pipeline/docs/setup.md` | Environment setup, native build, and run commands. |
| `aruco/pipeline/docs/settings.md` | Runtime configuration in `settings.py`. |
| `aruco/pipeline/docs/udp-json.md` | Unity-facing UDP JSON packet structure. |
| `aruco/pipeline/docs/runtime-ui-and-terminal-reference.md` | Runtime overlay, terminal output, and controls. |
| `aruco/pipeline/docs/testing.md` | Prerecorded-video, live-video, UDP, and logging tests. |

### ORB-SLAM3 documentation

| File | Purpose |
|---|---|
| `orbslam/platforms/docs/setup_linux.md` | Linux capture-card input, v4l2loopback, FFmpeg stream splitting, RTSP publishing, ORB-SLAM3 Docker validation, and pose output. |
| `orbslam/platforms/docs/setup_windows.md` | Windows MediaMTX, Python dependencies, YOLOv5nu fusion, UDP pose input, and Unity JSON output. |

Update documentation when any of these change:

- setup commands
- folder layout
- required dependencies
- default ports or stream paths
- runtime flags
- UDP packet structure
- logging behavior
- Unity receiver assumptions
- troubleshooting steps

---

## 4. Code Style

Use Ruff for Python linting and formatting.

Run from the folder that contains the Python code you changed:

```bash
ruff check .
ruff format .
```

If automatic fixes are safe:

```bash
ruff check . --fix
ruff format .
```

Optional pre-commit hook:

```bash
pre-commit install
```

Rust changes in the ArUco pipeline must also compile through the native build script:

```bash
bash build_native.sh
```

---

## 5. ArUco Native Module Workflow

The ArUco pipeline uses Rust/PyO3 helper code for selected hot paths while keeping Python-facing module names stable.

Rebuild the native module after changes to:

- `aruco/pipeline/src/*.rs`
- `aruco/pipeline/Cargo.toml`
- `aruco/pipeline/Cargo.lock`
- Python wrappers that call `xrdrone_native`
- build settings related to PyO3 or maturin

Standard rebuild:

```bash
cd aruco/pipeline
source yolovenv/bin/activate
bash build_native.sh
```

Do not commit Rust build output such as `target/`.

---

## 6. ORB-SLAM3 Workflow Expectations

ORB-SLAM3 changes should preserve the split-machine contract documented in `orbslam/platforms/docs/`.

The expected live flow is:

```text
Linux capture card
  -> v4l2loopback virtual camera
  -> ORB-SLAM3 pose output over UDP

Linux capture card
  -> FFmpeg RTSP publish
  -> Windows MediaMTX
  -> Windows YOLOv5nu fusion
  -> Unity UDP JSON
```

When changing the ORB-SLAM3 path, verify that the Linux and Windows docs remain consistent with each other for:

- RTSP path, usually `dji`
- RTSP port, usually `8554`
- ORB-SLAM3 pose UDP port, usually `9000`
- Unity JSON UDP port, usually `9002`
- pose-packet format
- startup order
- firewall expectations
- placeholder IP usage

The live ORB-SLAM3 sender is expected to output pose data compatible with the Windows fusion process. If the sender format changes, update the Windows setup documentation and any receiver/fusion notes in the same pull request.

Do not commit EuRoC datasets, Docker build artifacts, generated ORB-SLAM3 build directories, or local machine configuration.

---

## 7. Testing Changes

Use the smallest validation workflow that proves the change.

### ArUco changes

Typical validation:

```bash
cd aruco/pipeline
source yolovenv/bin/activate
ruff check .
ruff format .
bash build_native.sh
python main.py
```

For logged runtime validation:

```bash
python main.py --logs
```

For headless logged validation:

```bash
python main.py --logs --no-gui
```

If UDP schema or packet formatting changed:

```bash
python test_with_coverage.py
```

### ORB-SLAM3 documentation or platform changes

For docs-only changes, verify that:

- links point to existing files,
- commands use placeholders instead of private machine values,
- Linux and Windows setup docs agree on ports and stream paths,
- startup order is clear,
- no dataset or generated build artifact is referenced as a committed source file.

For runtime or script changes, validate the relevant side:

- Linux: capture-card discovery, `v4l2loopback`, FFmpeg split, RTSP publish, ORB-SLAM3 pose output.
- Windows: MediaMTX path, RTSP read, YOLOv5nu load, UDP pose input, Unity UDP JSON output.

If a full live test is not possible, state exactly what was tested and what remains unverified in the pull request.

---

## 8. UDP JSON Protocol Changes

Any change to Unity-facing packet structure must be documented.

For ArUco packet changes, update:

```text
aruco/pipeline/docs/udp-json.md
```

For ORB-SLAM3 fusion packet changes, update the relevant ORB-SLAM3 setup or receiver documentation under:

```text
orbslam/platforms/docs/
```

Packet changes should preserve compatibility with Unity when possible. If compatibility breaks, clearly document the required Unity-side update.

---

## 9. Configuration Changes

Configuration changes must be documented with safe defaults.

For ArUco settings:

1. Update `aruco/pipeline/settings.py`.
2. Update `aruco/pipeline/docs/settings.md`.
3. Verify the default run still works.

For ORB-SLAM3 settings or launch arguments:

1. Update the relevant Linux or Windows setup document.
2. Keep private IP addresses and local paths out of committed files.
3. Make sure Linux sender settings and Windows receiver settings still match.

---

## 10. Branch Workflow

Create a focused branch from the latest main branch:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c feature/your-feature
```

Make changes, run checks, then inspect what will be committed:

```bash
git status -sb
git diff --cached --name-status
```

Commit with a short descriptive message:

```bash
git add <files>
git commit -m "Short descriptive message"
git push -u origin feature/your-feature
```

Open a pull request after pushing.

---

## 11. Pull Request Requirements

Before submitting a pull request, confirm:

- the branch is focused on one architecture or one shared concern,
- Python code passes `ruff check .`,
- Python code is formatted with `ruff format .`,
- Rust/native changes build successfully when the ArUco native module is affected,
- documentation is updated for setup, runtime, packet, or configuration changes,
- no private machine values or credentials are committed,
- no generated folders, datasets, logs, archives, or local environments are committed,
- testing performed is described clearly in the PR.

A good PR description should include:

```text
Purpose:
Testing:
Documentation updated:
Known limitations:
```

---

## 12. Do Not Commit

Do not commit local, generated, or sensitive files.

Common examples:

```text
.DS_Store
__pycache__/
*.pyc
.venv/
yolovenv/
target/
logs/
runs/
*.zip
*.tar
*.mp4
*.mov
*.avi
Datasets/
orbslam3_docker/Datasets/
ORB_SLAM3/build/
ORB_SLAM3/Examples/*/build/
Library/
Temp/
Obj/
Build/
Builds/
Logs/
.vs/
```

Large model weights, videos, Unity assets, and third-party source trees should only be added when they are intentionally part of the repository and approved for the pull request.

Never commit:

- passwords,
- tokens,
- SSH keys,
- private hostnames,
- personal absolute paths,
- private IP addresses in committed configuration,
- machine-specific local settings.

---

## 13. Commit Message Guidelines

Use concise, action-oriented commit messages.

Examples:

```text
Add ArUco runtime logging
Update ORB-SLAM3 Linux setup guide
Document Windows fusion startup order
Fix UDP packet validation test
Run Ruff formatting
Ignore generated runtime artifacts
```

---

## 14. Questions

For large architecture changes, open an issue or discuss the approach before implementing. This is especially important for changes that affect:

- Unity-facing UDP packet schema,
- ORB-SLAM3 sender output format,
- Linux/Windows startup order,
- model files or large assets,
- Rust/PyO3 native helper behavior,
- cross-architecture documentation.
