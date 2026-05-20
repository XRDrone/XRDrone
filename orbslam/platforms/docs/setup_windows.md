# Windows Setup for the XRDrone ORB-SLAM3 Pipeline

This guide continues from `setup_linux.md`. Complete the Linux-side setup first so that the Linux machine can:

1. receive the drone/controller capture-card feed,
2. split the video stream,
3. publish RTSP video to the Windows machine, and
4. send ORB-SLAM3 pose packets to the Windows machine over UDP.

The Windows machine is responsible for receiving the RTSP stream through MediaMTX, running YOLOv5nu person detection and pose fusion, and sending compact UDP JSON scene-state packets to Unity.

---

## 1. Windows-side pipeline overview

```text
Linux machine
  ├── FFmpeg RTSP publish  ───────►  Windows MediaMTX
  └── ORB-SLAM3 UDP pose   ───────►  Windows Python fusion script

Windows machine
  ├── MediaMTX receives RTSP stream at /dji
  ├── Python reads RTSP frames with OpenCV
  ├── YOLOv5nu detects and tracks people
  ├── Python receives the newest ORB-SLAM3 pose packet over UDP
  ├── Python projects detections onto the ground plane when pose is valid/fresh
  └── Python sends UDP JSON scene-state packets to Unity

Unity 6
  └── receives UDP JSON and updates the XR scene
```

The Windows side should be started after MediaMTX is ready and before or during the Linux FFmpeg/ORB-SLAM3 run.

---

## 2. Expected Windows folder layout

Use the `windows/` folder for the Windows-side runtime files:

```text
windows/
├── .venv/
├── requirements.txt
├── rtsp_yolo_orbslam_fusion.py
└── yolov5nu.pt
```
---

## 3. Prerequisites

Install or prepare the following on the Windows machine:

- Windows 10 or Windows 11
- Python compatible with the project dependencies
- PowerShell or Command Prompt
- Git, if cloning or updating the repository
- MediaMTX
- Unity 6 project with the XRDrone UDP receiver scene/script
- `requirements.txt`
- `rtsp_yolo_orbslam_fusion.py`
- `yolov5nu.pt`
- Network access to the Linux machine

Optional but useful:

- FFmpeg / FFplay for testing RTSP streams
- NVIDIA GPU drivers if running YOLO on CUDA
- Windows Terminal for multiple concurrent terminals

---

## 4. Create and activate the Python virtual environment

Open PowerShell in the `windows/` folder:

```powershell
cd path\to\XRDrone\windows
```

Create the virtual environment:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, allow local scripts for the current user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

---

## 5. Install Python dependencies

If `requirements.txt` already exists, install from it:

```powershell
pip install -r requirements.txt
```

If a requirements file has not been created yet, install the core runtime packages manually:

```powershell
pip install opencv-python numpy ultralytics
```

Then install PyTorch using the command recommended by the official PyTorch selector for the Windows machine and CUDA version. For CPU-only testing, choose the CPU option. For GPU testing, choose the Windows + Pip + CUDA option that matches the installed NVIDIA driver/CUDA support.

After installation, verify PyTorch:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Verify the other packages:

```powershell
python -c "import cv2, numpy, ultralytics; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__); print('Ultralytics import OK')"
```

If dependencies were installed manually, update `requirements.txt`:

```powershell
pip freeze > requirements.txt
```

---

## 6. Install and start MediaMTX

Download the Windows release of MediaMTX from the official MediaMTX release page and extract it to a stable local folder, for example:

```text
C:\MediaMTX\
```

Expected file:

```text
C:\MediaMTX\mediamtx.exe
```

Start MediaMTX from PowerShell:

```powershell
cd C:\MediaMTX
.\mediamtx.exe
```

Keep this terminal open while running the pipeline.

### Optional MediaMTX API setting

The receiver monitor checks the MediaMTX API at:

```text
http://127.0.0.1:9997/v3/paths/list
```

If the API is not enabled in the local `mediamtx.yml`, enable it:

```yaml
api: yes
apiAddress: :9997
```

Restart MediaMTX after editing the configuration.

---

## 7. Open the required Windows firewall ports

The Linux machine publishes the RTSP stream to MediaMTX on the Windows machine. Allow inbound RTSP traffic:

```powershell
New-NetFirewallRule `
  -DisplayName "XRDrone MediaMTX RTSP" `
  -Direction Inbound `
  -Action Allow `
  -Protocol TCP `
  -LocalPort 8554
```

Allow inbound ORB-SLAM3 pose packets. Use the same UDP port configured in the Linux ORB-SLAM3 launch command and in the Windows fusion script:

```powershell
New-NetFirewallRule `
  -DisplayName "XRDrone ORB-SLAM3 Pose UDP" `
  -Direction Inbound `
  -Action Allow `
  -Protocol UDP `
  -LocalPort <POSE_UDP_PORT>
```

If Unity runs on the same Windows machine as the Python script, the Unity UDP output can use `127.0.0.1` and usually does not require an inbound firewall rule. If Unity runs on a different machine, also allow the Unity UDP JSON port.

---

## 8. Choose one consistent port configuration

The Linux setup, Windows fusion script, receiver monitor, and Unity receiver must agree on the same ports.

Recommended continuation from `setup_linux.md`:

| Purpose | Recommended value | Where it must match |
|---|---:|---|
| RTSP server port | `8554` | Linux FFmpeg publish URL, MediaMTX, Windows fusion RTSP URL |
| RTSP stream path | `dji` | Linux FFmpeg publish URL, MediaMTX path, Windows fusion RTSP URL |
| ORB-SLAM3 pose UDP port | `9000` | Linux ORB-SLAM3 UDP target, Windows fusion `--pose-port`, optional monitor UDP pose setting |
| Unity JSON UDP port | `9002` | Windows fusion `--output-port`, Unity UDP receiver, optional monitor Unity setting |

If the script defaults are different, either edit the script defaults or pass command-line arguments when running it. The important part is consistency.

---

## 9. Configure the fusion script

The Windows fusion script can be configured either by editing constants near the top of `rtsp_yolo_orbslam_fusion.py` or by passing command-line arguments.

Key settings:

| Setting | Purpose | Recommended value |
|---|---|---|
| `RTSP_URL` / `--rtsp-url` | RTSP stream read by OpenCV | `rtsp://127.0.0.1:8554/dji` if MediaMTX runs on the same Windows machine |
| `YOLO_MODEL_PATH` / `--model` | YOLOv5nu model file | `.\yolov5nu.pt` or the local path to the model file |
| `POSE_LISTEN_IP` / `--pose-listen-ip` | Local address for receiving ORB-SLAM3 pose packets | `0.0.0.0` |
| `POSE_PORT` / `--pose-port` | UDP pose input port | same as Linux ORB-SLAM3 pose target, for example `9000` |
| `POSE_FORMAT` / `--pose-format` | Pose packet format | `orbslam-text` |
| `UNITY_OUTPUT_HOST` / `--output-host` | Unity receiver address | `127.0.0.1` if Unity runs locally |
| `UNITY_OUTPUT_PORT` / `--output-port` | Unity UDP JSON receiver port | same as Unity receiver, for example `9002` |
| `DEVICE` / `--device` | YOLO inference device | `0` for CUDA GPU 0, or `cpu` |

The fusion script expects ORB-SLAM3 text pose packets in this format:

```text
frame timestamp x y z qx qy qz qw
```

---

## 10. Verify the model file

From the activated virtual environment:

```powershell
Test-Path .\yolov5nu.pt
```

Expected output:

```text
True
```

If the output is `False`, either move the model file into the `windows/` folder or pass the correct path when running the script:

```powershell
python .\rtsp_yolo_orbslam_fusion.py --model "path\to\yolov5nu.pt"
```

---

## 11. Start the Linux-side stream

On the Linux machine, run the Linux setup sequence from `setup_linux.md`:

1. connect the capture-card feed,
2. create the `v4l2loopback` virtual camera,
3. start the FFmpeg split that publishes RTSP to Windows,
4. start ORB-SLAM3 and send UDP pose output to the Windows IP and pose port.

The Linux FFmpeg RTSP target should follow this format:

```text
rtsp://<WINDOWS_IP>:8554/dji
```

The Linux ORB-SLAM3 pose target should follow this format:

```text
<WINDOWS_IP> <POSE_UDP_PORT>
```

Use local IP addresses only in local commands or local config files. Keep committed documentation generic.

---

## 12. Confirm MediaMTX receives the RTSP stream

After Linux starts publishing, the MediaMTX terminal should show activity for the `dji` path.

If the MediaMTX API is enabled, check active paths from PowerShell:

```powershell
Invoke-WebRequest http://127.0.0.1:9997/v3/paths/list
```

Optional RTSP playback test with FFplay:

```powershell
ffplay rtsp://127.0.0.1:8554/dji
```

If FFplay is not installed, skip this check and use the Python fusion script as the stream test.

---

## 13. Run the Windows fusion script

From the `windows/` folder with the virtual environment activated:

```powershell
python .\rtsp_yolo_orbslam_fusion.py `
  --rtsp-url "rtsp://127.0.0.1:8554/dji" `
  --model ".\yolov5nu.pt" `
  --pose-listen-ip "0.0.0.0" `
  --pose-port 9000 `
  --pose-format "orbslam-text" `
  --output-host "127.0.0.1" `
  --output-port 9002 `
  --device 0 `
  --show
```

For CPU-only testing, use:

```powershell
--device cpu
```

Expected terminal output should include:

```text
[pose] Listening for ORB-SLAM pose UDP on 0.0.0.0:<POSE_UDP_PORT>
[video] Opening RTSP stream: rtsp://127.0.0.1:8554/dji
[yolo] Loading model: .\yolov5nu.pt
[udp] Sending fused packets to 127.0.0.1:<UNITY_UDP_PORT>
```

If `--show` is enabled, an OpenCV preview window should display the RTSP video with detection overlays. The overlay should report whether pose is available or missing/stale.

---

## 14. Start Unity

Open the Unity 6 XRDrone project on the Windows machine.

Confirm that the Unity UDP receiver listens on the same port used by the Python fusion script:

```text
Unity UDP JSON port = <UNITY_UDP_PORT>
```

For the recommended continuation from `setup_linux.md`:

```text
Unity UDP JSON port = 9002
```

Start the Unity scene containing the XRDrone UDP receiver. When packets arrive, Unity should update the XR scene with the received frame metadata, detection state, pose-valid/world-valid state, and optional world-space coordinates.

---

## 15. Optional receiver monitor

If `xrdrone_pipeline_monitor.py` is included in the Windows folder, it can be used to check the receiver-side services and data paths.

Run it from a separate PowerShell terminal:

```powershell
cd path\to\XRDrone\windows
.\.venv\Scripts\Activate.ps1
python .\xrdrone_pipeline_monitor.py
```

The monitor checks:

- Windows network connection
- Linux reachability
- MediaMTX process/API
- RTSP stream path
- incoming ORB-SLAM3 UDP pose traffic
- YOLO-related Python process
- Unity process
- JSON output status

If the monitor uses different default ports than the fusion script, update the monitor configuration or the fusion-script arguments so both refer to the same runtime ports.

---

## 16. Recommended startup order

Use this order for a live run:

1. Connect the Linux and Windows machines to the same network.
2. Confirm the Windows IP address.
3. Start MediaMTX on Windows.
4. Start the Linux FFmpeg split stream and publish to `rtsp://<WINDOWS_IP>:8554/dji`.
5. Start ORB-SLAM3 on Linux and send pose UDP packets to `<WINDOWS_IP>:<POSE_UDP_PORT>`.
6. Activate the Windows `.venv`.
7. Run `rtsp_yolo_orbslam_fusion.py` on Windows.
8. Start the Unity scene with the UDP receiver.
9. Optionally run the Windows receiver monitor in a separate terminal.

---

## 17. Quick validation checklist

Before a test run, verify:

- [ ] MediaMTX is running on Windows.
- [ ] Windows firewall allows inbound TCP `8554`.
- [ ] Windows firewall allows the selected inbound UDP pose port.
- [ ] Linux FFmpeg is publishing to `rtsp://<WINDOWS_IP>:8554/dji`.
- [ ] The Windows fusion script reads `rtsp://127.0.0.1:8554/dji` or the equivalent Windows RTSP URL.
- [ ] ORB-SLAM3 sends pose packets to the same UDP port that the Windows script listens on.
- [ ] `yolov5nu.pt` is reachable by the fusion script.
- [ ] PyTorch can import successfully.
- [ ] CUDA is available if using `--device 0`.
- [ ] Unity listens on the same UDP JSON port that the fusion script sends to.
- [ ] The Unity scene is running before expecting visual updates.

---

## 18. Troubleshooting

### `Could not open RTSP stream`

Check:

1. MediaMTX is running on Windows.
2. Linux is publishing to the correct Windows IP.
3. Windows firewall allows TCP `8554`.
4. The stream path is exactly `dji`.
5. The Windows fusion script uses the correct RTSP URL.

Try:

```powershell
ffplay rtsp://127.0.0.1:8554/dji
```

### Pose shows `missing/stale`

Check:

1. ORB-SLAM3 is running on Linux.
2. ORB-SLAM3 is sending UDP pose packets to the Windows IP.
3. The Linux target pose port matches the Windows `--pose-port` value.
4. Windows firewall allows the selected UDP pose port.
5. The pose format matches `frame timestamp x y z qx qy qz qw`.

### YOLO model file not found

Check:

```powershell
Test-Path .\yolov5nu.pt
```

Then run with an explicit model path:

```powershell
python .\rtsp_yolo_orbslam_fusion.py --model "path\to\yolov5nu.pt"
```

### CUDA is not available

Check:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If CUDA is unavailable, either reinstall PyTorch using the official Windows CUDA command for the machine or run the fusion script with:

```powershell
--device cpu
```

### Unity receives no packets

Check:

1. Unity is running the scene with the UDP receiver.
2. The Unity UDP receiver port matches the fusion script `--output-port`.
3. The fusion script is sending to the correct host.
4. If Unity is local, use `127.0.0.1` as the output host.
5. If Unity is remote, use that machine's IP and allow inbound UDP on the Unity machine.

### High video latency

The fusion script is configured to prefer fresh frames:

```text
OPENCV_FFMPEG_CAPTURE_OPTIONS = rtsp_transport;tcp|fflags;nobuffer|flags;low_delay
CAP_PROP_BUFFERSIZE = 1
DROP_STALE_GRABS = 3
```

If latency remains high, try:

- lowering the RTSP resolution on the Linux FFmpeg command,
- lowering the YOLO image size with `--imgsz 480`,
- increasing `--drop-stale-grabs`,
- using CUDA instead of CPU,
- checking that MediaMTX, Python, and Unity are not competing for GPU/CPU resources.

---

## 19. Minimal live-run command set

Terminal 1 — MediaMTX:

```powershell
cd C:\MediaMTX
.\mediamtx.exe
```

Terminal 2 — Windows fusion:

```powershell
cd path\to\XRDrone\windows
.\.venv\Scripts\Activate.ps1
python .\rtsp_yolo_orbslam_fusion.py `
  --rtsp-url "rtsp://127.0.0.1:8554/dji" `
  --model ".\yolov5nu.pt" `
  --pose-listen-ip "0.0.0.0" `
  --pose-port 9000 `
  --pose-format "orbslam-text" `
  --output-host "127.0.0.1" `
  --output-port 9002 `
  --device 0 `
  --show
```

Terminal 3 — optional monitor:

```powershell
cd path\to\XRDrone\windows
.\.venv\Scripts\Activate.ps1
python .\xrdrone_pipeline_monitor.py
```

Unity should be open and listening on the configured UDP JSON port before expecting XR scene updates.
