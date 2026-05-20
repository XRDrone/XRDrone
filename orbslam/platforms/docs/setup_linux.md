# Linux Setup for the XRDrone ORB-SLAM3 Pipeline

This guide documents the Linux-side setup for the ORB-SLAM3-based XRDrone pipeline. The Linux machine is responsible for:

1. receiving the HDMI capture-card feed from the DJI RC 2 controller,
2. creating a virtual camera device for ORB-SLAM3,
3. splitting the live video stream with FFmpeg,
4. publishing the video branch to the Windows backend over RTSP,
5. running ORB-SLAM3 and sending pose output to the Windows fusion/Unity side.

The Windows side should run MediaMTX, the YOLO/person-detection fusion process, and Unity.

---

## 1. Pipeline overview

The ORB-SLAM3 path uses a split-machine workflow:

```text
DJI Neo
  ↓ wireless video
DJI RC 2 controller
  ↓ USB-C-to-HDMI adapter
HDMI capture card
  ↓ USB-A-to-USB-A USB 3.0
Linux machine
  ├── v4l2loopback virtual camera → ORB-SLAM3 pose estimation → UDP pose packets
  └── FFmpeg RTSP stream → Windows MediaMTX → YOLO/fusion → Unity UDP JSON
```

The Linux machine should expose one virtual video device for ORB-SLAM3 and one RTSP stream for the Windows backend. This avoids having ORB-SLAM3 and the detection pipeline compete for the same physical capture-card device.

---

## 2. Prerequisites

Install or prepare the following on the Linux machine:

- Ubuntu Linux
- Docker
- Git
- FFmpeg
- v4l2loopback
- v4l-utils
- USB/HDMI capture card connected to the DJI RC 2 controller output
- access to the Windows backend IP address
- access to the ORB-SLAM3 Docker repository:

```text
https://github.com/jahaniam/orbslam3_docker
```

The referenced ORB-SLAM3 Docker repository provides CPU and CUDA container options. Use the CPU version unless the Linux machine is configured for NVIDIA Docker/CUDA.

---

## 3. Install Linux video tools

```bash
sudo apt update
sudo apt install -y git docker.io ffmpeg v4l2loopback-dkms v4l-utils
```

Start and enable Docker:

```bash
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

After adding the user to the `docker` group, log out and log back in, or reboot.

Check Docker:

```bash
docker --version
docker run hello-world
```

---

## 4. Clone and build the ORB-SLAM3 Docker environment

Clone the Docker-based ORB-SLAM3 setup:

```bash
git clone https://github.com/jahaniam/orbslam3_docker.git
cd orbslam3_docker
```

Choose the build script that matches the machine:

### CPU build

```bash
chmod +x build_container_cpu.sh
./build_container_cpu.sh
```

### CUDA build

Use this only if `nvidia-smi` works on the Linux host and NVIDIA Docker is configured.

```bash
nvidia-smi
chmod +x build_container_cuda.sh
./build_container_cuda.sh
```

If the build succeeds, the container should compile ORB-SLAM3 and create a running container named similar to:

```text
orbslam3
```

Check the container:

```bash
docker ps
```

---

## 5. Validate ORB-SLAM3 with the EuRoC MH_02_easy dataset

Before using the live drone stream, validate that ORB-SLAM3 works with the standard EuRoC dataset.

Download the EuRoC `MH_02_easy` dataset from:

```text
https://doi.org/10.3929/ethz-b-000690084
```

Use the file named:

```text
MH_02_easy
```

Rename it to:

```text
MH02
```

Move the renamed `MH02` folder into:

```text
orbslam3_docker/Datasets/EuRoC/
```

The final path should look like:

```text
orbslam3_docker/
└── Datasets/
    └── EuRoC/
        └── MH02/
            └── mav0/
                └── cam0/
                    └── data/
                        └── <frame>.png
```

A frame should be reachable through this pattern:

```text
MH02/mav0/cam0/data/#.png
```

Then enter the ORB-SLAM3 container:

```bash
docker exec -it orbslam3 bash
```

From inside the container, follow the upstream repository's EuRoC test instructions. The repository example uses:

```bash
cd /ORB_SLAM3/Examples
bash ./euroc_examples.sh
```

The EuRoC example may take several minutes to initialize.

---

## 6. Connect and identify the HDMI capture card

Connect the hardware in this order:

```text
DJI Neo → DJI RC 2 controller → USB-C-to-HDMI adapter → HDMI capture card → Linux USB 3.0 port
```

List video devices:

```bash
v4l2-ctl --list-devices
```

Look for the HDMI capture card and note its device path, for example:

```text
/dev/video0
```

Inspect supported capture formats:

```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

Test the capture feed locally:

```bash
ffplay -f v4l2 -framerate 30 -video_size 1280x720 /dev/video0
```

If the preview does not open, try another listed resolution or frame rate from `v4l2-ctl --list-formats-ext`.

---

## 7. Create the ORB-SLAM3 virtual camera device

Load `v4l2loopback` and create a virtual camera device for ORB-SLAM3:

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="XRDrone ORB-SLAM3" exclusive_caps=1
```

Verify that the device exists:

```bash
v4l2-ctl --list-devices
ls -l /dev/video10
```

In this guide:

```text
Physical capture-card input: /dev/video0
Virtual ORB-SLAM3 camera:    /dev/video10
```

Adjust these values if the machine assigns different device numbers.

---

## 8. Start MediaMTX on the Windows backend

Before publishing RTSP from Linux, start MediaMTX on the Windows backend.

The RTSP path used by the receiver side is:

```text
dji
```

The expected RTSP URL format from Linux is:

```text
rtsp://<WINDOWS_IP>:8554/dji
```

Replace `<WINDOWS_IP>` with the actual Windows backend IP address on the same network.

Do not hard-code private lab IP addresses in committed documentation. Use placeholders and update them locally.

---

## 9. Split the Linux capture stream

Set local variables on the Linux machine:

```bash
export CAPTURE_DEV=/dev/video0
export VIRTUAL_DEV=/dev/video10
export WINDOWS_IP=<WINDOWS_IP>
export WIDTH=1280
export HEIGHT=720
export FPS=30
export RTSP_URL=rtsp://$WINDOWS_IP:8554/dji
```

Start the split stream:

```bash
ffmpeg \
  -hide_banner \
  -fflags nobuffer \
  -flags low_delay \
  -f v4l2 \
  -framerate $FPS \
  -video_size ${WIDTH}x${HEIGHT} \
  -i $CAPTURE_DEV \
  -filter_complex "[0:v]split=2[vslam][vrtsp]" \
  -map "[vslam]" \
  -pix_fmt yuyv422 \
  -f v4l2 \
  $VIRTUAL_DEV \
  -map "[vrtsp]" \
  -c:v libx264 \
  -preset ultrafast \
  -tune zerolatency \
  -pix_fmt yuv420p \
  -f rtsp \
  -rtsp_transport tcp \
  $RTSP_URL
```

Expected result:

- ORB-SLAM3 can read from `/dev/video10`.
- Windows MediaMTX receives the RTSP stream at `/dji`.

---

## 10. Verify the RTSP stream

On the Windows backend, confirm that MediaMTX shows an active path named:

```text
dji
```

From another machine on the same network, the stream should be testable with:

```bash
ffplay rtsp://<WINDOWS_IP>:8554/dji
```

If the stream fails:

1. confirm Windows and Linux are on the same network,
2. confirm the Windows firewall allows MediaMTX on port `8554`,
3. confirm MediaMTX is running before Linux starts the FFmpeg publisher,
4. confirm the RTSP path is exactly `/dji`,
5. retry with a lower resolution, such as `640x480`.

---

## 11. Run ORB-SLAM3 on the virtual camera

Keep the FFmpeg split stream running.

Enter the ORB-SLAM3 container:

```bash
docker exec -it orbslam3 bash
```

Run the repository's live-camera ORB-SLAM3 launch command and point the camera input to:

```text
/dev/video10
```

Use the live-camera configuration already provided by the project repository. The important requirement is that ORB-SLAM3 reads the virtual camera device created by `v4l2loopback`, not the physical HDMI capture card directly.

Expected behavior:

- ORB-SLAM3 receives frames from `/dev/video10`.
- ORB-SLAM3 estimates camera pose from the live stream.
- Pose output is sent from Linux to the Windows backend over UDP.

---

## 12. Pose UDP output

The Windows backend expects ORB-SLAM3 pose packets from the Linux machine.

Use this placeholder configuration:

```text
Windows pose receiver IP: <WINDOWS_IP>
Pose UDP port:           9000
```

If the ORB-SLAM3 pose sender is configurable, set:

```bash
export POSE_TARGET_IP=<WINDOWS_IP>
export POSE_TARGET_PORT=9000
```

To verify packets from Linux:

```bash
sudo tcpdump -n udp and host <WINDOWS_IP> and port 9000
```

On the Windows receiver side, the monitor should report UDP pose traffic on port `9000`.

---

## 13. Optional clock-offset check

If the receiver panel's clock-offset check is used, the Linux machine must run a small UDP time responder on port:

```text
9001
```

This is optional and only needed for clock-offset diagnostics. It is separate from the main pose stream on port `9000`.

---

## 14. Expected Windows-side receiver values

The Windows backend should be configured to expect:

```text
Linux host:        <LINUX_IP>
RTSP path:         dji
RTSP port:         8554
Pose UDP port:     9000
Unity JSON port:   9002
```

Use local machine IP addresses only in local config files. Keep committed documentation generic.

---

## 15. Troubleshooting

### Capture card does not appear

Run:

```bash
lsusb
v4l2-ctl --list-devices
```

Try another USB port, preferably USB 3.0.

### FFmpeg says the resolution is unsupported

List supported modes:

```bash
v4l2-ctl --device=$CAPTURE_DEV --list-formats-ext
```

Then update:

```bash
export WIDTH=<supported_width>
export HEIGHT=<supported_height>
export FPS=<supported_fps>
```

### `/dev/video10` does not exist

Reload `v4l2loopback`:

```bash
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="XRDrone ORB-SLAM3" exclusive_caps=1
```

### Docker container cannot see `/dev/video10`

Check whether the device exists on the host:

```bash
ls -l /dev/video10
```

If the container was started before the virtual device existed, restart or recreate the container with access to the video device. The container must be able to read the virtual camera path.

### RTSP stream does not reach Windows

Check network reachability:

```bash
ping <WINDOWS_IP>
```

Check that MediaMTX is running on Windows and listening on port `8554`.

Check the FFmpeg command output for RTSP connection errors.

### ORB-SLAM3 starts but does not track

Try the following:

1. verify that the virtual camera shows the live feed,
2. reduce motion and avoid blur during initialization,
3. use a lower resolution if the machine cannot process frames fast enough,
4. improve lighting and scene texture,
5. validate the ORB-SLAM3 environment again with the EuRoC `MH02` test.

---

## 16. Quick command checklist

```bash
# 1. Identify capture card
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext

# 2. Create virtual camera
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="XRDrone ORB-SLAM3" exclusive_caps=1

# 3. Set variables
export CAPTURE_DEV=/dev/video0
export VIRTUAL_DEV=/dev/video10
export WINDOWS_IP=<WINDOWS_IP>
export WIDTH=1280
export HEIGHT=720
export FPS=30
export RTSP_URL=rtsp://$WINDOWS_IP:8554/dji

# 4. Split feed to ORB-SLAM3 and Windows RTSP
ffmpeg \
  -hide_banner \
  -fflags nobuffer \
  -flags low_delay \
  -f v4l2 \
  -framerate $FPS \
  -video_size ${WIDTH}x${HEIGHT} \
  -i $CAPTURE_DEV \
  -filter_complex "[0:v]split=2[vslam][vrtsp]" \
  -map "[vslam]" \
  -pix_fmt yuyv422 \
  -f v4l2 \
  $VIRTUAL_DEV \
  -map "[vrtsp]" \
  -c:v libx264 \
  -preset ultrafast \
  -tune zerolatency \
  -pix_fmt yuv420p \
  -f rtsp \
  -rtsp_transport tcp \
  $RTSP_URL

# 5. Enter ORB-SLAM3 container
docker exec -it orbslam3 bash

# 6. Run the repository's live-camera ORB-SLAM3 command using /dev/video10
```

---

## 17. Reference links

- ORB-SLAM3 Docker repository: `https://github.com/jahaniam/orbslam3_docker`
- EuRoC MAV dataset landing page: `https://doi.org/10.3929/ethz-b-000690084`
- v4l2loopback repository: `https://github.com/v4l2loopback/v4l2loopback`
- FFmpeg repository/documentation entry point: `https://github.com/FFmpeg/FFmpeg`
