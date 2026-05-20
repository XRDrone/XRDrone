# XRDrone ORB-SLAM3 Pipeline

This folder contains the ORB-SLAM3-based XRDrone split-machine pipeline documentation and Windows-side runtime files. The workflow uses a Linux machine for video capture, stream splitting, virtual-camera output, and ORB-SLAM3 pose generation. The Windows machine receives the RTSP stream, runs YOLOv5nu-based fusion, and sends UDP JSON scene-state packets to Unity.

For setup, start with the documentation in [`docs/`](./docs/). Complete the setup in this order:

```text
Linux setup → Windows setup → Unity receiver
```

## Documentation

| File | Purpose |
|---|---|
| [`docs/setup_linux.md`](./docs/setup_linux.md) | Linux-side setup for capture-card input, v4l2loopback, FFmpeg stream splitting, RTSP publishing, and ORB-SLAM3 pose output |
| [`docs/setup_windows.md`](./docs/setup_windows.md) | Windows-side setup for MediaMTX, Python dependencies, YOLOv5nu fusion, UDP pose input, and Unity JSON output |

## Setup Order

1. Read and complete [`docs/setup_linux.md`](./docs/setup_linux.md).
2. Confirm the Linux machine can publish the RTSP stream and send ORB-SLAM3 pose packets.
3. Read and complete [`docs/setup_windows.md`](./docs/setup_windows.md).
4. Start MediaMTX, run the Windows fusion script, and open the Unity receiver scene.

## Notes

- Keep machine-specific IP addresses, ports, and local paths in local configuration only.
- Make sure the Linux and Windows machines use matching RTSP and UDP port settings.
- Use the documentation files as the source of truth for the full setup commands and troubleshooting steps.
