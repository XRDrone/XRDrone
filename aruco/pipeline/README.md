# XRDrone Pure ArUco Video Runner

This folder contains the ArUco-based backend runner for XRDrone. It supports prerecorded and live video input, marker-based camera pose estimation, optional human detection, UDP JSON scene-state output, and runtime logging.

For setup, usage, packet format, and testing details, start with the documentation in [`docs/`](./docs/).

## Documentation

| File | Purpose |
|---|---|
| [`docs/setup.md`](./docs/setup.md) | Environment setup and runtime commands |
| [`docs/settings.md`](./docs/settings.md) | Configuration options in `settings.py` |
| [`docs/udp-json.md`](./docs/udp-json.md) | Unity-facing UDP JSON packet structure |
| [`docs/runtime-ui-and-terminal-reference.md`](./docs/runtime-ui-and-terminal-reference.md) | Runtime overlay, terminal output, and controls |
| [`docs/testing.md`](./docs/testing.md) | Prerecorded-video, live-video, and logging test workflow |

## Quick Start

From this folder:

```bash
python main.py
```

Run with logs:

```bash
python main.py --logs
```

Run headless with logs:

```bash
python main.py --logs --no-gui
```