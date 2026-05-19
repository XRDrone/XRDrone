# Setup

This project uses a Python virtual environment, `requirements.txt`, and a local Rust extension build.

## Downloads

### Windows

- Python 3.13: download and install from `https://www.python.org/downloads/windows/`
- Rust (`rustup-init.exe`): download and install from `https://www.rust-lang.org/tools/install`
- NVIDIA CUDA Toolkit 12.8: download from `https://developer.nvidia.com/cuda-12-8-0-download-archive`

After installing Python and Rust, open a new terminal and verify:

```powershell
python --version
rustc --version
cargo --version
nvidia-smi
```

## Windows

From the project folder:

```powershell
py -m venv .venv
.venv\Scripts\Activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

bash build_native.sh
maturin develop --release

python -c "import torch; print('torch:', torch.__version__); print('cuda build:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU found')"
python -c "import xrdrone_native; print(xrdrone_native.__file__)"
```

## macOS / Linux

### Downloads

- Python 3.13: download from `https://www.python.org/downloads/`
- Rust: install from `https://www.rust-lang.org/tools/install`
- Linux CUDA 12.8 archive: `https://developer.nvidia.com/cuda-12-8-0-download-archive`
- Linux CUDA installation guide: `https://docs.nvidia.com/cuda/cuda-installation-guide-linux/`

Verify:

```bash
python3 --version
rustc --version
cargo --version
```

From the project folder:

```bash
python3 -m venv yolovenv
source yolovenv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

bash build_native.sh
maturin develop --release

python -c "import torch; print('torch:', torch.__version__); print('cuda build:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU found')"
python -c "import xrdrone_native; print(xrdrone_native.__file__)"
```

## Optional: enable pre-commit

```bash
pre-commit install
```

After installing the hook, the workflow is still:

```bash
git add .
git commit -m "your message"
git push
```

Installing `pre-commit` from `requirements.txt` only installs the Python package. Each teammate must still run `pre-commit install` once in their own local clone if they want the hook to run automatically on commit.

## Verify CUDA

A successful Windows CUDA verification usually shows:

- a Torch version
- `cuda build: 12.8`
- `cuda available: True`
- your GPU name instead of `No GPU found`

If `cuda available` is `False`, the most common causes are:

- the virtual environment is not activated
- PyTorch was installed without the CUDA wheel
- the system does not have a supported NVIDIA GPU
- the NVIDIA driver is missing or outdated

## Notes

- `requirements.txt` includes the main project dependencies, including `torch==2.10.0` and `torchvision==0.25.0`. The CUDA reinstall step replaces those with the CUDA-enabled builds for Windows.
- After activation, use the virtual environment's Python for running the project and tests.
- `pre-commit` is optional. If you do not install the hook, Git still works normally and checks must be run manually.
- The helper modules `id_flicker_mitigation.py`, `output_formatter.py`, `world_projection.py`, `adaptive_tuning.py`, and `motion_smoothing.py` keep their Python filenames but now delegate hot-path work to `xrdrone_native`.
- The native source is organized across `src/lib.rs`, `src/common.rs`, `src/geometry.rs`, `src/id_flicker.rs`, `src/world_projection.rs`, `src/udp.rs`, `src/adaptive_tuning.rs`, and `src/smoothing.rs`. Rebuild the extension after changes to any of those files.


## MediaMTX + ORB-SLAM middle-man setup

For the fused detector + ORB-SLAM workflow, use one shared stream source. A typical setup is:

1. Publish the camera or test video through MediaMTX.
2. Point the detector branch at that stream.
3. Point ORB-SLAM at the same stream.
4. Configure ORB-SLAM to send pose JSON packets to `ORBSLAM_UDP_LISTEN_IP:ORBSLAM_UDP_PORT`.
5. Start the XRDrone runtime with `ORBSLAM_FUSION_ENABLED = True`.

The XRDrone runtime does not launch ORB-SLAM for you. It expects the external process to already be sending UDP packets in this shape:

```json
{
  "source": "orbslam",
  "frame_id": 123,
  "timestamp": 1403636858.7517,
  "pose_valid": true,
  "tracking_state": "ok",
  "x": 0.023807,
  "y": -0.013339,
  "z": -0.007875,
  "qx": -0.001431,
  "qy": 0.004988,
  "qz": 0.000461,
  "qw": 0.999986
}
```

The middle-man binds a UDP socket, buffers recent ORB-SLAM packets, matches on `frame_id` first, and uses timestamp fallback when the frame counter does not line up exactly.
