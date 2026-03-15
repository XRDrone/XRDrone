# Setup

This project uses a Python virtual environment and `requirements.txt` for dependencies. On Windows, the most reliable CUDA setup was:

1. Create and activate a virtual environment.
2. Install `requirements.txt`.
3. Reinstall PyTorch with the CUDA 12.8 wheels.
4. Verify that CUDA is available in Python.

## Windows

From the project folder:

```powershell
py -m venv .venv
.venv\Scripts\Activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

python -c "import torch; print('torch:', torch.__version__); print('cuda build:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU found')"
```

## What each step does

- `py -m venv .venv` creates a local virtual environment.
- `.venv\Scripts\Activate` activates it.
- `pip install -r requirements.txt` installs the project dependencies.
- `pip uninstall ...` removes any non-CUDA PyTorch packages that may have been installed from `requirements.txt`.
- `pip install ... --index-url https://download.pytorch.org/whl/cu128` reinstalls PyTorch with CUDA 12.8 support.
- The final `python -c` command checks the installed Torch version, CUDA build, and whether a GPU is visible.

## Verify CUDA

A successful verification usually shows:

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
