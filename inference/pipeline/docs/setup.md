# Setup

This project uses a Python virtual environment and `requirements.txt` for dependencies. On Windows, the most reliable CUDA setup was:

1. Create and activate a virtual environment.
2. Install `requirements.txt`.
3. Reinstall PyTorch with the CUDA 12.8 wheels.
4. Verify that CUDA is available in Python.
5. Optionally install the local Git pre-commit hook.

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

## Optional: enable pre-commit

The repository includes a `.pre-commit-config.yaml` file and if you want linting or formatting to run automatically before each commit, run:

```powershell
pre-commit install
```

This command installs a local Git hook for your clone only. It does not replace normal Git commands.

After installing the hook, the workflow is still:

```powershell
git add .
git commit -m "your message"
git push
```

The difference is that `pre-commit` will run automatically when `git commit` starts. If the checks pass, the commit continues as normal. If a hook finds an issue or rewrites files, the commit may stop so you can review the changes, run `git add .` again if needed, and re-run `git commit`.

Installing `pre-commit` from `requirements.txt` only installs the Python package. Each teammate must still run `pre-commit install` once in their own local clone if they want the hook to run automatically on commit.

## What each step does

- `py -m venv .venv` creates a local virtual environment.
- `.venv\Scripts\Activate` activates it.
- `pip install -r requirements.txt` installs the project dependencies.
- `pip uninstall ...` removes any non-CUDA PyTorch packages that may have been installed from `requirements.txt`.
- `pip install ... --index-url https://download.pytorch.org/whl/cu128` reinstalls PyTorch with CUDA 12.8 support.
- The final `python -c` command checks the installed Torch version, CUDA build, and whether a GPU is visible.
- `pre-commit install` installs the local Git hook that runs configured checks before `git commit`.

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
- `pre-commit` is optional. If you do not install the hook, Git still works normally and checks must be run manually.
