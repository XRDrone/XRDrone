
# Contributing

Thank you for contributing to this project. This document describes the current workflow for making changes,
running the pipeline, and submitting pull requests.

The contribution process aligns with the repository documentation:
- Setup instructions
- UDP JSON protocol documentation
- Testing procedures
- Runtime configuration via settings
- Python dependency environment

---

# 1. Development Environment

Follow the setup guide before contributing.

## Create environment

```bash
python3 -m venv yolovenv
source yolovenv/bin/activate
pip install -r requirements.txt
```

Some contributors may need CUDA builds of PyTorch. If CUDA support is required:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

# 2. Repository Structure

Key documentation files:

- `README.md` – high‑level project overview
- `setup.md` – environment setup instructions
- `testing.md` – pipeline testing procedures
- `udp-json.md` – UDP packet schema used for Unity communication
- `settings.md` – runtime configuration parameters
- `requirements.txt` – Python dependencies

Developers should review these files before making structural changes.

---

# 3. Development Guidelines

## Code Style

The repository uses:

- **Ruff** for linting
- **Ruff format** for code formatting
- Python virtual environments

Run checks before committing:

```bash
ruff check .
ruff format .
```

---

# 4. Branch Workflow

1. Create a feature branch

```bash
git checkout -b feature/your-feature
```

2. Make your changes

3. Run linting and tests

4. Commit changes

```bash
git add .
git commit -m "Short descriptive message"
```

5. Push branch

```bash
git push origin feature/your-feature
```

6. Open a Pull Request

---

# 5. Testing Changes

Testing instructions are documented in `testing.md`.

Typical workflow:

1. Run the inference pipeline
2. Verify UDP JSON packets are produced correctly
3. Confirm Unity receives detections and pose data
4. Validate object tracking and marker pose estimation

When contributing algorithm changes (tracking, smoothing, pose solving),
include a demonstration or reproduction method.

---

# 6. UDP JSON Protocol Changes

Changes to the UDP schema must:

1. Be documented in `udp-json.md`
2. Maintain backward compatibility when possible
3. Include testing verification

Unity components rely on the schema remaining stable.

---

# 7. Configuration Changes

Runtime behavior is controlled by `settings.py`.

Examples include:

- detection thresholds
- smoothing parameters
- tracking configuration
- pipeline behavior flags

If new configuration options are added:

1. Document them in `settings.md`
2. Provide default values
3. Maintain backwards compatibility

---

# 8. Pull Request Requirements

Before submitting a PR ensure:

- Code passes `ruff check`
- Code is formatted (`ruff format`)
- The pipeline runs without runtime errors
- Documentation is updated if behavior changes
- No unnecessary files are committed

Do **not commit**:

- `.DS_Store`
- `__pycache__/`
- temporary media or test videos
- local environment files

---

# 9. Commit Message Guidelines

Use concise, descriptive commit messages.

Examples:

```
Add motion smoothing for ArUco pose updates
Optimize UDP packet serialization
Implement Bot‑Sort tracking integration
Update documentation for pipeline testing
```

---

# 10. Documentation Updates

Documentation should remain synchronized with implementation.

Update the following when needed:

- `README.md`
- `udp-json.md`
- `testing.md`
- `settings.md`
- `setup.md`

If a change alters the pipeline architecture or packet format,
documentation updates are **required**.

---

# 11. Questions

If unsure about architecture decisions or pipeline changes,
open an issue before implementing major modifications.
