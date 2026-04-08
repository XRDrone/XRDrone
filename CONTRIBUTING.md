# Contributing

Thank you for contributing to XRDrone. This document describes the workflow for development, testing, documentation updates, and submitting pull requests.

The repository uses a docs-first structure, so detailed documentation lives in the `docs/` directory rather than in the top-level README.

---

# 1. Development Environment

Before contributing, set up both the Python environment and the Rust native build used by the mixed Python + Rust pipeline.

Recommended steps:

```bash
python3 -m venv yolovenv
source yolovenv/bin/activate
pip install -r requirements.txt
bash build_native.sh
python -c "import xrdrone_native; print('ok')"
```

If CUDA support is required on Windows:

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
bash build_native.sh
```

See: `docs/setup.md`

Each contributor must build the native module in their own local clone and active virtual environment.

---

# 2. Repository Documentation

Main documentation files:

- `docs/setup.md` – environment setup, native build, and verification
- `docs/settings.md` – runtime configuration settings
- `docs/udp-json.md` – UDP packet schema used for Unity communication
- `docs/runtime-ui-and-terminal-reference.md` – runtime overlay and terminal text reference
- `docs/testing.md` – UDP validation and testing procedures

Contributors should review these files before making structural changes.

---

# 3. Code Style

The repository uses Ruff for Python linting and formatting.

Run before committing:

```bash
ruff check .
ruff format .
```

Optional automatic hook:

```bash
pre-commit install
```

Rust changes should also compile cleanly through the project build script:

```bash
bash build_native.sh
```

---

# 4. Native Module Workflow

The files below keep their Python filenames but are backed by the `xrdrone_native` extension after build:

- `id_flicker_mitigation.py`
- `output_formatter.py`
- `world_projection.py`
- `adaptive_tuning.py`
- `motion_smoothing.py`

Rebuild the native module whenever these inputs change:

- `src/lib.rs`
- `Cargo.toml`
- Rust/PyO3 build settings
- Python wrappers that expect newly added Rust symbols

Standard rebuild command:

```bash
source yolovenv/bin/activate
bash build_native.sh
```

Do not treat `target/` or other local build output as source files to commit.

---

# 5. Branch Workflow

Create a feature branch:

```bash
git checkout -b feature/your-feature
```

Make your changes, run checks, then commit:

```bash
git add .
git commit -m "Short descriptive message"
```

Push the branch:

```bash
git push origin feature/your-feature
```

Then open a Pull Request.

---

# 6. Testing Changes

Testing instructions are documented in `docs/testing.md`.

Typical validation workflow:

1. Build the native module (`bash build_native.sh`)
2. Run the pipeline (`python main.py`)
3. Verify detections and pose behavior
4. Confirm UDP packets match schema
5. Confirm Unity receives detections and pose data

Algorithm changes should include a reproducible demonstration.

---

# 7. UDP JSON Protocol Changes

Changes to the UDP schema must:

1. Be documented in `docs/udp-json.md`
2. Maintain compatibility with Unity when possible
3. Be validated using `test_with_coverage.py`

Unity components depend on this schema remaining stable.

---

# 8. Configuration Changes

Runtime behavior is controlled by `settings.py`.

Examples:

- detection thresholds
- smoothing parameters
- pose configuration
- adaptive tuning parameters

If new configuration options are added:

1. Add them to `settings.py`
2. Document them in `docs/settings.md`
3. Provide default values

If a change is implementation-only and preserves the current Python-facing settings contract, document the behavior where relevant without renaming existing settings unnecessarily.

---

# 9. Documentation Updates

Documentation must stay synchronized with implementation.

Update documentation when:

- pipeline behavior changes
- settings are added or removed
- runtime overlays or terminal outputs change
- UDP packet structure changes
- setup or build steps change
- native build requirements change

Relevant documentation:

- `docs/setup.md`
- `docs/settings.md`
- `docs/udp-json.md`
- `docs/runtime-ui-and-terminal-reference.md`
- `docs/testing.md`

---

# 10. Pull Request Requirements

Before submitting a PR ensure:

- Python code passes `ruff check`
- Python code is formatted with `ruff format`
- `bash build_native.sh` succeeds
- the pipeline runs without runtime errors
- documentation is updated if behavior changes
- no unnecessary files are committed

Do not commit:

- `.DS_Store`
- `__pycache__/`
- `target/`
- temporary media files
- local environment folders
- editor-specific local files

---

# 11. Commit Message Guidelines

Examples:

- `Add motion smoothing for ArUco pose updates`
- `Optimize UDP packet serialization`
- `Implement adaptive runtime tuning`
- `Port hot-path helper modules to Rust`
- `Update documentation for native build workflow`

---

# 12. Questions

If unsure about architecture decisions or major pipeline changes, open an issue before implementing large modifications.
