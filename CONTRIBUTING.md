# Contributing

Thank you for contributing to XRDrone. This document describes the expected workflow for development, testing, documentation updates, and pull requests.

XRDrone is a docs-first repository. The top-level README is a project summary, while detailed technical references live in `docs/`.

---

# 1. Development Environment

Set up the Python environment and build the Rust native module used by the mixed Python + Rust backend.

Recommended setup:

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

See `docs/setup.md` for environment and runtime setup details.

Each contributor must build the native module in their own clone and active virtual environment.

---

# 2. Repository Documentation

Read the relevant docs before making structural changes.

Main documentation files:

- `docs/setup.md` – environment setup and runtime setup
- `docs/settings.md` – runtime configuration settings
- `docs/udp-json.md` – UDP packet schema used for Unity communication
- `docs/runtime-ui-and-terminal-reference.md` – runtime overlay and terminal reference
- `docs/testing.md` – validation and testing procedures

If your change affects behavior, setup, schema, overlays, or testing flow, update the matching document in the same PR.

---

# 3. Code Style

The repository uses Ruff for Python linting and formatting.

Run before committing:

```bash
ruff check .
ruff format .
```

Optional pre-commit setup:

```bash
pre-commit install
```

Rust changes should compile cleanly through the project build script:

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

Do not commit `target/` or other local build artifacts.

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

Then open a pull request.

---

# 6. ORB-SLAM Middle-Man Workflow

The current backend architecture assumes a split pipeline:

1. A shared video source is exposed through MediaMTX
2. The XRDrone detection pipeline reads that stream
3. ORB-SLAM reads that same stream separately
4. ORB-SLAM sends pose packets over UDP to the backend middle-man
5. The backend fuses detection output with ORB-SLAM pose input
6. The fused UDP packet is sent to Unity

If you are changing the ORB-SLAM integration path, update all affected areas together:

- runtime settings in `settings.py`
- packet schema in `docs/udp-json.md`
- testing instructions in `docs/testing.md`
- runtime status or overlays in `docs/runtime-ui-and-terminal-reference.md`

Do not leave the implementation and docs out of sync.

---

# 7. Testing Changes

Testing instructions are documented in `docs/testing.md`.

Typical validation workflow:

1. Build the native module with `bash build_native.sh`
2. Run the pipeline with `python main.py`
3. Verify detection runtime behavior
4. Verify ORB-SLAM pose reception if fusion is enabled
5. Confirm UDP packets match the documented schema
6. Confirm Unity receives the expected fused output
7. Run `python test_with_coverage.py` when packet structure or transport behavior changes

Algorithm or architecture changes should include a reproducible demonstration, screenshot, or comparable artifact when practical.

---

# 8. UDP JSON Protocol Changes

Changes to the UDP schema must:

1. Be documented in `docs/udp-json.md`
2. Preserve Unity compatibility when possible
3. Be validated with `test_with_coverage.py`

Unity-facing packet changes are high impact and should not be merged without synchronized documentation.

---

# 9. Configuration Changes

Runtime behavior is controlled by `settings.py`.

Examples include:

- detection thresholds
- smoothing parameters
- UDP output settings
- MediaMTX input settings
- ORB-SLAM UDP receiver settings
- pose and projection configuration
- adaptive tuning parameters

If you add or modify a setting:

1. Update `settings.py`
2. Document it in `docs/settings.md`
3. Provide a reasonable default
4. Note any expected downstream impact on Unity, tests, or runtime setup

---

# 10. Documentation Updates

Documentation must stay synchronized with implementation.

Update docs when:

- pipeline behavior changes
- settings are added or removed
- runtime overlays or terminal outputs change
- UDP packet structure changes
- setup or build steps change
- MediaMTX or ORB-SLAM integration changes
- native build requirements change

Relevant documentation:

- `docs/setup.md`
- `docs/settings.md`
- `docs/udp-json.md`
- `docs/runtime-ui-and-terminal-reference.md`
- `docs/testing.md`

---

# 11. Pull Request Requirements

Before submitting a PR, ensure:

- Python code passes `ruff check`
- Python code is formatted with `ruff format`
- `bash build_native.sh` succeeds when relevant
- the pipeline runs without known runtime errors for the changed path
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

# 12. Commit Message Guidelines

Examples:

- `Set up ORB-SLAM middle-man UDP receiver`
- `Update Unity packet schema for fused ORB-SLAM data`
- `Add integration testing for ORB-SLAM fusion path`
- `Update documentation for MediaMTX and ORB-SLAM workflow`

---

# 13. Questions

If you are unsure about a major architecture decision, schema change, or workflow change, open an issue or start a design discussion before implementing a large change.
