# XRDrone UDP Pipeline Documentation

This repository keeps the pipeline documentation split into focused Markdown files inside `docs/`. Use those files as the main reference for setup, runtime behavior, packet structure, settings, and runtime output.

## Start Here

The top-level README is intentionally brief. Detailed documentation lives in `docs/` so setup, runtime behavior, and protocol details stay organized and easier to maintain.

## Native Module Build

The pipeline is now a mixed Python + Rust project.

The application entrypoints and orchestration are still Python:

- `main.py`
- `live_runner.py`
- `test_runner.py`
- `pose_estimator.py`
- rendering, capture, and UDP orchestration

Several hot-path helper modules keep their original Python filenames but are backed by the native `xrdrone_native` extension after build:

- `id_flicker_mitigation.py`
- `output_formatter.py`
- `world_projection.py`
- `adaptive_tuning.py`
- `motion_smoothing.py`

The Rust implementation is now split into focused source files under `src/` instead of keeping everything in one monolithic `lib.rs`:

- `src/lib.rs`: Python module registration only
- `src/common.rs`: shared Python interop helpers, clamps, parsing, and history utilities
- `src/geometry.rs`: matrix, quaternion, projection, and filter math helpers
- `src/id_flicker.rs`: tracked-object continuity and coasting
- `src/world_projection.rs`: foot-point extraction and world-ground projection
- `src/udp.rs`: Unity UDP packet formatting
- `src/adaptive_tuning.rs`: bounded runtime tuning controller
- `src/smoothing.rs`: One Euro filters plus pose and world-track smoothing

Before running `main.py` or `test_with_coverage.py`, build the native module in your active virtual environment:

```bash
bash build_native.sh
python -c "import xrdrone_native; print('ok')"
```

If any file in `src/` or `Cargo.toml` changes, rebuild the extension.

## Documentation Files

- `docs/setup.md`  
  Environment setup, Python dependencies, Rust toolchain notes, native build, CUDA usage, and basic verification.

- `docs/settings.md`  
  Explanation of the configurable values in `settings.py`, including input selection, UDP output, pose options, smoothing, adaptive tuning, overlays, and keybinds.

- `docs/udp-json.md`  
  The UDP JSON contract, including top-level packet fields, detection fields, pose fields, and field meanings.

- `docs/runtime-ui-and-terminal-reference.md`  
  Reference for runtime text shown on the video output and messages printed to the terminal during normal pipeline execution. This includes items such as pose-mode text, hold states, adaptive tuning log lines, toggle messages, and other runtime status output.

- `docs/testing.md`  
  How to validate the UDP contract and transport behavior, including native-build prerequisites, available test modes, expected pass/fail output, and packet/statistics checks.

## Suggested Reading Order

1. `docs/setup.md`
2. `docs/settings.md`
3. `docs/udp-json.md`
4. `docs/runtime-ui-and-terminal-reference.md`
5. `docs/testing.md`

## Notes

- Use the files in `docs/` as the main source of truth for pipeline documentation.
- Keep detailed operational and protocol documentation there instead of expanding this top-level README.
- Add new documentation files to `docs/` so related information remains grouped together.
- The UDP schema is unchanged by the Rust port. The native module accelerates selected helper stages while preserving the existing Python-facing module names and packet contract.
