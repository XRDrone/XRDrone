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
- rendering, capture, and UDP orchestration

The active native-backed helper modules used by the detection branch are:

- `id_flicker_mitigation.py`
- `output_formatter.py`

The Rust implementation is organized across focused source files under `src/`:

- `src/lib.rs`: Python module registration only
- `src/common.rs`: shared Python interop helpers, clamps, parsing, and history utilities
- `src/geometry.rs`: geometry and normalization helpers
- `src/id_flicker.rs`: tracked-object continuity and coasting
- `src/udp.rs`: Unity UDP packet formatting

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
  Explanation of the configurable values in `settings.py`, including input selection, UDP output, ID continuity tuning, overlays, and keybinds.

- `docs/udp-json.md`
  The UDP JSON contract for the human-detection branch.

- `docs/runtime-ui-and-terminal-reference.md`
  Reference for runtime text shown on the video output and messages printed to the terminal during normal pipeline execution.

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
