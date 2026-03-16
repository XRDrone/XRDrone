# XRDrone UDP Pipeline Documentation

This repository keeps the pipeline documentation split into focused Markdown files inside `docs/`. Use those files as the main reference for setup, runtime behavior, packet structure, settings, and runtime output.

## Start Here

The top-level README is intentionally brief. Detailed documentation lives in `docs/` so setup, runtime behavior, and protocol details stay organized and easier to maintain.

## Documentation Files

- `docs/setup.md`  
  Environment setup, dependency installation, CUDA usage, and basic verification.

- `docs/settings.md`  
  Explanation of the configurable values in `settings.py`, including input selection, UDP output, pose options, smoothing, adaptive tuning, overlays, and keybinds.

- `docs/udp-json.md`  
  The UDP JSON contract, including top-level packet fields, detection fields, pose fields, and field meanings.

- `docs/runtime-ui-and-terminal-reference.md`  
  Reference for runtime text shown on the video output and messages printed to the terminal during normal pipeline execution. This includes items such as pose-mode text, hold states, adaptive tuning log lines, toggle messages, and other runtime status output.

- `docs/testing.md`  
  How to validate the UDP contract and transport behavior, including available test modes, expected pass/fail output, and packet/statistics checks.

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
