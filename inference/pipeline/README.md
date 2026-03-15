# XRDrone UDP Pipeline Documentation

This repository includes multiple documentation files. To understand setup, runtime behavior, packet structure, and testing, start by exploring the `docs/` folder.

## Start Here

The top-level README is intentionally brief. The main documentation is split into focused files inside `docs/`.

## Documentation Files

- `docs/setup.md`  
  How to create a virtual environment, install dependencies, enable CUDA, and verify CUDA.

- `docs/udp-json.md`  
  Defines the UDP JSON packet structure, including top-level fields, detection fields, pose fields, and field meanings.

- `docs/testing.md`  
  Explains how to run the validation script, what each test mode checks, and what pass/fail output means.

- `docs/settings.md`  
  Describes the runtime configuration options in `settings.py` in simple terms.

## Suggested Reading Order

1. `docs/setup.md`
2. `docs/settings.md`
3. `docs/udp-json.md`
4. `docs/testing.md`

## Notes

- Use the documentation in `docs/` as the main reference for setup and runtime behavior.
- Keep the detailed documentation there instead of expanding this top-level README.
- When adding new documentation, place it in `docs/` so related information stays organized.
