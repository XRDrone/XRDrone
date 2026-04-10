# Setup

## Build the native module

```bash
bash build_native.sh
python -c "import xrdrone_native; print('ok')"
```

## Notes

- `requirements.txt` includes the main project dependencies.
- After activation, use the virtual environment's Python for running the project and tests.
- `pre-commit` is optional. If you do not install the hook, Git still works normally and checks must be run manually.
- The active helper modules `id_flicker_mitigation.py` and `output_formatter.py` delegate hot-path work to `xrdrone_native`.
- The active native source is organized across `src/lib.rs`, `src/common.rs`, `src/geometry.rs`, `src/id_flicker.rs`, and `src/udp.rs`. Rebuild the extension after changes to any of those files.
