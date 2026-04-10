# Testing

## UDP contract validation

`test_with_coverage.py` validates the human-detection UDP contract and the UDP transport path.

### Required top-level keys

Each packet must contain exactly:

- `frame_id`
- `timestamp`
- `width`
- `height`
- `detections`

### Required detection keys

Each detection object must contain exactly:

- `id`
- `cls`
- `conf`
- `cx`
- `cy`
- `w`
- `h`
- `foot_x`
- `foot_y`

### Additional validation rules

The script also checks that:

- `frame_id`, `width`, and `height` are integers
- `timestamp` is numeric
- `detections` is a list
- `conf` is in the range `[0.0, 1.0]`
- `cx`, `cy`, `w`, `h`, `foot_x`, and `foot_y` are all in `[0.0, 1.0]`
- `width` and `height` are positive
- no extra keys are present in the packet or detection objects

## Sample workflow

```bash
bash build_native.sh
python test_with_coverage.py
```

To validate packets from a running pipeline:

```bash
python test_with_coverage.py --live --packets 5 --timeout 8
```
