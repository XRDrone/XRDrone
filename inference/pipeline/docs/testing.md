# testing.md

## UDP Validation and Testing

This document explains how to validate the XRDrone UDP JSON contract using `test_with_coverage.py`.

The goal of the test script is to verify that:

- the UDP packet schema matches the documented README contract
- UDP send/receive works correctly
- live packets from the running pipeline remain valid
- runtime packet statistics can be collected for analysis

This test file is a validation utility. It is not part of the live inference loop itself.

---

## Native build prerequisite

The pipeline now depends on the `xrdrone_native` extension for several hot-path helper modules. Build the native module in the active virtual environment before running any test mode that imports the pipeline:

```bash
bash build_native.sh
python -c "import xrdrone_native; print('ok')"
```

If the native build has not succeeded yet, `main.py`, `test_runner.py`, and `test_with_coverage.py` can fail during import or runtime because the Rust-backed helper modules are not available.

---

## Test Script

File:

```bash
python test_with_coverage.py
```

The script performs several types of validation depending on the command-line flags you use.

Core checks implemented by the script:

1. **Formatter/schema test**
   - Builds a sample UDP packet using the same formatter used by the pipeline.
   - Verifies that the packet contains the exact expected keys and valid field types.

2. **UDP loopback transport test**
   - Sends a real UDP packet through `UDPPublisher`.
   - Receives it back on localhost.
   - Parses it as JSON and validates the received packet again.

3. **Optional live packet test**
   - Listens on a UDP port for packets sent by a running `main.py` process.
   - Validates live runtime packets against the same UDP contract.

4. **Optional stats mode**
   - Collects packet and timing statistics from live UDP traffic.
   - Can also run the pipeline automatically on a video file and collect stats during the run.

---

## Default Validation

Command:

```bash
bash build_native.sh
python test_with_coverage.py
```

What it does:

1. Ensures the Rust-backed helper module is built.
2. Builds a sample UDP packet.
3. Validates that the packet structure matches the documented contract.
4. Sends the packet over UDP on localhost.
5. Receives the same packet back.
6. Validates the received packet again.

This confirms:

- the formatter output is valid
- the packet schema matches the documented structure
- UDP send/receive works on localhost

Expected success output:

```text
PASSED: UDP formatter structure and UDP send/receive are valid
```

Typical failure causes:

- the native module was not built in the active environment
- missing required fields
- extra unexpected fields
- wrong field types
- invalid JSON payload
- UDP loopback send/receive failure

---

## Live UDP Validation

Command:

```bash
python test_with_coverage.py --live
```

This mode validates real packets from the running pipeline.

What it does:

1. Runs the formatter/schema validation.
2. Runs the localhost UDP loopback test.
3. Opens a UDP listener on the configured host and port.
4. Waits for valid packets from a running pipeline process.
5. Validates each received packet against the UDP contract.

Expected success output:

```text
PASSED: live UDP transport and README packet structure are valid
```

Important:

- `--live` does **not** generate packets by itself.
- A separate pipeline process must already be running and sending UDP packets.
- That pipeline process must have been started in an environment where `xrdrone_native` is already built.

Typical usage:

Terminal 1:

```bash
python main.py
```

Terminal 2:

```bash
python test_with_coverage.py --live
```

Example with custom packet count and timeout:

```bash
python test_with_coverage.py --live --packets 5 --timeout 8
```

Common reasons this mode fails:

- `main.py` is not running
- UDP output is disabled
- the listener is bound to the wrong port
- another process is already using the port
- no valid packets arrive before timeout
- the native module was not built in the environment running the pipeline

---

## Stats Collection Mode

Command:

```bash
python test_with_coverage.py --stats
```

This mode collects runtime statistics from UDP packets sent by a running pipeline.

What it does:

1. Runs the formatter/schema validation.
2. Runs the localhost UDP loopback test.
3. Opens a UDP listener on the configured port.
4. Collects packet and timing statistics from incoming UDP packets.
5. Prints the collected stats to the terminal.

Expected success output:

```text
PASSED: UDP formatter structure, UDP send/receive, and stats collection are valid
```

Example:

```bash
python test_with_coverage.py --stats --packets 120 --timeout 8
```

Important:

- `--stats` does **not** generate packets by itself unless used with `--video`.
- A separate pipeline process must already be running and sending UDP packets.
- That process must already have the native build available.

This mode is useful for measuring:

- average packet size
- minimum and maximum packet size
- estimated source FPS
- estimated arrival FPS
- source timestamp jitter
- arrival timestamp jitter
- frame gaps
- duplicate frames
- out-of-order frames
- estimated person ID switches

### Stats Fields

The script prints a `STATS:` line with the following values:

- `valid_packets`: number of valid packets received
- `invalid_packets`: number of packets that failed validation
- `avg_packet_bytes`: average UDP packet size in bytes
- `min_packet_bytes`: smallest packet size seen
- `max_packet_bytes`: largest packet size seen
- `source_fps_est`: FPS estimated from packet timestamps generated by the source
- `arrival_fps_est`: FPS estimated from packet arrival times at the listener
- `source_dt_jitter_s`: variation in source timestamp spacing
- `arrival_dt_jitter_s`: variation in packet arrival spacing
- `frame_gaps`: missed frame IDs inferred from jumps in `frame_id`
- `duplicate_frames`: repeated frame IDs
- `out_of_order_frames`: frame IDs that arrive in reverse order
- `person_id_switches_est`: estimated person track-ID switches between nearby detections across frames

---

## Video-Based Stats Mode

Command:

```bash
python test_with_coverage.py --video "/path/to/video.mp4" --stats
```

This mode launches the pipeline automatically on a video file and collects UDP stats during the run.

What it does:

1. Runs the formatter/schema validation.
2. Runs the localhost UDP loopback test.
3. Starts the pipeline on the specified video file.
4. Collects UDP packet statistics while the video is processed.
5. Prints the collected stats after the run finishes.

Expected success output:

```text
PASSED: UDP formatter structure, UDP send/receive, and video stats collection are valid
```

Example:

```bash
python test_with_coverage.py --video "/path/to/video.mp4" --stats
```

Notes:

- `--video` requires `--stats`
- when `--video` is used, the script processes the full video automatically
- a progress bar is shown while packets are collected
- at least one valid UDP packet must be received during the run
- the native module must already be built in the environment used by the spawned pipeline process

Internally, this mode temporarily switches the pipeline input mode to file-based input, sets the target video path, enables UDP output, runs the pipeline, then restores the previous settings.

---

## Optional Arguments

The script supports these commonly used arguments:

```bash
--live
--stats
--video "/path/to/video.mp4"
--host 0.0.0.0
--port <udp_port>
--packets <count>
--timeout <seconds>
```

### Argument Details

- `--live`  
  Validate packets from a running `main.py` process.

- `--stats`  
  Collect packet/runtime statistics.

- `--video`  
  Launch the pipeline on the provided video file and collect stats. Must be used with `--stats`.

- `--host`  
  Host or interface to bind for live UDP listening. Default is `0.0.0.0`.

- `--port`  
  UDP port to bind for live listening. Defaults to `settings.UDP_PORT`.

- `--packets`  
  Number of valid packets required for `--live` or `--stats`. Ignored for `--video` mode.

- `--timeout`  
  Socket timeout in seconds.

---

## Exact Contract Validation

The test script validates the UDP packet with exact key matching.

### Required top-level keys

- `frame_id`
- `timestamp`
- `width`
- `height`
- `detections`
- `pose`

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
- `world_valid`
- `world_x`
- `world_y`
- `world_z`

### Required pose keys

The pose object must contain exactly:

- `x`
- `altitude`
- `z`
- `yaw`
- `pitch`
- `roll`
- `hfov`
- `markers_used`
- `pose_valid`

### Additional validation rules

The script also checks that:

- `frame_id`, `width`, `height`, and `markers_used` are integers
- `timestamp` and numeric pose fields are numeric
- `detections` is a list
- `pose` is an object
- `world_valid` and `pose_valid` are booleans
- `conf` is in the range `[0.0, 1.0]`
- `cx`, `cy`, `w`, `h`, `foot_x`, and `foot_y` are all in `[0.0, 1.0]`
- `width` and `height` are positive
- `markers_used` is non-negative
- no extra keys are present in the packet, detection objects, or pose object

The test fails if any required field is missing, any unexpected field is present, or any field has an invalid type or range.

---

## Sample Test Workflow

### Basic contract + UDP transport

```bash
bash build_native.sh
python test_with_coverage.py
```

### Validate live packets from a running pipeline

```bash
bash build_native.sh
python main.py
python test_with_coverage.py --live --packets 5 --timeout 8
```

### Collect stats from a running pipeline

```bash
bash build_native.sh
python main.py
python test_with_coverage.py --stats --packets 120 --timeout 8
```

### Run a video through the pipeline and collect stats automatically

```bash
bash build_native.sh
python test_with_coverage.py --video "/path/to/video.mp4" --stats
```

---

## Failure Output

The script prints exactly one failure line and exits with code `1` when validation fails.

Failure messages vary by mode:

- default mode: formatter or loopback failure
- `--live`: live UDP or packet-contract failure
- `--stats`: stats collection or packet-contract failure
- `--video --stats`: video stats collection or packet-contract failure

This makes the script suitable for quick local checks and for automated validation during development.

---

## Summary

Use `test_with_coverage.py` to validate that the XRDrone pipeline is producing UDP packets that match the documented packet contract.

Recommended usage:

- build `xrdrone_native` first
- use the default mode for fast formatter and UDP transport checks
- use `--live` to verify real packets from `main.py`
- use `--stats` to measure packet behavior during runtime
- use `--video --stats` to benchmark a full video run automatically
