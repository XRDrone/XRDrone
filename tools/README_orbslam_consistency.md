# ORB-SLAM3 vs FlightRecord Consistency Check

This script checks whether ORB-SLAM3 motion and orientation are **consistent** with DJI flight telemetry.

It does **not** treat the flight record as absolute coordinate ground truth.

## Requirements

Install Python packages:

```bash
python -m pip install numpy pandas
```

If your system uses `python3` instead:

```bash
python3 -m pip install numpy pandas
```

## Input files

### 1. ORB-SLAM3 pose text
Expected line format:

```text
POSE frame=2 t=1403636858.7517 x=0.023807 y=-0.013339 z=-0.007875 qx=-0.001431 qy=0.004988 qz=0.000461 qw=0.999986
```

### 2. DJI FlightRecord CSV
Use the exported FlightRecord CSV format that begins with `sep=,`.

## How to run

From the folder containing the script:

```bash
python orbslam_flight_consistency.py \
  --orbslam orbslam3_cord.txt \
  --flight "FlightRecord_2025-11-25_[11-05-08].csv" \
  --outdir consistency_report
```

If you are one folder deeper than the script, use relative paths such as:

```bash
python ../orbslam_flight_consistency.py \
  --orbslam ../orbslam3_cord.txt \
  --flight ../FlightRecord_2025-11-25_[11-05-08].csv \
  --outdir ../consistency_report
```

## What the script outputs

It prints a console summary and saves:

- `summary.json` — all metrics and the overall score
- `aligned_samples.csv` — ORB and flight samples aligned by **relative time in seconds**

## How alignment works

The script does **not** align by frame number.

It converts both logs to **relative elapsed time**:

- ORB-SLAM3 uses `t=...`
- FlightRecord uses `OSD.flyTime [s]`

Then it matches each ORB sample to the nearest flight sample in time.

## How to interpret the report

### Overall consistency score
A quick summary from **0 to 100**.

- **80–100**: strong
- **60–80**: moderate-to-strong
- **40–60**: weak-to-moderate
- **20–40**: weak
- **0–20**: very weak

Use this as a summary only. The category metrics matter more.

### Orientation correlations
These compare ORB quaternion-derived attitude against flight telemetry attitude:

- yaw
- pitch
- roll

Interpretation:

- **1.0** = very strong match
- **0.0** = weak / no useful match

### Rate correlations
These compare how fast yaw, pitch, and roll change.

This is stricter than raw orientation. A system can have decent yaw correlation but poor yaw-rate correlation.

### Speed correlation
Compares ORB frame-to-frame displacement against drone speed.

High means ORB motion magnitude rises and falls with drone speed.

### Vertical trend
Finds the ORB axis that best matches flight height data.

This is often one of the strongest checks for physical consistency.

### State agreement
Compares hover vs movement periods.

- **1.0** = perfect agreement
- **0.5** = about half the time
- **0.0** = no agreement

### Timing deltas
Compares event timing in seconds.

Lower is better.

Examples:

- turns
- climb starts
- descent starts
- takeoff-like
- landing-like

If `count=0`, the script did not find enough comparable events to score that category.

## What `mode` means

For many comparisons, the script tests both:

- `same_sign`
- `flipped_sign`

`same_sign` means the signals matched best as-is.

`flipped_sign` means the signals matched better after multiplying one by `-1`.
This usually means the systems use opposite axis sign conventions, not necessarily that the result is bad.

## What you can claim

You can say the script checks whether ORB-SLAM3 is **physically consistent** with the flight telemetry.

You should **not** say it proves that ORB-SLAM3 absolute coordinates are true world coordinates.
