#!/usr/bin/env python3
"""
Compare ORB-SLAM3 pose output against a DJI FlightRecord CSV for consistency.

This does NOT treat telemetry as absolute position ground truth.
Instead, it checks whether ORB-SLAM3 motion/orientation trends are physically
consistent with the drone telemetry.

Inputs
------
1) ORB-SLAM3 pose text file with lines like:
   POSE frame=2 t=1403636858.7517 x=0.023807 y=-0.013339 z=-0.007875 \
        qx=-0.001431 qy=0.004988 qz=0.000461 qw=0.999986

2) DJI FlightRecord CSV (the export format that begins with "sep=,").

Outputs
-------
- Console summary
- JSON summary report
- CSV of aligned comparison samples

Example
-------
python orbslam_flight_consistency.py \
  --orbslam orbslam3_cord.txt \
  --flight "FlightRecord_2025-11-25_[11-05-08].csv" \
  --outdir consistency_report
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

POSE_RE = re.compile(
    r"POSE\s+frame=(?P<frame>\d+)\s+"
    r"t=(?P<t>-?\d+(?:\.\d+)?)\s+"
    r"x=(?P<x>-?\d+(?:\.\d+)?)\s+"
    r"y=(?P<y>-?\d+(?:\.\d+)?)\s+"
    r"z=(?P<z>-?\d+(?:\.\d+)?)\s+"
    r"qx=(?P<qx>-?\d+(?:\.\d+)?)\s+"
    r"qy=(?P<qy>-?\d+(?:\.\d+)?)\s+"
    r"qz=(?P<qz>-?\d+(?:\.\d+)?)\s+"
    r"qw=(?P<qw>-?\d+(?:\.\d+)?)"
)


@dataclass
class EventSet:
    turns: list[float]
    climb_starts: list[float]
    descent_starts: list[float]
    takeoff_like: list[float]
    landing_like: list[float]
    hover_spans: list[tuple[float, float]]
    motion_spans: list[tuple[float, float]]


# -----------------------------
# Parsing helpers
# -----------------------------


def load_orbslam(path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            match = POSE_RE.search(line)
            if not match:
                continue
            row = {k: float(v) for k, v in match.groupdict().items()}
            row["frame"] = int(row["frame"])
            rows.append(row)

    if not rows:
        raise ValueError("No ORB-SLAM3 pose lines were parsed.")

    df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    df["t_rel"] = df["t"] - df["t"].iloc[0]

    # Translation derivatives.
    for axis in ["x", "y", "z"]:
        df[f"d{axis}"] = df[axis].diff()
    dt = df["t"].diff().replace(0, np.nan)
    df["disp"] = np.sqrt(df[["dx", "dy", "dz"]].pow(2).sum(axis=1))
    df["slam_speed"] = df["disp"] / dt

    # Quaternion -> Euler (degrees).
    yaw, pitch, roll = quaternion_to_euler_deg(
        df["qx"].to_numpy(),
        df["qy"].to_numpy(),
        df["qz"].to_numpy(),
        df["qw"].to_numpy(),
    )
    df["yaw_deg"] = unwrap_degrees(yaw)
    df["pitch_deg"] = pitch
    df["roll_deg"] = roll

    df["yaw_rate_deg_s"] = derivative(df["yaw_deg"].to_numpy(), df["t"].to_numpy())
    df["pitch_rate_deg_s"] = derivative(df["pitch_deg"].to_numpy(), df["t"].to_numpy())
    df["roll_rate_deg_s"] = derivative(df["roll_deg"].to_numpy(), df["t"].to_numpy())
    return df


def load_flightrecord(path: str | Path) -> pd.DataFrame:
    # DJI export contains a first line: sep=,
    df = pd.read_csv(path, skiprows=1)

    if "OSD.flyTime [s]" not in df.columns:
        raise ValueError("DJI FlightRecord CSV missing 'OSD.flyTime [s]' column.")

    df = df.copy()
    numeric_candidates = [
        "OSD.flyTime [s]",
        "OSD.height [ft]",
        "OSD.vpsHeight [ft]",
        "OSD.altitude [ft]",
        "OSD.hSpeed [MPH]",
        "OSD.xSpeed [MPH]",
        "OSD.ySpeed [MPH]",
        "OSD.zSpeed [MPH]",
        "OSD.pitch",
        "OSD.roll",
        "OSD.yaw",
        "OSD.yaw [360]",
        "GIMBAL.pitch",
        "GIMBAL.roll",
        "GIMBAL.yaw",
        "GIMBAL.yaw [360]",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("OSD.flyTime [s]").reset_index(drop=True)
    df["t_rel"] = df["OSD.flyTime [s]"] - df["OSD.flyTime [s]"].iloc[0]

    # Prefer explicit yaw if available, otherwise 360 version.
    if "OSD.yaw" in df.columns and df["OSD.yaw"].notna().any():
        df["yaw_deg"] = unwrap_degrees(df["OSD.yaw"].to_numpy())
    elif "OSD.yaw [360]" in df.columns:
        df["yaw_deg"] = unwrap_degrees(df["OSD.yaw [360]"].to_numpy())
    else:
        df["yaw_deg"] = np.nan

    if "OSD.pitch" in df.columns:
        df["pitch_deg"] = df["OSD.pitch"]
    else:
        df["pitch_deg"] = np.nan

    if "OSD.roll" in df.columns:
        df["roll_deg"] = df["OSD.roll"]
    else:
        df["roll_deg"] = np.nan

    if "OSD.hSpeed [MPH]" in df.columns:
        df["drone_speed_mps"] = mph_to_mps(df["OSD.hSpeed [MPH]"])
    else:
        df["drone_speed_mps"] = np.nan

    # Vertical references in meters for trend checks.
    for src, dst in [
        ("OSD.height [ft]", "height_m"),
        ("OSD.vpsHeight [ft]", "vps_height_m"),
        ("OSD.altitude [ft]", "altitude_m"),
    ]:
        if src in df.columns:
            df[dst] = ft_to_m(df[src])
        else:
            df[dst] = np.nan

    df["yaw_rate_deg_s"] = derivative(df["yaw_deg"].to_numpy(), df["OSD.flyTime [s]"].to_numpy())
    df["pitch_rate_deg_s"] = derivative(
        df["pitch_deg"].to_numpy(), df["OSD.flyTime [s]"].to_numpy()
    )
    df["roll_rate_deg_s"] = derivative(df["roll_deg"].to_numpy(), df["OSD.flyTime [s]"].to_numpy())
    return df


# -----------------------------
# Math helpers
# -----------------------------


def quaternion_to_euler_deg(
    qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, qw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))

    # pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))
    return yaw, pitch, roll


def unwrap_degrees(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    radians = np.deg2rad(arr)
    return np.rad2deg(np.unwrap(radians))


def derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    if len(values) < 2:
        return out
    dt = np.diff(times)
    dv = np.diff(values)
    safe = np.where(dt == 0, np.nan, dt)
    out[1:] = dv / safe
    return out


def safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return None
    a2 = a[mask]
    b2 = b[mask]
    if np.nanstd(a2) == 0 or np.nanstd(b2) == 0:
        return None
    return float(np.corrcoef(a2, b2)[0, 1])


def zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std == 0:
        return np.full_like(arr, np.nan)
    return (arr - mean) / std


def ft_to_m(series: pd.Series) -> pd.Series:
    return series * 0.3048


def mph_to_mps(series: pd.Series) -> pd.Series:
    return series * 0.44704


def choose_vertical_axis(
    orb: pd.DataFrame, flight: pd.DataFrame
) -> tuple[str | None, dict[str, float | None]]:
    preferred_ref = None
    for ref in ["vps_height_m", "height_m", "altitude_m"]:
        if ref in flight.columns and flight[ref].notna().sum() > 3:
            preferred_ref = ref
            break
    if preferred_ref is None:
        return None, {}

    merged = align_timeseries(
        orb[["t_rel", "x", "y", "z"]],
        flight[["t_rel", preferred_ref]],
        time_col="t_rel",
    )
    if merged.empty:
        return None, {}

    scores: dict[str, float | None] = {}
    for axis in ["x", "y", "z"]:
        corr = safe_corr(merged[axis].to_numpy(), merged[preferred_ref].to_numpy())
        if corr is not None:
            scores[axis] = max(corr, -corr)
        else:
            scores[axis] = None

    valid = {k: v for k, v in scores.items() if v is not None}
    if not valid:
        return None, scores
    best_axis = max(valid, key=valid.get)
    return best_axis, scores


# -----------------------------
# Alignment and event detection
# -----------------------------


def align_timeseries(
    left: pd.DataFrame, right: pd.DataFrame, time_col: str = "t_rel"
) -> pd.DataFrame:
    left = left.sort_values(time_col).reset_index(drop=True)
    right = right.sort_values(time_col).reset_index(drop=True)
    return pd.merge_asof(left, right, on=time_col, direction="nearest")


def detect_spans(
    mask: np.ndarray, times: np.ndarray, min_duration_s: float = 0.5
) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = times[i]
        elif not value and start is not None:
            end = times[i - 1]
            if end - start >= min_duration_s:
                spans.append((float(start), float(end)))
            start = None
    if start is not None:
        end = times[-1]
        if end - start >= min_duration_s:
            spans.append((float(start), float(end)))
    return spans


def detect_events(df: pd.DataFrame, source: str, vertical_col: str | None = None) -> EventSet:
    if source == "slam":
        t = df["t_rel"].to_numpy()
        speed = df["slam_speed"].to_numpy()
        yaw_rate = np.abs(df["yaw_rate_deg_s"].to_numpy())
        vertical_rate = (
            derivative(df[vertical_col].to_numpy(), df["t"].to_numpy())
            if vertical_col
            else np.full(len(df), np.nan)
        )
    else:
        t = df["t_rel"].to_numpy()
        speed = df["drone_speed_mps"].to_numpy()
        yaw_rate = np.abs(df["yaw_rate_deg_s"].to_numpy())
        candidate = None
        for col in ["vps_height_m", "height_m", "altitude_m"]:
            if col in df.columns and df[col].notna().sum() > 3:
                candidate = col
                break
        vertical_rate = (
            derivative(df[candidate].to_numpy(), df["OSD.flyTime [s]"].to_numpy())
            if candidate
            else np.full(len(df), np.nan)
        )

    moving_mask = np.nan_to_num(speed, nan=0.0) > 0.35
    hover_mask = np.nan_to_num(speed, nan=0.0) < 0.15
    turn_mask = np.nan_to_num(yaw_rate, nan=0.0) > 12.0
    climb_mask = np.nan_to_num(vertical_rate, nan=0.0) > 0.20
    descent_mask = np.nan_to_num(vertical_rate, nan=0.0) < -0.20

    motion_spans = detect_spans(moving_mask, t)
    hover_spans = detect_spans(hover_mask, t)
    turn_spans = detect_spans(turn_mask, t, min_duration_s=0.25)
    climb_spans = detect_spans(climb_mask, t, min_duration_s=0.25)
    descent_spans = detect_spans(descent_mask, t, min_duration_s=0.25)

    takeoff_like = []
    landing_like = []
    if motion_spans:
        takeoff_like.append(motion_spans[0][0])
        landing_like.append(motion_spans[-1][1])

    return EventSet(
        turns=[span[0] for span in turn_spans],
        climb_starts=[span[0] for span in climb_spans],
        descent_starts=[span[0] for span in descent_spans],
        takeoff_like=takeoff_like,
        landing_like=landing_like,
        hover_spans=hover_spans,
        motion_spans=motion_spans,
    )


def nearest_event_deltas(reference_times: list[float], candidate_times: list[float]) -> list[float]:
    if not reference_times or not candidate_times:
        return []
    out = []
    for t in reference_times:
        out.append(float(min(abs(t - c) for c in candidate_times)))
    return out


# -----------------------------
# Main comparison logic
# -----------------------------


def compare(orb: pd.DataFrame, flight: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    vertical_axis, vertical_scores = choose_vertical_axis(orb, flight)

    merged = align_timeseries(
        orb[
            [
                "t_rel",
                "yaw_deg",
                "pitch_deg",
                "roll_deg",
                "yaw_rate_deg_s",
                "pitch_rate_deg_s",
                "roll_rate_deg_s",
                "slam_speed",
                "x",
                "y",
                "z",
            ]
        ],
        flight[
            [
                "t_rel",
                "yaw_deg",
                "pitch_deg",
                "roll_deg",
                "yaw_rate_deg_s",
                "pitch_rate_deg_s",
                "roll_rate_deg_s",
                "drone_speed_mps",
                "height_m",
                "vps_height_m",
                "altitude_m",
            ]
        ],
        time_col="t_rel",
    )

    merged = merged.rename(
        columns={
            "yaw_deg_x": "slam_yaw_deg",
            "pitch_deg_x": "slam_pitch_deg",
            "roll_deg_x": "slam_roll_deg",
            "yaw_rate_deg_s_x": "slam_yaw_rate_deg_s",
            "pitch_rate_deg_s_x": "slam_pitch_rate_deg_s",
            "roll_rate_deg_s_x": "slam_roll_rate_deg_s",
            "yaw_deg_y": "drone_yaw_deg",
            "pitch_deg_y": "drone_pitch_deg",
            "roll_deg_y": "drone_roll_deg",
            "yaw_rate_deg_s_y": "drone_yaw_rate_deg_s",
            "pitch_rate_deg_s_y": "drone_pitch_rate_deg_s",
            "roll_rate_deg_s_y": "drone_roll_rate_deg_s",
        }
    )

    metrics: dict[str, Any] = {}
    metrics["orientation_correlation"] = {
        "yaw": best_signed_correlation(
            merged["slam_yaw_deg"].to_numpy(), merged["drone_yaw_deg"].to_numpy()
        ),
        "pitch": best_signed_correlation(
            merged["slam_pitch_deg"].to_numpy(), merged["drone_pitch_deg"].to_numpy()
        ),
        "roll": best_signed_correlation(
            merged["slam_roll_deg"].to_numpy(), merged["drone_roll_deg"].to_numpy()
        ),
    }

    metrics["rate_correlation"] = {
        "yaw_rate": best_signed_correlation(
            merged["slam_yaw_rate_deg_s"].to_numpy(), merged["drone_yaw_rate_deg_s"].to_numpy()
        ),
        "pitch_rate": best_signed_correlation(
            merged["slam_pitch_rate_deg_s"].to_numpy(), merged["drone_pitch_rate_deg_s"].to_numpy()
        ),
        "roll_rate": best_signed_correlation(
            merged["slam_roll_rate_deg_s"].to_numpy(), merged["drone_roll_rate_deg_s"].to_numpy()
        ),
    }

    metrics["speed_correlation"] = best_signed_correlation(
        merged["slam_speed"].to_numpy(), merged["drone_speed_mps"].to_numpy()
    )

    if vertical_axis is not None:
        # Compare against the most populated telemetry height reference.
        ref = next(
            col
            for col in ["vps_height_m", "height_m", "altitude_m"]
            if merged[col].notna().sum() > 3
        )
        metrics["vertical_axis"] = {
            "chosen_orbslam_axis": vertical_axis,
            "axis_score_abs_corr": vertical_scores,
            "reference_used": ref,
            "signed_correlation": best_signed_correlation(
                merged[vertical_axis].to_numpy(), merged[ref].to_numpy()
            ),
        }
    else:
        metrics["vertical_axis"] = None

    # Hover vs movement agreement.
    slam_hover = np.nan_to_num(merged["slam_speed"].to_numpy(), nan=0.0) < 0.08
    drone_hover = np.nan_to_num(merged["drone_speed_mps"].to_numpy(), nan=0.0) < 0.15
    slam_move = np.nan_to_num(merged["slam_speed"].to_numpy(), nan=0.0) > 0.20
    drone_move = np.nan_to_num(merged["drone_speed_mps"].to_numpy(), nan=0.0) > 0.35
    metrics["state_agreement"] = {
        "hover_match_fraction": fraction_match(slam_hover, drone_hover),
        "movement_match_fraction": fraction_match(slam_move, drone_move),
    }

    slam_events = detect_events(orb, source="slam", vertical_col=vertical_axis)
    drone_events = detect_events(flight, source="flight")

    metrics["timing_deltas_seconds"] = {
        "turns": summarize_event_deltas(
            nearest_event_deltas(drone_events.turns, slam_events.turns)
        ),
        "climb_starts": summarize_event_deltas(
            nearest_event_deltas(drone_events.climb_starts, slam_events.climb_starts)
        ),
        "descent_starts": summarize_event_deltas(
            nearest_event_deltas(drone_events.descent_starts, slam_events.descent_starts)
        ),
        "takeoff_like": summarize_event_deltas(
            nearest_event_deltas(drone_events.takeoff_like, slam_events.takeoff_like)
        ),
        "landing_like": summarize_event_deltas(
            nearest_event_deltas(drone_events.landing_like, slam_events.landing_like)
        ),
    }

    metrics["sample_counts"] = {
        "orbslam_rows": int(len(orb)),
        "flight_rows": int(len(flight)),
        "aligned_rows": int(len(merged)),
    }

    metrics["consistency_score"] = compute_consistency_score(metrics)

    metrics["notes"] = [
        "Flight telemetry is used here as a consistency reference, "
        "not absolute position ground truth.",
        "Raw ORB-SLAM3 x/y/z should not be compared numerically to drone absolute coordinates.",
        "Correlation sign may flip because axis directions may differ between frames.",
    ]

    # Add z-scored views for plotting or inspection.
    merged["slam_speed_z"] = zscore(merged["slam_speed"].to_numpy())
    merged["drone_speed_z"] = zscore(merged["drone_speed_mps"].to_numpy())
    if vertical_axis is not None:
        merged[f"slam_{vertical_axis}_z"] = zscore(merged[vertical_axis].to_numpy())

    return metrics, merged


def best_signed_correlation(a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    corr_same = safe_corr(a, b)
    corr_flipped = safe_corr(a, -b)
    candidates = {
        "same_sign": corr_same,
        "flipped_sign": corr_flipped,
    }
    valid = {k: v for k, v in candidates.items() if v is not None}
    if not valid:
        return {"best": None, "mode": None, "same_sign": corr_same, "flipped_sign": corr_flipped}
    mode = max(valid, key=valid.get)
    return {
        "best": valid[mode],
        "mode": mode,
        "same_sign": corr_same,
        "flipped_sign": corr_flipped,
    }


def fraction_match(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return None
    return float(np.mean(a[mask] == b[mask]))


def summarize_event_deltas(deltas: list[float]) -> dict[str, float | int | None]:
    if not deltas:
        return {"count": 0, "mean": None, "median": None, "max": None}
    arr = np.asarray(deltas, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def score_correlation(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(np.clip(value, 0.0, 1.0))


def score_fraction(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(np.clip(value, 0.0, 1.0))


def score_timing(stats: dict[str, float | int | None], cutoff_s: float = 30.0) -> float | None:
    mean = stats.get("mean")
    count = stats.get("count", 0)
    if not count or mean is None or not np.isfinite(mean):
        return None
    return float(np.clip(1.0 - (float(mean) / cutoff_s), 0.0, 1.0))


def compute_consistency_score(metrics: dict[str, Any]) -> dict[str, Any]:
    components: list[tuple[str, float, float]] = []

    for axis, weight in [("yaw", 0.16), ("pitch", 0.10), ("roll", 0.10)]:
        value = score_correlation(metrics["orientation_correlation"][axis]["best"])
        if value is not None:
            components.append((f"orientation_{axis}", value, weight))

    for axis, weight in [("yaw_rate", 0.06), ("pitch_rate", 0.04), ("roll_rate", 0.04)]:
        value = score_correlation(metrics["rate_correlation"][axis]["best"])
        if value is not None:
            components.append((axis, value, weight))

    speed_value = score_correlation(metrics["speed_correlation"]["best"])
    if speed_value is not None:
        components.append(("speed", speed_value, 0.14))

    vertical = metrics.get("vertical_axis")
    if vertical and vertical.get("signed_correlation"):
        vertical_value = score_correlation(vertical["signed_correlation"]["best"])
        if vertical_value is not None:
            components.append(("vertical", vertical_value, 0.18))

    for name, weight in [("hover_match_fraction", 0.05), ("movement_match_fraction", 0.05)]:
        value = score_fraction(metrics["state_agreement"][name])
        if value is not None:
            components.append((name, value, weight))

    for name, weight in [
        ("turns", 0.02),
        ("climb_starts", 0.02),
        ("descent_starts", 0.02),
        ("takeoff_like", 0.03),
        ("landing_like", 0.03),
    ]:
        value = score_timing(metrics["timing_deltas_seconds"][name])
        if value is not None:
            components.append((name, value, weight))

    if not components:
        return {
            "score_0_to_100": None,
            "score_0_to_1": None,
            "grade": None,
            "components_used": [],
            "note": "No scoreable metrics were available.",
        }

    total_weight = sum(weight for _, _, weight in components)
    weighted = sum(value * weight for _, value, weight in components) / total_weight
    score_0_to_100 = round(weighted * 100.0, 1)

    if score_0_to_100 >= 85:
        grade = "very strong"
    elif score_0_to_100 >= 70:
        grade = "strong"
    elif score_0_to_100 >= 55:
        grade = "moderate"
    elif score_0_to_100 >= 40:
        grade = "weak-to-moderate"
    else:
        grade = "weak"

    return {
        "score_0_to_100": score_0_to_100,
        "score_0_to_1": round(weighted, 3),
        "grade": grade,
        "components_used": [
            {"name": name, "value_0_to_1": round(value, 3), "weight": weight}
            for name, value, weight in components
        ],
        "note": (
            "This is a summary score derived from the category metrics. "
            "Use the per-category metrics as the primary evidence."
        ),
    }


def print_summary(metrics: dict[str, Any]) -> None:
    print("\n=== ORB-SLAM3 vs FlightRecord Consistency Summary ===")
    print(f"ORB rows:    {metrics['sample_counts']['orbslam_rows']}")
    print(f"Flight rows: {metrics['sample_counts']['flight_rows']}")
    print(f"Aligned rows:{metrics['sample_counts']['aligned_rows']}")

    score = metrics.get("consistency_score", {})
    if score.get("score_0_to_100") is not None:
        print(f"Overall consistency score: {score['score_0_to_100']:.1f}/100 ({score['grade']})")

    print("\nOrientation correlations (best of same sign / flipped sign):")
    for axis in ["yaw", "pitch", "roll"]:
        entry = metrics["orientation_correlation"][axis]
        print(f"  {axis:<5} best={fmt(entry['best'])} mode={entry['mode']}")

    print("\nRate correlations:")
    for axis in ["yaw_rate", "pitch_rate", "roll_rate"]:
        entry = metrics["rate_correlation"][axis]
        print(f"  {axis:<10} best={fmt(entry['best'])} mode={entry['mode']}")

    speed = metrics["speed_correlation"]
    print(f"\nSpeed correlation: best={fmt(speed['best'])} mode={speed['mode']}")

    vertical = metrics["vertical_axis"]
    if vertical is not None:
        axis_name = vertical["chosen_orbslam_axis"]
        ref_name = vertical["reference_used"]
        best_value = fmt(vertical["signed_correlation"]["best"])
        best_mode = vertical["signed_correlation"]["mode"]

        print(
            f"Vertical trend: ORB axis '{axis_name}' vs {ref_name} "
            f"=> best={best_value} mode={best_mode}"
        )
    else:
        print("Vertical trend: unavailable")

    print("\nState agreement:")
    print(f"  hover match fraction:    {fmt(metrics['state_agreement']['hover_match_fraction'])}")
    print(
        f"  movement match fraction: {fmt(metrics['state_agreement']['movement_match_fraction'])}"
    )

    print("\nTiming deltas (seconds, lower is better):")
    for name, stats in metrics["timing_deltas_seconds"].items():
        print(
            f"  {name:<12} count={stats['count']} mean={fmt(stats['mean'])} "
            f"median={fmt(stats['median'])} max={fmt(stats['max'])}"
        )


def fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.3f}"


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ORB-SLAM3 / DJI FlightRecord consistency.")
    parser.add_argument("--orbslam", required=True, help="Path to ORB-SLAM3 pose text file.")
    parser.add_argument("--flight", required=True, help="Path to DJI FlightRecord CSV.")
    parser.add_argument(
        "--outdir",
        default="consistency_report",
        help="Directory for JSON summary and aligned CSV output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    orb = load_orbslam(args.orbslam)
    flight = load_flightrecord(args.flight)
    metrics, merged = compare(orb, flight)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = outdir / "summary.json"
    aligned_path = outdir / "aligned_samples.csv"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    merged.to_csv(aligned_path, index=False)

    print_summary(metrics)
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved aligned samples: {aligned_path}")


if __name__ == "__main__":
    main()
