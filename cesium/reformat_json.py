#!/usr/bin/env python3
import json
import os
from pathlib import Path


def reformat_frames(in_path: str, out_path: str) -> int:
    """
    Read frames.json (from dji-log-parser) and rewrite it as a list of
    per-timestamp packets with all original data preserved.
    Returns the number of frames processed.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    out_packets = []

    for frame in frames:
        packet = {}

        # Add timestamp first (derived from custom.dateTime)
        custom = frame.get("custom", {})
        ts = None
        if isinstance(custom, dict):
            ts = custom.get("dateTime")

        packet["timestamp"] = ts  # always the first field

        # Copy all original frame sections unchanged
        for key, value in frame.items():
            packet[key] = value

        out_packets.append(packet)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_packets, f, indent=2)

    return len(out_packets)


if __name__ == "__main__":
    print("Enter the frames JSON filename (must be in the current directory):")
    filename = input("> ").strip()

    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' not found in current directory.")
        raise SystemExit(1)

    base, ext = os.path.splitext(filename)
    if not ext:
        ext = ".json"

    output_name = f"{base}_reformatted{ext}"

    frame_count = reformat_frames(filename, output_name)

    print(f"Frames processed: {frame_count}")
    print(f"File done: {output_name}")
