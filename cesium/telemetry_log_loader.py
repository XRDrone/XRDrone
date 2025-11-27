import csv
import json
import os
from datetime import datetime

def compute_epoch(date_local: str, time_local: str) -> float:
    """
    Convert DJI's CUSTOM.date [local] and CUSTOM.updateTime [local]
    into Unix epoch seconds (float).
    Example:
      date_local = "11/25/2025"
      time_local = "7:05:08.97 PM"
    """
    if not date_local or not time_local:
        return None

    try:
        dt = datetime.strptime(
            f"{date_local} {time_local}",
            "%m/%d/%Y %I:%M:%S.%f %p"
        )
        return dt.timestamp()
    except Exception as e:
        print(f"Failed to parse datetime: {date_local} {time_local} ({e})")
        return None


def csv_to_json(csv_path, json_path):
    rows = []

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Extract DJI fields
            date_local = row.get("CUSTOM.date [local]", "").strip()
            time_local = row.get("CUSTOM.updateTime [local]", "").strip()

            # Compute epoch timestamp
            epoch = compute_epoch(date_local, time_local)

            # Insert into row dictionary
            row["timestamp_local"] = time_local
            row["date_local"] = date_local
            row["timestamp_epoch"] = epoch

            rows.append(row)

    with open(json_path, "w", encoding="utf-8") as out:
        json.dump(rows, out, indent=2)

    print(f"\nSuccess! Wrote {len(rows)} JSON objects to: {json_path}")


if __name__ == "__main__":
    print("Enter the CSV filename (must be in the current directory):")
    filename = input("> ").strip()

    if not os.path.isfile(filename):
        print(f"\nError: File '{filename}' not found in current directory.")
        exit(1)

    # Output JSON name
    output_name = os.path.splitext(filename)[0] + ".json"

    csv_to_json(filename, output_name)
