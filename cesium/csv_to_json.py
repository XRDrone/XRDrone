import csv
import json
import os

def csv_to_json(csv_path, json_path):
    rows = []

    # Read CSV
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Write JSON list
    with open(json_path, "w", encoding="utf-8") as out:
        json.dump(rows, out, indent=2)

    print(f"\nSuccess! Wrote {len(rows)} JSON objects to: {json_path}")


if __name__ == "__main__":
    print("Enter the CSV filename (must be in the current directory):")
    filename = input("> ").strip()

    # Ensure it exists in current directory
    if not os.path.isfile(filename):
        print(f"\nError: File '{filename}' not found in current directory.")
        exit(1)

    # Create output filename
    output_name = os.path.splitext(filename)[0] + ".json"

    csv_to_json(filename, output_name)
