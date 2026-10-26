import csv
import os
from pathlib import Path

path = Path("dataset/fatigue_data.csv")
temp_path = path.with_suffix(".tmp")

new_columns = [
    "fatigue_score",
    "predicted_status",
    "prediction_available",
]

with path.open("r", newline="") as source:
    reader = csv.DictReader(source)
    old_fields = reader.fieldnames

    if not old_fields:
        raise SystemExit("CSV has no header; migration cancelled.")

    fields = old_fields + [
        column for column in new_columns
        if column not in old_fields
    ]

    rows = list(reader)

with temp_path.open("w", newline="") as destination:
    writer = csv.DictWriter(destination, fieldnames=fields)
    writer.writeheader()

    for row in rows:
        for column in new_columns:
            if column not in row:
                row[column] = ""
        writer.writerow(row)

os.replace(temp_path, path)
print(f"Migrated {len(rows)} rows.")
print(f"New columns: {', '.join(fields)}")
