import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import data_collector


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_csv = Path(temp_dir) / "fatigue_data.csv"

        features = {
            "mean_ear": 0.25,
            "min_ear": 0.18,
            "ear_std": 0.02,
            "ear_range": 0.07,
            "blink_rate": 14.0,
            "mean_blink_duration": 0.30,
            "max_blink_duration": 0.60,
            "perclos": 12.0,
        }

        with patch.object(data_collector, "DATASET_PATH", str(test_csv)):
            data_collector.save_features(
                features,
                fatigue_label=1,
                current_window_id=1,
                fatigue_score=0.72,
                predicted_status="High",
            )

        with test_csv.open("r", newline="") as file:
            rows = list(csv.DictReader(file))

        assert len(rows) == 1, "Expected one CSV row"
        row = rows[0]

        assert row["fatigue_label"] == "1"
        assert row["fatigue_score"] == "0.72"
        assert row["predicted_status"] == "High"
        assert row["prediction_available"] == "True"
        assert float(row["mean_ear"]) == 0.25

        print("CSV logging test passed!")
        print("Self-reported label and predicted status are stored separately.")


if __name__ == "__main__":
    main()
