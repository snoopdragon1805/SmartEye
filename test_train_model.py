import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from train_model import FEATURES, train_and_evaluate


class ModelTrainingTest(unittest.TestCase):
    def test_participant_grouped_training_and_inference(self):
        rows = []
        for participant in range(12):
            for label in (0, 1, 2):
                rows.append(
                    {
                        "participant_id": f"P{participant:02d}",
                        "fatigue_label": label,
                        "mean_ear": 0.30 - label * 0.05,
                        "min_ear": 0.20 - label * 0.04,
                        "blink_rate": 10.0 + label * 5,
                        "mean_blink_duration": 0.15 + label * 0.1,
                        "max_blink_duration": 0.25 + label * 0.15,
                        "perclos": label * 30.0,
                    }
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model_fixture.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                model = train_and_evaluate(path)

        self.assertIsNotNone(model)
        prediction = model.predict(
            pd.DataFrame(
                [
                    {
                        "mean_ear": 0.30,
                        "min_ear": 0.20,
                        "blink_rate": 10.0,
                        "mean_blink_duration": 0.15,
                        "max_blink_duration": 0.25,
                        "perclos": 0.0,
                    }
                ],
                columns=FEATURES,
            )
        )
        self.assertIn(int(prediction[0]), (0, 1, 2))

    def test_one_participant_is_not_evaluated(self):
        rows = []
        for label in (0, 1, 2):
            rows.append(
                {
                    "participant_id": "P01",
                    "fatigue_label": label,
                    "mean_ear": 0.3,
                    "min_ear": 0.2,
                    "blink_rate": 10,
                    "mean_blink_duration": 0.2,
                    "max_blink_duration": 0.3,
                    "perclos": 10,
                }
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "one_participant.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                model = train_and_evaluate(path)

        self.assertIsNone(model)
        self.assertIn("at least two participants", output.getvalue())


if __name__ == "__main__":
    unittest.main()
