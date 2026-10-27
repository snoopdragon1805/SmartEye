import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import data_collector
from fatigue_scoring import classify_fatigue, load_profile
from feature_engineering import calculate_features


class RuntimePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_profile("models/user_profile.json")

    def test_profile_threshold_reaches_each_analyzer_and_survives_reset(self):
        threshold, blink, perclos, temporal = (
            data_collector.create_analysis_components(self.profile)
        )

        self.assertEqual(threshold, self.profile["ear_threshold"])
        self.assertEqual(blink.ear_threshold, threshold)
        self.assertEqual(perclos.ear_threshold, threshold)
        self.assertEqual(temporal.ear_threshold, threshold)

        blink.update(threshold / 2, 0.0)
        perclos.update(0.0, threshold / 2)
        temporal.update(0.0, threshold / 2)
        blink.reset_window()
        perclos.reset_window()
        temporal.reset_window()
        self.assertEqual(blink.ear_threshold, threshold)
        self.assertEqual(perclos.ear_threshold, threshold)
        self.assertEqual(temporal.ear_threshold, threshold)
        self.assertFalse(blink.eyes_closed)
        self.assertFalse(perclos.history)
        self.assertFalse(temporal.ear_history)
        self.assertFalse(temporal.eyes_closed)

    def test_face_loss_breaks_closure_without_fabricating_observations(self):
        _, blink, perclos, temporal = (
            data_collector.create_analysis_components(self.profile)
        )
        ear_values = []

        data_collector.update_analysis(
            0.10, 0.0, ear_values, blink, perclos, temporal, self.profile
        )
        data_collector.update_analysis(
            0.10, 0.1, ear_values, blink, perclos, temporal, self.profile
        )
        data_collector.update_analysis(
            None, 0.2, ear_values, blink, perclos, temporal, self.profile
        )
        resumed = data_collector.update_analysis(
            0.28, 0.3, ear_values, blink, perclos, temporal, self.profile
        )

        self.assertEqual(blink.get_statistics()["blink_count"], 0)
        self.assertEqual(ear_values, [0.1, 0.1, 0.28])
        self.assertFalse(resumed["temporal_stats"]["eyes_closed"])
        self.assertEqual(resumed["observation_time"], 0.1)
        self.assertEqual(resumed["perclos"], 100.0)

    def test_scoring_waits_for_sufficient_observations(self):
        _, blink, perclos, temporal = (
            data_collector.create_analysis_components(self.profile)
        )
        ear_values = []
        latest = data_collector.update_analysis(
            0.28,
            0.0,
            ear_values,
            blink,
            perclos,
            temporal,
            self.profile,
        )

        self.assertIsNone(latest["fatigue_result"])
        self.assertEqual(latest["fatigue_label"], "Calibrating...")

    def test_scoring_uses_valid_observations_despite_gaps_in_rolling_window(self):
        _, blink, perclos, temporal = (
            data_collector.create_analysis_components(self.profile)
        )
        ear_values = []
        first_ready = None

        for frame in range(900):
            timestamp = frame / 30.0
            face_available = timestamp % 4.0 < 2.0
            latest = data_collector.update_analysis(
                0.28 if face_available else None,
                timestamp,
                ear_values,
                blink,
                perclos,
                temporal,
                self.profile,
            )
            if (
                latest is not None
                and latest["observation_time"] >= 10.0
            ):
                first_ready = latest
                break

        self.assertIsNotNone(first_ready)
        self.assertGreaterEqual(first_ready["observation_time"], 10.0)
        self.assertLess(
            first_ready["temporal_stats"]["observed_duration"],
            data_collector.MINIMUM_OBSERVATION_SECONDS,
        )
        self.assertIsNotNone(first_ready["fatigue_result"])

    def test_invalid_eye_geometry_is_not_treated_as_closed_eyes(self):
        invalid_eye = [(1, 1)] * 6
        self.assertIsNone(data_collector.calculate_ear(invalid_eye))

    def test_end_to_end_analysis_and_prediction_logging(self):
        _, blink, perclos, temporal = (
            data_collector.create_analysis_components(self.profile)
        )
        ear_values = []
        latest = None

        open_eye = [
            (0, 0),
            (25, -15),
            (75, -15),
            (100, 0),
            (75, 15),
            (25, 15),
        ]
        closed_eye = [
            (0, 0),
            (25, -5),
            (75, -5),
            (100, 0),
            (75, 5),
            (25, 5),
        ]
        self.assertAlmostEqual(data_collector.calculate_ear(open_eye), 0.3)
        self.assertAlmostEqual(data_collector.calculate_ear(closed_eye), 0.1)

        for frame in range(481):
            timestamp = frame / 30.0
            if 2.0 <= timestamp < 2.17:
                ear = data_collector.calculate_ear(closed_eye)
            else:
                ear = data_collector.calculate_ear(open_eye)

            if 4.0 <= timestamp < 5.0:
                latest = data_collector.update_analysis(
                    None,
                    timestamp,
                    ear_values,
                    blink,
                    perclos,
                    temporal,
                    self.profile,
                )
                continue

            latest = data_collector.update_analysis(
                ear,
                timestamp,
                ear_values,
                blink,
                perclos,
                temporal,
                self.profile,
            )

        self.assertGreater(blink.get_statistics()["blink_count"], 0)
        self.assertGreater(len(ear_values), 300)
        self.assertIsNotNone(latest)
        self.assertGreaterEqual(latest["observation_time"], 10.0)
        self.assertIsNotNone(latest["fatigue_result"])
        self.assertGreaterEqual(latest["perclos"], 0.0)
        self.assertLessEqual(latest["perclos"], 100.0)
        score = latest["fatigue_result"]["fatigue_score"]
        status = classify_fatigue(score)
        self.assertIn(status, {"Normal", "Moderate", "High"})
        self.assertTrue(math.isfinite(score))

        features = calculate_features(
            ear_values,
            blink.get_statistics()["blink_durations"],
            latest["perclos"],
            60.0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fatigue_data.csv"
            with patch.object(data_collector, "DATASET_PATH", str(path)):
                data_collector.save_features(
                    features,
                    fatigue_label=2,
                    current_window_id=1,
                    fatigue_score=score,
                    predicted_status=status,
                )

            with path.open(newline="") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["fatigue_label"], "2")
            self.assertAlmostEqual(float(row["fatigue_score"]), score)
            self.assertEqual(row["predicted_status"], status)
            self.assertEqual(row["prediction_available"], "True")

    def test_prediction_fields_cannot_disagree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fatigue_data.csv"
            with patch.object(data_collector, "DATASET_PATH", str(path)):
                with self.assertRaises(ValueError):
                    data_collector.save_features(
                        {},
                        fatigue_label=0,
                        current_window_id=1,
                        fatigue_score=None,
                        predicted_status="High",
                    )
                with self.assertRaises(ValueError):
                    data_collector.save_features(
                        {},
                        fatigue_label=0,
                        current_window_id=1,
                        fatigue_score=0.1,
                        predicted_status="High",
                    )

    def test_fatigue_classification_thresholds_are_configurable(self):
        self.assertEqual(classify_fatigue(0.329), "Normal")
        self.assertEqual(classify_fatigue(0.33), "Moderate")
        self.assertEqual(classify_fatigue(0.659), "Moderate")
        self.assertEqual(classify_fatigue(0.66), "High")
        self.assertEqual(classify_fatigue(0.39, 0.4, 0.7), "Normal")
        self.assertEqual(classify_fatigue(0.5, 0.4, 0.7), "Moderate")
        self.assertEqual(classify_fatigue(0.7, 0.4, 0.7), "High")
        with self.assertRaises(ValueError):
            classify_fatigue(0.5, 0.7, 0.4)


if __name__ == "__main__":
    unittest.main()
