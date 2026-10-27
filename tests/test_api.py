import csv
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
import data_collector


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_health_and_saved_profile_are_real(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["ok"])

        profile = self.client.get("/api/calibration/profile")
        self.assertEqual(profile.status_code, 200)
        self.assertTrue(profile.json()["available"])
        self.assertIn("ear_threshold", profile.json()["profile"])

    def test_history_and_session_detail_read_real_csv_rows(self):
        sessions = self.client.get("/api/sessions")
        self.assertEqual(sessions.status_code, 200)
        self.assertTrue(sessions.json())
        session_id = sessions.json()[0]["session_id"]

        detail = self.client.get(f"/api/sessions/{session_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["session_id"], session_id)
        self.assertTrue(detail.json()["windows"])
        self.assertIn("fatigue_label", detail.json()["windows"][0])
        self.assertIn("predicted_status", detail.json()["windows"][0])

    def test_filtered_export_preserves_original_schema_and_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.csv"
            fields = [
                "participant_id",
                "session_id",
                "window_id",
                "fatigue_label",
                "fatigue_score",
                "predicted_status",
                "prediction_available",
            ]
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "participant_id": "P-test",
                        "session_id": "S-test",
                        "window_id": 1,
                        "fatigue_label": 2,
                        "fatigue_score": 0.75,
                        "predicted_status": "High",
                        "prediction_available": True,
                    }
                )
            with patch.object(api, "DATASET_PATH", path):
                response = self.client.get(
                    "/api/export/csv", params={"session_id": "S-test"}
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/csv; charset=utf-8")
        row = next(csv.DictReader(response.text.splitlines()))
        self.assertEqual(row["fatigue_label"], "2")
        self.assertEqual(row["predicted_status"], "High")
        self.assertEqual(row["fatigue_score"], "0.75")

    def test_missing_session_and_empty_export_return_not_found(self):
        detail = self.client.get("/api/sessions/no-such-session")
        self.assertEqual(detail.status_code, 404)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(api, "DATASET_PATH", Path(temp_dir) / "empty.csv"):
                response = self.client.get("/api/export/csv")
        self.assertEqual(response.status_code, 404)

    def test_websocket_reports_status_without_camera(self):
        with self.client.websocket_connect("/ws/monitoring") as websocket:
            message = websocket.receive_json()
        self.assertIn("status", message)
        self.assertIn("metrics", message)
        self.assertIn("pending_windows", message)

    def test_live_metrics_report_prediction_observation_requirement(self):
        service = api.MonitorService()
        service.window_started = time.monotonic()
        metrics = service._make_metrics(False, None, None)

        self.assertEqual(
            metrics["observation"]["minimum_required_seconds"],
            api.MINIMUM_OBSERVATION_SECONDS,
        )
        self.assertFalse(metrics["prediction_available"])

    def test_start_stop_controls_do_not_spawn_a_camera_worker(self):
        with patch.object(api.service, "_monitor_loop", return_value=None):
            started = self.client.post("/api/session/start")
            self.assertEqual(started.status_code, 202)
            self.assertTrue(started.json()["session_active"])
            stopped = self.client.post("/api/session/stop")
        self.assertEqual(stopped.status_code, 200)
        self.assertFalse(stopped.json()["session_active"])

    def test_window_self_report_is_saved_separately_from_prediction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "window.csv"
            with patch.object(api.service, "session_id", "S-test"), patch.dict(
                api.service.pending_windows,
                {
                    ("S-test", 1): {
                        "features": {"mean_ear": 0.25},
                        "fatigue_score": 0.8,
                        "predicted_status": "High",
                    }
                },
                clear=True,
            ), patch.object(
                data_collector, "DATASET_PATH", str(path)
            ), patch.object(data_collector, "SESSION_ID", "S-before"):
                response = self.client.post(
                    "/api/session/window/label",
                    json={
                        "session_id": "S-test",
                        "window_id": 1,
                        "fatigue_label": 0,
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()["prediction_available"])
                with path.open(newline="") as file:
                    row = next(csv.DictReader(file))
                self.assertEqual(row["fatigue_label"], "0")
                self.assertEqual(row["predicted_status"], "High")
                self.assertEqual(row["fatigue_score"], "0.8")


if __name__ == "__main__":
    unittest.main()
