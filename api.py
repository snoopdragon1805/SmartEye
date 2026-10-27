import asyncio
import csv
import math
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pydantic import BaseModel, Field

import data_collector
from calibration import calculate_calibration, save_profile
from fatigue_scoring import load_profile
from feature_engineering import calculate_features


ROOT = Path(__file__).resolve().parent
PROFILE_PATH = ROOT / "models" / "user_profile.json"
MODEL_PATH = ROOT / "models" / "face_landmarker.task"
DATASET_PATH = ROOT / "dataset" / "fatigue_data.csv"
WINDOW_DURATION = data_collector.WINDOW_DURATION
MINIMUM_OBSERVATION_SECONDS = data_collector.MINIMUM_OBSERVATION_SECONDS

@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    if service.mode != "idle":
        await asyncio.to_thread(service.stop)


app = FastAPI(title="SmartEye API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SelfReport(BaseModel):
    session_id: str
    window_id: int = Field(ge=1)
    fatigue_label: int = Field(ge=0, le=2)


class WindowReference(BaseModel):
    session_id: str
    window_id: int = Field(ge=1)


class CalibrationRequest(BaseModel):
    confirm_replace: bool = False


class MonitorService:
    def __init__(self):
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.mode = "idle"
        self.session_id = None
        self.session_started = None
        self.session_elapsed = 0.0
        self.window_id = 1
        self.window_started = None
        self.profile = None
        self.ear_threshold = None
        self.blink_detector = None
        self.perclos_tracker = None
        self.temporal_analyzer = None
        self.ear_values = []
        self.latest_prediction = None
        self.latest_metrics = None
        self.latest_frame = None
        self.history = deque(maxlen=240)
        self.pending_windows = {}
        self.error = None
        self.calibration_progress = 0.0
        self.calibration_message = None
        self.camera_connected = False
        self.face_detected = False

    def status(self):
        with self.lock:
            elapsed = self.session_elapsed
            if self.mode == "monitoring" and self.session_started is not None:
                elapsed = time.monotonic() - self.session_started
            profile_loaded = self.profile is not None
            threshold = self.ear_threshold
            if not profile_loaded:
                try:
                    saved_profile = load_profile(PROFILE_PATH)
                    threshold = float(saved_profile["ear_threshold"])
                    profile_loaded = math.isfinite(threshold)
                except (OSError, KeyError, TypeError, ValueError):
                    profile_loaded = False
            return {
                "backend_connected": True,
                "mode": self.mode,
                "session_active": self.mode == "monitoring",
                "session_id": self.session_id,
                "session_elapsed_seconds": elapsed,
                "window_id": self.window_id,
                "camera_connected": self.camera_connected,
                "face_detected": self.face_detected,
                "calibration_available": profile_loaded,
                "ear_threshold": threshold,
                "error": self.error,
                "calibration_progress": self.calibration_progress,
                "calibration_message": self.calibration_message,
            }

    def current_metrics(self):
        with self.lock:
            return self.latest_metrics

    def pending_window_summaries(self):
        with self.lock:
            return [
                {"session_id": session_id, "window_id": window_id}
                for session_id, window_id in self.pending_windows
            ]

    def start_monitoring(self):
        with self.lock:
            if self.mode != "idle":
                raise ValueError(f"Pipeline is already {self.mode}.")
            try:
                profile = load_profile(PROFILE_PATH)
                components = data_collector.create_analysis_components(profile)
            except (OSError, ValueError, TypeError) as error:
                raise ValueError(f"Calibration profile is unavailable: {error}") from error

            self.profile = profile
            (
                self.ear_threshold,
                self.blink_detector,
                self.perclos_tracker,
                self.temporal_analyzer,
            ) = components
            self.session_id = time.strftime("S_%Y%m%d_%H%M%S")
            self.session_started = time.monotonic()
            self.window_started = self.session_started
            self.session_elapsed = 0.0
            self.window_id = 1
            self.ear_values = []
            self.latest_prediction = None
            self.latest_metrics = None
            self.latest_frame = None
            self.history.clear()
            self.pending_windows = {}
            self.error = None
            self.camera_connected = False
            self.face_detected = False
            self.stop_event.clear()
            self.mode = "monitoring"
            self.thread = threading.Thread(
                target=self._monitor_loop,
                name="smarteye-camera",
                daemon=True,
            )
            self.thread.start()

    def stop(self):
        with self.lock:
            if self.mode == "idle":
                return
            self.stop_event.set()
            worker = self.thread
        if worker and worker is not threading.current_thread():
            worker.join(timeout=10)
            if worker.is_alive():
                raise RuntimeError("Camera worker did not stop in time.")
        with self.lock:
            if self.session_started is not None:
                self.session_elapsed = time.monotonic() - self.session_started
            self.mode = "idle"
            self.camera_connected = False
            self.face_detected = False
            self.thread = None

    def start_calibration(self):
        with self.lock:
            if self.mode != "idle":
                raise ValueError(f"Pipeline is already {self.mode}.")
            self.profile = None
            try:
                self.profile = load_profile(PROFILE_PATH)
            except FileNotFoundError:
                pass
            if self.profile is not None:
                raise FileExistsError(
                    "A calibration profile already exists; confirm replacement first."
                )
            self._start_calibration_thread()

    def replace_calibration(self):
        with self.lock:
            if self.mode != "idle":
                raise ValueError(f"Pipeline is already {self.mode}.")
            self.profile = None
            self._start_calibration_thread()

    def _start_calibration_thread(self):
        self.error = None
        self.calibration_progress = 0.0
        self.calibration_message = "Starting calibration."
        self.camera_connected = False
        self.stop_event.clear()
        self.mode = "calibrating"
        self.thread = threading.Thread(
            target=self._calibration_loop,
            name="smarteye-calibration",
            daemon=True,
        )
        self.thread.start()

    def cancel_calibration(self):
        with self.lock:
            if self.mode != "calibrating":
                raise ValueError("Calibration is not running.")
            self.stop_event.set()

    def submit_self_report(self, report):
        key = (report.session_id, report.window_id)
        with self.lock:
            pending = self.pending_windows.pop(key, None)
            if pending is None:
                raise KeyError("That completed window is not awaiting a label.")

        data_collector.SESSION_ID = report.session_id
        data_collector.save_features(
            pending["features"],
            report.fatigue_label,
            report.window_id,
            fatigue_score=pending["fatigue_score"],
            predicted_status=pending["predicted_status"],
        )
        return {
            "saved": True,
            "session_id": report.session_id,
            "window_id": report.window_id,
            "fatigue_label": report.fatigue_label,
            "prediction_available": pending["fatigue_score"] is not None,
        }

    def skip_window(self, session_id, window_id):
        key = (session_id, window_id)
        with self.lock:
            if self.pending_windows.pop(key, None) is None:
                raise KeyError("That completed window is not awaiting a label.")

    def _create_landmarker(self):
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def _make_metrics(self, face_detected, ear, analysis):
        elapsed = time.monotonic() - self.window_started
        with self.lock:
            metrics = {
                "session_id": self.session_id,
                "window_id": self.window_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "face_detected": face_detected,
                "prediction_available": False,
                "ear": None,
                "blink": None,
                "eye_closure": None,
                "fatigue": {"score": None, "category": "Unavailable"},
                "observation": {
                    "window_duration_seconds": min(elapsed, WINDOW_DURATION),
                    "valid_duration_seconds": None,
                    "minimum_required_seconds": MINIMUM_OBSERVATION_SECONDS,
                },
            }

            if analysis is not None:
                temporal = analysis["temporal_stats"]
                blink_stats = analysis["blink_stats"]
                durations = blink_stats["blink_durations"]
                metrics["ear"] = {
                    "current": ear,
                    "mean": temporal["mean_ear"],
                    "minimum": temporal["min_ear"],
                    "maximum": temporal["max_ear"],
                    "std": self._std_ear(),
                    "range": temporal["ear_range"],
                }
                metrics["blink"] = {
                    "count": blink_stats["blink_count"],
                    "rate_per_minute": (
                        temporal["blink_rate"]
                        if temporal["observed_duration"] > 0
                        else None
                    ),
                    "mean_duration_seconds": (
                        sum(durations) / len(durations) if durations else None
                    ),
                    "maximum_duration_seconds": max(durations) if durations else None,
                }
                metrics["eye_closure"] = {
                    "perclos_percent": (
                        analysis["perclos"]
                        if analysis["observation_time"] > 0
                        else None
                    ),
                    "current_duration_seconds": temporal[
                        "current_closure_duration"
                    ],
                    "long_closure": temporal["long_closure"],
                }
                metrics["observation"]["valid_duration_seconds"] = (
                    analysis["observation_time"]
                )
                result = analysis["fatigue_result"]
                if result is not None:
                    metrics["prediction_available"] = True
                    metrics["fatigue"] = {
                        "score": result["fatigue_score"],
                        "category": analysis["fatigue_label"],
                    }
                    self.latest_prediction = {
                        "score": result["fatigue_score"],
                        "category": analysis["fatigue_label"],
                    }
            self.latest_metrics = metrics
            self.face_detected = face_detected
            self.history.append(metrics)
            return metrics

    def _std_ear(self):
        values = self.ear_values
        if len(values) < 2:
            return None
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

    def _finish_window(self, window_start, final_perclos, stats, observed):
        if (
            len(self.ear_values) >= 2
            and observed >= MINIMUM_OBSERVATION_SECONDS
        ):
            features = calculate_features(
                self.ear_values,
                stats["blink_durations"],
                final_perclos,
                WINDOW_DURATION,
            )
            with self.lock:
                key = (self.session_id, self.window_id)
                self.pending_windows[key] = {
                    "features": features,
                    "fatigue_score": (
                        self.latest_prediction["score"]
                        if self.latest_prediction
                        else None
                    ),
                    "predicted_status": (
                        self.latest_prediction["category"]
                        if self.latest_prediction
                        else "Unavailable"
                    ),
                }
        with self.lock:
            self.window_id += 1
            self.window_started = window_start + WINDOW_DURATION
            profile = self.profile
            (
                self.ear_threshold,
                self.blink_detector,
                self.perclos_tracker,
                self.temporal_analyzer,
            ) = data_collector.create_analysis_components(profile)
            self.ear_values = []
            self.latest_prediction = None

    def _monitor_loop(self):
        camera = None
        landmarker = None
        frame_timestamp_ms = 0
        try:
            landmarker = self._create_landmarker()
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise RuntimeError(
                    "Could not open webcam. Check camera permissions and device availability."
                )
            with self.lock:
                self.camera_connected = True

            while not self.stop_event.is_set():
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Could not read a frame from the webcam.")
                now = time.monotonic()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, frame_timestamp_ms)
                frame_timestamp_ms += 33
                face_detected = bool(result.face_landmarks)
                ear = None
                analysis = None

                if face_detected:
                    face = result.face_landmarks[0]
                    height, width = frame.shape[:2]
                    eye_landmarks = [
                        face[index]
                        for index in (
                            data_collector.LEFT_EYE
                            + data_collector.RIGHT_EYE
                        )
                    ]
                    if all(
                        math.isfinite(point.x) and math.isfinite(point.y)
                        for point in eye_landmarks
                    ):
                        left = [
                            (int(face[index].x * width), int(face[index].y * height))
                            for index in data_collector.LEFT_EYE
                        ]
                        right = [
                            (int(face[index].x * width), int(face[index].y * height))
                            for index in data_collector.RIGHT_EYE
                        ]
                        left_ear = data_collector.calculate_ear(left)
                        right_ear = data_collector.calculate_ear(right)
                        if left_ear is not None and right_ear is not None:
                            ear = (left_ear + right_ear) / 2.0
                    analysis = data_collector.update_analysis(
                        ear,
                        now,
                        self.ear_values,
                        self.blink_detector,
                        self.perclos_tracker,
                        self.temporal_analyzer,
                        self.profile,
                    )
                else:
                    data_collector.update_analysis(
                        None,
                        now,
                        self.ear_values,
                        self.blink_detector,
                        self.perclos_tracker,
                        self.temporal_analyzer,
                        self.profile,
                    )

                cv2.putText(
                    frame,
                    "Face detected" if face_detected else "Face not detected",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 220, 180) if face_detected else (0, 0, 255),
                    2,
                )
                if ear is not None:
                    cv2.putText(
                        frame,
                        f"EAR: {ear:.3f}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                self._make_metrics(face_detected, ear, analysis)
                encoded, jpeg = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                )
                if encoded:
                    with self.lock:
                        self.latest_frame = jpeg.tobytes()

                elapsed = time.monotonic() - self.window_started
                if elapsed >= WINDOW_DURATION:
                    perclos, observed = self.perclos_tracker.calculate()
                    self._finish_window(
                        self.window_started,
                        perclos,
                        self.blink_detector.get_statistics(),
                        observed,
                    )
                with self.lock:
                    if self.session_started is not None:
                        self.session_elapsed = now - self.session_started
        except Exception as error:
            with self.lock:
                self.error = str(error)
                if self.mode == "monitoring":
                    self.mode = "error"
        finally:
            if camera is not None:
                camera.release()
            if landmarker is not None:
                landmarker.close()
            with self.lock:
                self.camera_connected = False
                self.face_detected = False

    def _calibration_loop(self):
        camera = None
        landmarker = None
        start = time.monotonic()
        ears = []
        detector = None
        face_frames = 0
        total_frames = 0
        frame_timestamp_ms = 0
        try:
            landmarker = self._create_landmarker()
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise RuntimeError(
                    "Could not open webcam. Check camera permissions and device availability."
                )
            with self.lock:
                self.camera_connected = True
                self.calibration_message = "Look toward the camera and remain alert."
            detector = data_collector.BlinkDetector(ear_threshold=0.20)
            while (
                not self.stop_event.is_set()
                and time.monotonic() - start < 30.0
            ):
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Could not read a frame from the webcam.")
                now = time.monotonic()
                total_frames += 1
                result = landmarker.detect_for_video(
                    mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    ),
                    frame_timestamp_ms,
                )
                frame_timestamp_ms += 33
                if result.face_landmarks:
                    face_frames += 1
                    face = result.face_landmarks[0]
                    height, width = frame.shape[:2]
                    eyes = []
                    for indices in (
                        data_collector.LEFT_EYE,
                        data_collector.RIGHT_EYE,
                    ):
                        landmarks = [face[index] for index in indices]
                        if not all(
                            math.isfinite(point.x) and math.isfinite(point.y)
                            for point in landmarks
                        ):
                            eyes.append(None)
                        else:
                            points = [
                                (
                                    int(point.x * width),
                                    int(point.y * height),
                                )
                                for point in landmarks
                            ]
                            eyes.append(
                                data_collector.calculate_ear(points)
                            )
                    if all(value is not None for value in eyes):
                        ear = sum(eyes) / 2.0
                        ears.append(ear)
                        detector.update(ear, now)
                progress = min(1.0, (now - start) / 30.0)
                with self.lock:
                    self.calibration_progress = progress
                time.sleep(0.001)
            if self.stop_event.is_set():
                with self.lock:
                    self.calibration_message = "Calibration cancelled."
            elif len(ears) < 2 or face_frames / max(total_frames, 1) < 0.80:
                raise RuntimeError(
                    "Calibration needs at least two valid EAR samples and face detection in 80% of frames."
                )
            else:
                profile = calculate_calibration(
                    ears,
                    detector.get_statistics()["blink_durations"],
                    30.0,
                )
                profile["face_detection_rate"] = face_frames / total_frames
                save_profile(profile, PROFILE_PATH)
                with self.lock:
                    self.profile = profile
                    self.ear_threshold = profile["ear_threshold"]
                    self.calibration_message = "Calibration completed and profile saved."
        except Exception as error:
            with self.lock:
                self.error = str(error)
                self.calibration_message = str(error)
        finally:
            if camera is not None:
                camera.release()
            if landmarker is not None:
                landmarker.close()
            with self.lock:
                self.camera_connected = False
                self.mode = "idle"
                self.thread = None
                self.stop_event.clear()


service = MonitorService()


def _read_rows():
    if not DATASET_PATH.exists():
        return []
    with DATASET_PATH.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _parse_number(value):
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


@app.get("/api/health")
def health():
    return {"ok": True, **service.status()}


@app.get("/api/status")
def status():
    return service.status()


@app.post("/api/session/start", status_code=202)
def start_session():
    try:
        service.start_monitoring()
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return service.status()


@app.post("/api/session/stop")
def stop_session():
    try:
        service.stop()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return service.status()


@app.get("/api/session/current")
def current_session():
    return {
        **service.status(),
        "metrics": service.current_metrics(),
        "pending_windows": service.pending_window_summaries(),
    }


@app.get("/api/metrics/current")
def current_metrics():
    return service.current_metrics() or {"available": False}


@app.get("/api/metrics/history")
def metrics_history(limit: int = Query(default=120, ge=1, le=240)):
    with service.lock:
        return list(service.history)[-limit:]


@app.get("/api/camera/stream")
def camera_stream():
    def frames():
        while service.mode == "monitoring":
            with service.lock:
                frame = service.latest_frame
            if frame:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            time.sleep(0.05)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/calibration/profile")
def calibration_profile():
    try:
        profile = load_profile(PROFILE_PATH)
    except FileNotFoundError:
        return {"available": False, "profile": None}
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"available": True, "profile": profile}


@app.post("/api/calibration/start", status_code=202)
def start_calibration(request: CalibrationRequest):
    try:
        if request.confirm_replace:
            service.replace_calibration()
        else:
            service.start_calibration()
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return service.status()


@app.post("/api/calibration/cancel")
def cancel_calibration():
    try:
        service.cancel_calibration()
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return service.status()


@app.post("/api/session/window/label")
def label_window(report: SelfReport):
    if report.session_id != service.session_id:
        raise HTTPException(status_code=404, detail="Session not found.")
    try:
        return service.submit_self_report(report)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/session/window/skip")
def skip_window(report: WindowReference):
    try:
        service.skip_window(report.session_id, report.window_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"skipped": True}


@app.get("/api/sessions")
def sessions():
    grouped = {}
    for row in _read_rows():
        session_id = row.get("session_id")
        if not session_id:
            continue
        group = grouped.setdefault(
            session_id,
            {
                "session_id": session_id,
                "participant_id": row.get("participant_id"),
                "windows": 0,
                "predictions_available": 0,
                "scores": [],
                "ears": [],
                "blink_rates": [],
                "perclos_values": [],
                "categories": {"Normal": 0, "Moderate": 0, "High": 0},
            },
        )
        group["windows"] += 1
        if row.get("prediction_available", "").lower() == "true":
            group["predictions_available"] += 1
        for column, target in (
            ("fatigue_score", "scores"),
            ("mean_ear", "ears"),
            ("blink_rate", "blink_rates"),
            ("perclos", "perclos_values"),
        ):
            number = _parse_number(row.get(column))
            if number is not None:
                group[target].append(number)
        category = row.get("predicted_status")
        if category in group["categories"]:
            group["categories"][category] += 1

    output = []
    for group in grouped.values():
        try:
            group["started_at"] = datetime.strptime(
                group["session_id"], "S_%Y%m%d_%H%M%S"
            ).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            group["started_at"] = None
        for source, target in (
            ("scores", "average_fatigue_score"),
            ("ears", "average_ear"),
            ("blink_rates", "average_blink_rate"),
            ("perclos_values", "average_perclos"),
        ):
            values = group.pop(source)
            group[target] = sum(values) / len(values) if values else None
        output.append(group)
    return sorted(output, key=lambda row: row["session_id"], reverse=True)


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: str):
    rows = [row for row in _read_rows() if row.get("session_id") == session_id]
    if not rows:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session_id": session_id, "windows": rows}


@app.get("/api/export/csv")
def export_csv(session_id: str | None = None):
    rows = _read_rows()
    if session_id is not None:
        rows = [row for row in rows if row.get("session_id") == session_id]
    if not rows:
        raise HTTPException(status_code=404, detail="No CSV records to export.")

    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    safe_session_id = "".join(
        character for character in (session_id or "all-sessions")
        if character.isalnum() or character in "-_"
    )
    filename = f"smarteye-{safe_session_id or 'session'}.csv"
    return Response(
        output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.websocket("/ws/monitoring")
async def monitoring_socket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(
                {
                    "status": service.status(),
                    "metrics": service.current_metrics(),
                    "pending_windows": service.pending_window_summaries(),
                }
            )
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
from contextlib import asynccontextmanager
