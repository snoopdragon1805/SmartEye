# SmartEye

SmartEye is a webcam-based eye-behavior monitoring prototype. It calculates
eye aspect ratio (EAR), detects blinks, estimates PERCLOS, summarizes recent
temporal behavior, and displays a rule-based fatigue score. The score is a
prototype heuristic, not a medical assessment or a clinically validated
measure.

## Requirements

- Python with the versions supported by the pinned packages in
  `requirements.txt`
- A working webcam and permission for OpenCV to use it
- The MediaPipe face-landmarker task file at
  `models/face_landmarker.task`

Install the declared dependencies in your chosen Python environment:

```bash
python -m pip install -r requirements.txt
```

Run commands from the repository root because model, profile, and dataset
paths are relative to the current working directory.

## Calibrate and run

1. Run `python run_calibration.py` and look normally at the camera for the
   30-second calibration period. The script saves the personalized baseline
   and `ear_threshold` to `models/user_profile.json` when calibration has
   sufficient valid observations and face detection.
2. Run `python data_collector.py`. It loads that profile before creating its
   blink, PERCLOS, and temporal analyzers. Keep your face visible; each
   60-second collection window needs at least 10 seconds of valid observations
   before it is saved. Press `q` to quit.
3. At the end of a saved window, enter the self-reported fatigue label:
   `0` (not tired), `1` (tired), or `2` (very tired).

The dashboard's live score is available after enough valid observation time.
The score combines baseline-relative EAR, blink-rate deviation, PERCLOS, and
closure-duration components with weights `0.35`, `0.10`, `0.30`, and `0.25`.
The result is clipped to `[0, 1]`; default categories are Normal below `0.33`,
Moderate from `0.33` to below `0.66`, and High from `0.66`. The category
thresholds can be configured through `classify_fatigue`. These values are
prototype heuristics. The High banner is triggered by a current High score;
the project does not define or validate a separate sustained-high warning
duration.

## Dataset and model training

`dataset/fatigue_data.csv` stores window-level features, the independently
self-reported `fatigue_label`, and—when live scoring is available—the
`fatigue_score`, `predicted_status`, and `prediction_available` fields.
Existing rows whose prediction columns are blank are historical records; they
are not backfilled because the original predictions are unavailable.

Run `python train_model.py` to check whether the available labels support a
participant-independent Random Forest evaluation. Participants are kept
disjoint between training and test partitions. With the current data, which
contains only participant `P01`, the script correctly skips training and
metrics: an independent participant holdout cannot be formed. Add data from
more participants before interpreting classifier results. Training is
currently an offline experiment and does not replace the live rule-based
dashboard.

## Tests and checks

Run the camera-free unit tests with:

```bash
python -m unittest discover -v
```

Tests are organized in `tests/`. The webcam-backed preprocessing demo is
separate from unit-test discovery; run it explicitly with
`python -m tests.test_eye_preprocessing` when a camera and the MediaPipe model
are available. Its synthetic preprocessing test remains camera-free.
For a simple webcam capture check without face detection, run
`python -m tests.camera_test`.

## Dataset analysis images

Run `python analyze_dataset.py` from the repository root to generate feature
distribution plots and a correlation heatmap in `images/analysis/`.

The camera-backed end-to-end path is not exercised by synthetic tests; run
`python data_collector.py` with an attached, permitted webcam to validate your
camera and MediaPipe installation. Camera capture and MediaPipe resources are
released on normal exit and capture-loop errors. FPS and camera-to-dashboard
latency have not been benchmarked.

## Web dashboard

The React/Vite dashboard connects to the FastAPI camera service. Install the
Python dependencies (including FastAPI and Uvicorn) and the frontend packages:

```bash
.venv/bin/python -m pip install -r requirements.txt
cd frontend
npm install
```

In one terminal at the repository root, start the backend with one worker and
no reload process. The API owns the webcam; do not run the CLI collector or
calibration script at the same time:

```bash
.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

In a second terminal, also from the repository root, start the frontend:

```bash
cd frontend
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173`. From **Personal calibration**, run the 30-second
camera calibration. Replacing an existing profile requires confirmation. Then
start monitoring from the overview or live-monitoring page. At the end of each
valid 60-second feature window, submit a self-report to save the real features
and independent prediction fields to the CSV, or explicitly skip that window.
Session history and CSV export are based only on saved records. Camera access
is local to the backend process; this local-only API is not intended to be
exposed to an untrusted network.

To create a production frontend bundle, run `npm run build` from `frontend`.
