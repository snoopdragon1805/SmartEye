
import json
import math
import statistics
import time
from pathlib import Path

PROFILE_PATH = Path("models/user_profile.json")


def calculate_calibration(ear_values, blink_durations, duration):
    """
    Calculate a personal baseline from an alert calibration session.
    """

    if duration <= 0:
        raise ValueError("Calibration duration must be positive.")

    if len(ear_values) < 2:
        raise ValueError("Not enough EAR observations for calibration.")

    if any(not math.isfinite(value) or value < 0 for value in ear_values):
        raise ValueError("EAR observations must be finite and non-negative.")

    if any(
        not math.isfinite(value) or value <= 0
        for value in blink_durations
    ):
        raise ValueError("Blink durations must be finite and positive.")

    baseline_ear = statistics.mean(ear_values)
    ear_std = statistics.pstdev(ear_values)

    blink_rate = len(blink_durations) / duration * 60.0

    baseline_closure_duration = (
        statistics.mean(blink_durations)
        if blink_durations
        else 0.0
    )

    # Initial personalised threshold. This is a heuristic and
    # must be validated with the user's observed open/closed EAR.
    ear_threshold = max(
        0.10,
        min(0.25, baseline_ear - 2.0 * ear_std)
    )

    return {
        "baseline_ear": baseline_ear,
        "ear_std": ear_std,
        "baseline_blink_rate": blink_rate,
        "baseline_closure_duration": baseline_closure_duration,
        "ear_threshold": ear_threshold,
        "calibration_duration_seconds": duration,
        "calibration_timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
    }


def save_profile(profile, path=PROFILE_PATH):
    """Save the calibration profile as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=4)

    print(f"Calibration profile saved to: {path}")


def load_profile(path=PROFILE_PATH):
    """Load a previously saved calibration profile."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
