
import json
import math
from pathlib import Path


DEFAULT_WEIGHTS = {
    "ear": 0.35,
    "blink_rate": 0.10,
    "perclos": 0.30,
    "closure_duration": 0.25,
}


def load_profile(path="models/user_profile.json"):
    """Load the personal calibration profile."""
    with Path(path).open("r", encoding="utf-8") as file:
        profile = json.load(file)

    if not isinstance(profile, dict):
        raise ValueError("Calibration profile must be a JSON object.")

    required = [
        "baseline_ear",
        "baseline_blink_rate",
        "baseline_closure_duration",
        "ear_threshold",
    ]

    missing = [key for key in required if key not in profile]
    if missing:
        raise ValueError(f"Calibration profile missing: {missing}")

    try:
        numeric_values = {
            key: float(profile[key])
            for key in required
        }
    except (TypeError, ValueError) as error:
        raise ValueError("Calibration profile values must be numeric.") from error

    if not all(math.isfinite(value) for value in numeric_values.values()):
        raise ValueError("Calibration profile values must be finite.")
    if (
        numeric_values["baseline_ear"] <= 0
        or numeric_values["baseline_blink_rate"] < 0
        or numeric_values["baseline_closure_duration"] < 0
        or not 0 < numeric_values["ear_threshold"] < 1
    ):
        raise ValueError("Calibration profile values are outside valid ranges.")

    return profile


def clip01(value):
    """Clip a numeric value to the range [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def calculate_fatigue_score(
    current_ear,
    current_blink_rate,
    perclos,
    current_closure_duration,
    profile,
    recent_mean_blink_duration=None,
    weights=None,
):
    """
    Calculate an initial rule-based fatigue score in [0, 1].

    This is a prototype heuristic, not a clinically validated measure.
    PERCLOS must be provided as a percentage from 0 to 100.
    """

    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    if set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError("Weights must define all four features.")

    if any(
        not isinstance(weight, (int, float))
        or isinstance(weight, bool)
        or not math.isfinite(weight)
        or weight < 0
        for weight in weights.values()
    ):
        raise ValueError("Weights cannot be negative.")

    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("Weights must sum to 1.")

    values = {
        "current_ear": current_ear,
        "current_blink_rate": current_blink_rate,
        "perclos": perclos,
        "current_closure_duration": current_closure_duration,
    }

    for name, value in values.items():
        if (
            value is None
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a numeric value.")
        if not 0 <= value:
            raise ValueError(f"{name} cannot be negative.")

    if perclos > 100:
        raise ValueError("PERCLOS must be between 0 and 100.")

    try:
        baseline_ear = float(profile["baseline_ear"])
        baseline_blink_rate = float(profile["baseline_blink_rate"])
        baseline_closure = float(profile["baseline_closure_duration"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Profile contains invalid fatigue baselines.") from error

    if (
        not all(
            math.isfinite(value)
            for value in (
                baseline_ear,
                baseline_blink_rate,
                baseline_closure,
            )
        )
        or baseline_ear <= 0
        or baseline_blink_rate < 0
        or baseline_closure < 0
    ):
        raise ValueError("Profile contains invalid fatigue baselines.")

    # Lower EAR than baseline increases the EAR contribution.
    ear_deviation = clip01(
        (baseline_ear - current_ear) / baseline_ear
    )

    # Blink frequency can change in either direction.
    # Blink-rate changes are supporting evidence only.
    # A large change in either direction is capped and given less weight.
    if baseline_blink_rate > 0:
        relative_change = abs(
            current_blink_rate - baseline_blink_rate
        ) / baseline_blink_rate

        blink_deviation = clip01(relative_change / 1.5)
    else:
        blink_deviation = 0.0

    # PERCLOS is already a percentage; convert to [0, 1].
    perclos_deviation = clip01(perclos / 100.0)

    # Closures longer than baseline contribute more.
    # Use completed blink duration as well as an ongoing closure.
    # This prevents the duration signal from becoming zero just
    # because the eyes happen to be open at the scoring instant.

    if recent_mean_blink_duration is None:
        closure_measure = current_closure_duration
    else:
        if (
            not isinstance(recent_mean_blink_duration, (int, float))
            or isinstance(recent_mean_blink_duration, bool)
            or not math.isfinite(recent_mean_blink_duration)
            or recent_mean_blink_duration < 0
        ):
            raise ValueError(
                "Blink duration must be finite and non-negative."
            )

        closure_measure = max(
            current_closure_duration,
            recent_mean_blink_duration,
        )

    if baseline_closure > 0:
        closure_deviation = clip01(
            (closure_measure - baseline_closure)
            / baseline_closure
        )
    else:
        closure_deviation = clip01(
            closure_measure / 0.80
        )

    components = {
        "ear": ear_deviation,
        "blink_rate": blink_deviation,
        "perclos": perclos_deviation,
        "closure_duration": closure_deviation,
    }

    score = sum(
        weights[name] * components[name]
        for name in weights
    )

    return {
        "fatigue_score": clip01(score),
        "components": components,
        "weights": weights.copy(),
    }


def classify_fatigue(
    score,
    moderate_threshold=0.33,
    high_threshold=0.66,
):
    """Map a score to the report's initial alertness categories."""
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(score)
    ):
        raise ValueError("Score must be a finite number.")
    if not 0.0 <= score <= 1.0:
        raise ValueError("Score must be between 0 and 1.")
    if (
        not isinstance(moderate_threshold, (int, float))
        or isinstance(moderate_threshold, bool)
        or not math.isfinite(moderate_threshold)
        or not isinstance(high_threshold, (int, float))
        or isinstance(high_threshold, bool)
        or not math.isfinite(high_threshold)
        or not 0.0 <= moderate_threshold < high_threshold <= 1.0
    ):
        raise ValueError(
            "Thresholds must satisfy 0 <= moderate < high <= 1."
        )

    if score < moderate_threshold:
        return "Normal"
    if score < high_threshold:
        return "Moderate"
    return "High"
