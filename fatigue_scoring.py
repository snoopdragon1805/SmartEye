
import json
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

    required = [
        "baseline_ear",
        "baseline_blink_rate",
        "baseline_closure_duration",
        "ear_threshold",
    ]

    missing = [key for key in required if key not in profile]
    if missing:
        raise ValueError(f"Calibration profile missing: {missing}")

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

    weights = weights or DEFAULT_WEIGHTS.copy()

    if set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError("Weights must define all four features.")

    if any(weight < 0 for weight in weights.values()):
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
        if value is None or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a numeric value.")
        if not 0 <= value:
            raise ValueError(f"{name} cannot be negative.")

    if perclos > 100:
        raise ValueError("PERCLOS must be between 0 and 100.")

    baseline_ear = float(profile["baseline_ear"])
    baseline_blink_rate = float(profile["baseline_blink_rate"])
    baseline_closure = float(profile["baseline_closure_duration"])

    if baseline_ear <= 0:
        raise ValueError("Baseline EAR must be positive.")

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
        if recent_mean_blink_duration < 0:
            raise ValueError("Blink duration cannot be negative.")

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


def classify_fatigue(score):
    """Map a score to the report's initial alertness categories."""
    if not 0.0 <= score <= 1.0:
        raise ValueError("Score must be between 0 and 1.")

    if score < 0.33:
        return "Normal"
    if score < 0.66:
        return "Moderate"
    return "High"
