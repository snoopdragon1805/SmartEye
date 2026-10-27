
import math

import numpy as np


def calculate_features(
    ear_values,
    blink_durations,
    perclos,
    window_duration
):
    """
    Extract statistical features from one observation window.
    """

    if len(ear_values) == 0:
        return None

    if (
        not isinstance(window_duration, (int, float))
        or isinstance(window_duration, bool)
        or not math.isfinite(window_duration)
        or window_duration <= 0
    ):
        raise ValueError("window_duration must be positive")

    ear_values = np.asarray(ear_values, dtype=float)
    blink_durations = np.asarray(
        blink_durations, dtype=float
    )

    if (
        not np.isfinite(ear_values).all()
        or (ear_values <= 0).any()
    ):
        raise ValueError("EAR observations must be finite and positive.")
    if (
        not np.isfinite(blink_durations).all()
        or (blink_durations < 0).any()
    ):
        raise ValueError(
            "Blink durations must be finite and non-negative."
        )
    if (
        not isinstance(perclos, (int, float))
        or isinstance(perclos, bool)
        or not math.isfinite(perclos)
        or not 0 <= perclos <= 100
    ):
        raise ValueError("PERCLOS must be finite and between 0 and 100.")

    features = {
        "mean_ear": float(np.mean(ear_values)),
        "min_ear": float(np.min(ear_values)),

        # New temporal/statistical features
        "ear_std": float(np.std(ear_values)),
        "ear_range": float(
            np.max(ear_values) - np.min(ear_values)
        ),

        "blink_rate": (
            len(blink_durations) / window_duration
        ) * 60.0,

        "mean_blink_duration": (
            float(np.mean(blink_durations))
            if len(blink_durations) > 0
            else 0.0
        ),

        "max_blink_duration": (
            float(np.max(blink_durations))
            if len(blink_durations) > 0
            else 0.0
        ),

        "perclos": float(perclos),
    }

    return features


if __name__ == "__main__":

    ear_values = [
        0.26, 0.25, 0.24, 0.23,
        0.21, 0.19, 0.08, 0.07,
        0.24, 0.25, 0.26
    ]

    blink_durations = [0.18, 0.21, 0.16, 0.24]

    features = calculate_features(
        ear_values=ear_values,
        blink_durations=blink_durations,
        perclos=12.5,
        window_duration=60.0
    )

    print("Extracted Features:")

    for name, value in features.items():
        print(f"{name}: {value:.4f}")
