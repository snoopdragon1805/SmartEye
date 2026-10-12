import numpy as np


def calculate_features(ear_values, blink_durations, perclos, window_duration):
    """
    Convert raw eye/blink measurements from a time window
    into features that can later be used by an ML model.
    """

    if not ear_values:
        return None

    features = {
        "mean_ear": float(np.mean(ear_values)),
        "min_ear": float(np.min(ear_values)),
        "blink_rate": (len(blink_durations) / window_duration) * 60,
        "mean_blink_duration": (
            float(np.mean(blink_durations))
            if blink_durations
            else 0.0
        ),
        "max_blink_duration": (
            float(np.max(blink_durations))
            if blink_durations
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

    blink_durations = [
        0.18,
        0.21,
        0.16,
        0.24
    ]

    perclos = 12.5
    window_duration = 60

    features = calculate_features(
        ear_values,
        blink_durations,
        perclos,
        window_duration
    )

    print("Extracted Features:")
    for name, value in features.items():
        print(f"{name}: {value:.3f}")