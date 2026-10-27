
from calibration import calculate_calibration, save_profile, load_profile
from pathlib import Path
import tempfile

ear_values = [0.28, 0.29, 0.27, 0.30, 0.28, 0.29]
blink_durations = [0.15, 0.18, 0.16]

profile = calculate_calibration(
    ear_values=ear_values,
    blink_durations=blink_durations,
    duration=30.0
)

print("Calibration profile:")
for key, value in profile.items():
    print(f"{key}: {value}")

assert profile["baseline_ear"] > 0
assert profile["baseline_blink_rate"] == 6.0
assert 0.10 <= profile["ear_threshold"] <= 0.25

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "user_profile.json"

    save_profile(profile, path)
    loaded = load_profile(path)

    assert loaded == profile

print("\nAll calibration tests passed!")
