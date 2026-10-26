
from temporal_analysis import TemporalAnalyzer

analyzer = TemporalAnalyzer(
    window_duration=10.0,
    ear_threshold=0.20,
    long_closure_duration=0.80
)

# Simulate 2 seconds of observations at 10 FPS.
# Eyes are open initially, then remain closed.
for i in range(21):
    timestamp = i * 0.1

    if timestamp < 1.0:
        ear = 0.28
        blink_detected = False
    else:
        ear = 0.10
        blink_detected = False

    analyzer.update(
        timestamp,
        ear,
        blink_detected
    )

stats = analyzer.get_statistics(current_time=2.0)

print("Temporal Analysis Results")
for name, value in stats.items():
    print(f"{name}: {value}")

assert stats["mean_ear"] > 0
assert stats["min_ear"] == 0.10
assert stats["max_ear"] == 0.28
assert stats["eyes_closed"] is True
assert stats["long_closure"] is True

print("\nAll temporal-analysis tests passed!")
