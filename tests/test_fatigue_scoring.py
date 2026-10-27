
from fatigue_scoring import (
    load_profile,
    calculate_fatigue_score,
    classify_fatigue,
)

profile = load_profile()

# Simulated values representing relatively open eyes and low PERCLOS.
normal = calculate_fatigue_score(
    current_ear=profile["baseline_ear"],
    current_blink_rate=profile["baseline_blink_rate"],
    perclos=2.0,
    current_closure_duration=0.20,
    profile=profile,
)

# Simulated values representing reduced eye openness and prolonged closure.
fatigued = calculate_fatigue_score(
    current_ear=0.12,
    current_blink_rate=20.0,
    perclos=70.0,
    current_closure_duration=1.5,
    profile=profile,
)

print("Normal-like example:")
print(normal)
print("Classification:", classify_fatigue(normal["fatigue_score"]))

print("\nFatigue-like example:")
print(fatigued)
print("Classification:", classify_fatigue(fatigued["fatigue_score"]))

assert 0 <= normal["fatigue_score"] <= 1
assert 0 <= fatigued["fatigue_score"] <= 1
assert fatigued["fatigue_score"] > normal["fatigue_score"]

print("\nAll fatigue-scoring tests passed!")

def test_longer_blinks_increase_score():
    profile = load_profile()

    baseline_result = calculate_fatigue_score(
        current_ear=0.2616,
        current_blink_rate=16.0,
        perclos=10.99,
        current_closure_duration=0.0,
        recent_mean_blink_duration=profile[
            "baseline_closure_duration"
        ],
        profile=profile,
    )

    longer_blink_result = calculate_fatigue_score(
        current_ear=0.2616,
        current_blink_rate=16.0,
        perclos=10.99,
        current_closure_duration=0.0,
        recent_mean_blink_duration=0.4093,
        profile=profile,
    )

    assert (
        longer_blink_result["fatigue_score"]
        > baseline_result["fatigue_score"]
    )


if __name__ == "__main__":
    test_longer_blinks_increase_score()
    print("Closure-duration regression test passed!")