
from perclos_calculator import PERCLOSCalculator


def run_test(name, ear_values, expected_perclos):
    tracker = PERCLOSCalculator(max_valid_gap=0.20)

    for i, ear in enumerate(ear_values):
        tracker.update(i * 0.1, ear)

    perclos, observation_time = tracker.calculate()

    print(
        f"{name}: PERCLOS={perclos:.2f}%, "
        f"observed={observation_time:.2f}s"
    )

    assert abs(perclos - expected_perclos) < 0.01
    assert observation_time > 0


run_test("Eyes open", [0.26] * 11, 0.0)
run_test("Eyes closed", [0.10] * 11, 100.0)
run_test(
    "Half open, half closed",
    [0.26] * 5 + [0.10] * 6,
    50.0
)

print("All PERCLOS tests passed!")
