from perclos_calculator import PERCLOSCalculator


tracker = PERCLOSCalculator()

# Simulated observations
tracker.update(0.0, 0.26)
tracker.update(1.0, 0.26)
tracker.update(2.0, 0.10)
tracker.update(3.0, 0.10)
tracker.update(4.0, 0.26)
tracker.update(5.0, 0.26)

perclos, observation_time = tracker.calculate()

print("PERCLOS:", perclos)
print("Observation time:", observation_time)    