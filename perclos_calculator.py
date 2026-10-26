
from collections import deque


class PERCLOSCalculator:

    def __init__(
        self,
        ear_threshold=0.20,
        window_duration=60.0,
        max_valid_gap=0.20
    ):
        self.ear_threshold = ear_threshold
        self.window_duration = window_duration
        self.max_valid_gap = max_valid_gap

        # Stores (timestamp, EAR)
        self.history = deque()

    def update(self, timestamp, ear):
        self.history.append((timestamp, ear))

        cutoff_time = timestamp - self.window_duration

        while (
            self.history
            and self.history[0][0] < cutoff_time
        ):
            self.history.popleft()

    def calculate(self):
        if len(self.history) < 2:
            return 0.0, 0.0

        closed_time = 0.0
        observed_time = 0.0

        for i in range(len(self.history) - 1):
            current_timestamp, current_ear = self.history[i]
            next_timestamp, _ = self.history[i + 1]

            interval = next_timestamp - current_timestamp

            # Ignore invalid or excessively long gaps.
            if interval <= 0 or interval > self.max_valid_gap:
                continue

            observed_time += interval

            if current_ear < self.ear_threshold:
                closed_time += interval

        if observed_time <= 0:
            return 0.0, 0.0

        perclos = (closed_time / observed_time) * 100.0

        return perclos, observed_time

    def get_observation_time(self):
        _, observed_time = self.calculate()
        return observed_time

    def reset_window(self):
        self.history.clear()
