
from collections import deque
import math


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

        # Stores (timestamp, EAR), with None marking an invalid sample.
        self.history = deque()

    def update(self, timestamp, ear):
        if (
            not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or not math.isfinite(timestamp)
        ):
            raise ValueError("timestamp must be a finite number.")

        if ear is not None and (
            not isinstance(ear, (int, float))
            or isinstance(ear, bool)
            or not math.isfinite(ear)
            or ear <= 0
        ):
            ear = None

        if self.history and timestamp <= self.history[-1][0]:
            raise ValueError("timestamps must be strictly increasing.")

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
            next_timestamp, next_ear = self.history[i + 1]

            interval = next_timestamp - current_timestamp

            # Ignore invalid or excessively long gaps.
            if (
                interval <= 0
                or interval > self.max_valid_gap
                or current_ear is None
                or next_ear is None
            ):
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
