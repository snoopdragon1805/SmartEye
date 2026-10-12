from collections import deque


class PERCLOSCalculator:

    def __init__(
        self,
        ear_threshold=0.20,
        window_duration=60.0
    ):
        self.ear_threshold = ear_threshold
        self.window_duration = window_duration

        # Stores:
        # (timestamp, EAR)
        self.history = deque()

    def update(self, timestamp, ear):

        self.history.append(
            (timestamp, ear)
        )

        # Remove observations older than
        # the rolling window
        cutoff_time = (
            timestamp - self.window_duration
        )

        while (
            self.history
            and self.history[0][0] < cutoff_time
        ):
            self.history.popleft()

    def calculate(self):

        if len(self.history) < 2:
            return 0.0, 0.0

        total_time = (
            self.history[-1][0]
            - self.history[0][0]
        )

        if total_time <= 0:
            return 0.0, 0.0

        closed_time = 0.0

        for i in range(len(self.history) - 1):

            current_timestamp, current_ear = (
                self.history[i]
            )

            next_timestamp, _ = (
                self.history[i + 1]
            )

            interval = (
                next_timestamp
                - current_timestamp
            )

            if current_ear < self.ear_threshold:
                closed_time += interval

        perclos = (
            closed_time / total_time
        ) * 100.0

        return perclos, total_time

    def get_observation_time(self):

        if len(self.history) < 2:
            return 0.0

        return (
            self.history[-1][0]
            - self.history[0][0]
        )

    def reset_window(self):

        self.history.clear()