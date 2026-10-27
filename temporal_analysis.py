
from collections import deque
import time


class TemporalAnalyzer:
    """
    Analyses eye behaviour over a rolling time window.

    Tracks:
    - EAR variation
    - Blink frequency
    - Long eye closures
    - Sustained low EAR
    """

    def __init__(
        self,
        window_duration=10.0,
        ear_threshold=0.20,
        long_closure_duration=0.80
    ):
        self.window_duration = window_duration
        self.ear_threshold = ear_threshold
        self.long_closure_duration = long_closure_duration

        self.ear_history = deque()
        self.blink_timestamps = deque()

        self.eyes_closed = False
        self.closure_start = None

    def update(self, timestamp, ear, blink_detected=False):
        self.ear_history.append((timestamp, ear))

        if blink_detected:
            self.blink_timestamps.append(timestamp)

        if ear < self.ear_threshold:
            if not self.eyes_closed:
                self.eyes_closed = True
                self.closure_start = timestamp
        else:
            self.eyes_closed = False
            self.closure_start = None

        cutoff = timestamp - self.window_duration

        while self.ear_history and self.ear_history[0][0] < cutoff:
            self.ear_history.popleft()

        while self.blink_timestamps and self.blink_timestamps[0] < cutoff:
            self.blink_timestamps.popleft()

    def get_statistics(self, current_time=None):
        if current_time is None:
            current_time = time.time()

        ear_values = [ear for _, ear in self.ear_history]

        if ear_values:
            mean_ear = sum(ear_values) / len(ear_values)
            min_ear = min(ear_values)
            max_ear = max(ear_values)
        else:
            mean_ear = min_ear = max_ear = 0.0

        blink_count = len(self.blink_timestamps)

        if len(self.ear_history) >= 2:
            observed_duration = (
                self.ear_history[-1][0] - self.ear_history[0][0]
            )
        else:
            observed_duration = 0.0

        blink_rate = (
            blink_count / observed_duration * 60.0
            if observed_duration > 0
            else 0.0
        )

        current_closure_duration = (
            current_time - self.closure_start
            if self.eyes_closed and self.closure_start is not None
            else 0.0
        )

        return {
            "mean_ear": mean_ear,
            "min_ear": min_ear,
            "max_ear": max_ear,
            "ear_range": max_ear - min_ear,
            "blink_count": blink_count,
            "blink_rate": blink_rate,
            "observed_duration": observed_duration,
            "eyes_closed": self.eyes_closed,
            "current_closure_duration": current_closure_duration,
            "long_closure": (
                current_closure_duration >= self.long_closure_duration
            ),
        }
