
from collections import deque
import math
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
        long_closure_duration=0.80,
        max_valid_gap=0.20,
    ):
        self.window_duration = window_duration
        self.ear_threshold = ear_threshold
        self.long_closure_duration = long_closure_duration
        self.max_valid_gap = max_valid_gap

        self.ear_history = deque()
        self.blink_timestamps = deque()

        self.eyes_closed = False
        self.closure_start = None

    def update(self, timestamp, ear, blink_detected=False):
        if (
            not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or not math.isfinite(timestamp)
        ):
            raise ValueError("timestamp must be a finite number.")

        if (
            not isinstance(ear, (int, float))
            or isinstance(ear, bool)
            or not math.isfinite(ear)
            or ear <= 0
        ):
            self.mark_unavailable(timestamp)
            return

        if self.ear_history and timestamp <= self.ear_history[-1][0]:
            raise ValueError("timestamps must be strictly increasing.")

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
            current_time = time.monotonic()

        ear_values = [
            ear for _, ear in self.ear_history if ear is not None
        ]

        if ear_values:
            mean_ear = sum(ear_values) / len(ear_values)
            min_ear = min(ear_values)
            max_ear = max(ear_values)
        else:
            mean_ear = min_ear = max_ear = 0.0

        blink_count = len(self.blink_timestamps)

        if len(self.ear_history) >= 2:
            observed_duration = sum(
                next_timestamp - timestamp
                for (timestamp, ear), (
                    next_timestamp,
                    next_ear,
                ) in zip(
                    self.ear_history,
                    list(self.ear_history)[1:],
                )
                if (
                    ear is not None
                    and next_ear is not None
                    and 0 < next_timestamp - timestamp <= self.max_valid_gap
                )
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

    def mark_unavailable(self, timestamp):
        if (
            not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or not math.isfinite(timestamp)
        ):
            raise ValueError("timestamp must be a finite number.")
        if self.ear_history and timestamp <= self.ear_history[-1][0]:
            raise ValueError("timestamps must be strictly increasing.")

        self.ear_history.append((timestamp, None))
        cutoff = timestamp - self.window_duration
        while self.ear_history and self.ear_history[0][0] < cutoff:
            self.ear_history.popleft()
        while self.blink_timestamps and self.blink_timestamps[0] < cutoff:
            self.blink_timestamps.popleft()
        self.break_tracking()

    def break_tracking(self):
        self.eyes_closed = False
        self.closure_start = None

    def reset_window(self):
        self.ear_history.clear()
        self.blink_timestamps.clear()
        self.break_tracking()
