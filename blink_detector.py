
import math


class BlinkDetector:

    def __init__(
        self,
        ear_threshold=0.20,
        min_blink_duration=0.08,
        max_blink_duration=0.80
    ):
        self.ear_threshold = ear_threshold
        self.min_blink_duration = min_blink_duration
        self.max_blink_duration = max_blink_duration

        self.eyes_closed = False
        self.eye_close_start = None

        self.blink_count = 0
        self.blink_durations = []

    def update(self, ear, current_time):
        blink_detected = False

        if (
            not isinstance(ear, (int, float))
            or isinstance(ear, bool)
            or not math.isfinite(ear)
            or ear <= 0
            or not isinstance(current_time, (int, float))
            or isinstance(current_time, bool)
            or not math.isfinite(current_time)
        ):
            self.reset_tracking()
            return False

        if ear < self.ear_threshold:

            if not self.eyes_closed:
                self.eyes_closed = True
                self.eye_close_start = current_time

        elif self.eyes_closed:

            eye_close_duration = (
                current_time - self.eye_close_start
            )

            if (
                self.min_blink_duration
                <= eye_close_duration
                <= self.max_blink_duration
            ):
                self.blink_count += 1
                self.blink_durations.append(
                    eye_close_duration
                )
                blink_detected = True

            self.eyes_closed = False
            self.eye_close_start = None

        return blink_detected

    def get_statistics(self):
        return {
            "blink_count": self.blink_count,
            "blink_durations": self.blink_durations.copy()
        }

    def reset_window(self):
        self.blink_count = 0
        self.blink_durations.clear()
        self.reset_tracking()

    def reset_tracking(self):
        self.eyes_closed = False
        self.eye_close_start = None
