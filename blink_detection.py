import cv2
import mediapipe as mp
import math
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/face_landmarker.task"

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Initial prototype threshold
EAR_THRESHOLD = 0.20

# Minimum and maximum duration for a blink
MIN_BLINK_DURATION = 0.08
MAX_BLINK_DURATION = 0.80


def euclidean_distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


def calculate_ear(eye_points):

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)

    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


class BlinkDetector:

    def __init__(
        self,
        ear_threshold=EAR_THRESHOLD,
        min_blink_duration=MIN_BLINK_DURATION,
        max_blink_duration=MAX_BLINK_DURATION
    ):

        self.ear_threshold = ear_threshold
        self.min_blink_duration = min_blink_duration
        self.max_blink_duration = max_blink_duration

        self.eyes_closed = False
        self.eye_close_start = None

        self.blink_count = 0
        self.blink_durations = []


    def update(self, ear):

        current_time = time.time()

        blink_detected = False


        # Eyes are currently closed
        if ear < self.ear_threshold:

            # Eyes just became closed
            if not self.eyes_closed:

                self.eyes_closed = True
                self.eye_close_start = current_time


        # Eyes are currently open
        else:

            # Eyes just became open
            if self.eyes_closed:

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


# --------------------------------
# MediaPipe setup
# --------------------------------

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# --------------------------------
# Camera
# --------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


frame_timestamp_ms = 0

# Create blink detector
blink_detector = BlinkDetector()


print("Blink detection started.")
print("Press 'q' to quit.")


# --------------------------------
# Main loop
# --------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break


    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )

    frame_timestamp_ms += 33


    if result.face_landmarks:

        face = result.face_landmarks[0]

        height, width, _ = frame.shape

        left_eye_points = []
        right_eye_points = []


        # --------------------------------
        # Left eye
        # --------------------------------

        for index in LEFT_EYE:

            landmark = face[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            left_eye_points.append((x, y))


        # --------------------------------
        # Right eye
        # --------------------------------

        for index in RIGHT_EYE:

            landmark = face[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            right_eye_points.append((x, y))


        # --------------------------------
        # EAR
        # --------------------------------

        left_ear = calculate_ear(
            left_eye_points
        )

        right_ear = calculate_ear(
            right_eye_points
        )

        average_ear = (
            left_ear + right_ear
        ) / 2.0


        # --------------------------------
        # Blink detection
        # --------------------------------

        blink_detector.update(
            average_ear
        )


        # --------------------------------
        # Display eye points
        # --------------------------------

        for point in left_eye_points:

            cv2.circle(
                frame,
                point,
                3,
                (0, 255, 0),
                -1
            )


        for point in right_eye_points:

            cv2.circle(
                frame,
                point,
                3,
                (0, 255, 0),
                -1
            )


        # --------------------------------
        # Display statistics
        # --------------------------------

        stats = blink_detector.get_statistics()

        cv2.putText(
            frame,
            f"EAR: {average_ear:.3f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Blinks: {stats['blink_count']}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        status = (
            "CLOSED"
            if blink_detector.eyes_closed
            else "OPEN"
        )

        cv2.putText(
            frame,
            f"Eyes: {status}",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    cv2.imshow(
        "SmartEye - Blink Detection",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
landmarker.close()
cv2.destroyAllWindows()