import cv2
import mediapipe as mp
import math
import time
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/face_landmarker.task"

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Initial prototype threshold
EAR_THRESHOLD = 0.20

# Rolling observation window
WINDOW_DURATION = 60.0

# Minimum data required before displaying PERCLOS
MIN_OBSERVATION_TIME = 10.0


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


class PERCLOSCalculator:

    def __init__(
        self,
        ear_threshold=EAR_THRESHOLD,
        window_duration=WINDOW_DURATION
    ):

        self.ear_threshold = ear_threshold
        self.window_duration = window_duration

        # Stores:
        # (timestamp, EAR)
        self.history = deque()


    def update(self, timestamp, ear):

        # Add current observation
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
            closed_time
            / total_time
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


# --------------------------------------------------
# MediaPipe
# --------------------------------------------------

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


# --------------------------------------------------
# Camera
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


frame_timestamp_ms = 0

# Create PERCLOS calculator
perclos_tracker = PERCLOSCalculator()


print("PERCLOS detection started.")
print("Press 'q' to quit.")


# --------------------------------------------------
# Main loop
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break


    current_time = time.time()


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
        # Update PERCLOS
        # --------------------------------

        perclos_tracker.update(
            current_time,
            average_ear
        )


        perclos, observation_time = (
            perclos_tracker.calculate()
        )


        # --------------------------------
        # Eye state
        # --------------------------------

        if average_ear < EAR_THRESHOLD:
            eye_status = "CLOSED"
        else:
            eye_status = "OPEN"


        # --------------------------------
        # Draw landmarks
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
        # Display EAR
        # --------------------------------

        cv2.putText(
            frame,
            f"EAR: {average_ear:.3f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # --------------------------------
        # Display PERCLOS
        # --------------------------------

        if observation_time >= MIN_OBSERVATION_TIME:

            cv2.putText(
                frame,
                f"PERCLOS: {perclos:.2f}%",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:

            remaining = (
                MIN_OBSERVATION_TIME
                - observation_time
            )

            cv2.putText(
                frame,
                f"PERCLOS warming up: {remaining:.1f}s",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


        # --------------------------------
        # Display eye state
        # --------------------------------

        cv2.putText(
            frame,
            f"Eyes: {eye_status}",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # --------------------------------
        # Display observation window
        # --------------------------------

        cv2.putText(
            frame,
            f"Window: {observation_time:.1f}s",
            (30, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    cv2.imshow(
        "SmartEye - PERCLOS",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
landmarker.close()
cv2.destroyAllWindows()