import cv2
import mediapipe as mp
import math
import time
import csv
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from blink_detector import BlinkDetector
from perclos_calculator import PERCLOSCalculator
from feature_engineering import calculate_features


MODEL_PATH = "models/face_landmarker.task"

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.20

WINDOW_DURATION = 60.0

PARTICIPANT_ID = "P01"

SESSION_ID = time.strftime(
    "S_%Y%m%d_%H%M%S"
)

DATASET_PATH = "dataset/fatigue_data.csv"


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

    return (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)


def save_features(features, fatigue_label):

    os.makedirs(
        os.path.dirname(DATASET_PATH),
        exist_ok=True
    )

    file_exists = os.path.exists(DATASET_PATH)

    fieldnames = [
        "participant_id",
        "session_id",
        "window_id",
        "mean_ear",
        "min_ear",
        "blink_rate",
        "mean_blink_duration",
        "max_blink_duration",
        "perclos",
        "fatigue_label"
    ]

    with open(
        DATASET_PATH,
        "a",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "participant_id": PARTICIPANT_ID,
            "session_id": SESSION_ID,
            "window_id": window_id,
            **features,
            "fatigue_label": fatigue_label        })


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

# --------------------------------------------------
# Detectors
# --------------------------------------------------

blink_detector = BlinkDetector(
    ear_threshold=EAR_THRESHOLD
)

perclos_tracker = PERCLOSCalculator(
    ear_threshold=EAR_THRESHOLD,
    window_duration=WINDOW_DURATION
)


# --------------------------------------------------
# Feature collection state
# --------------------------------------------------

ear_values = []

window_id = 1

window_start_time = time.time()

print("SmartEye data collection started.")
print("Collecting one feature window of 60 seconds.")
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
        # Store EAR
        # --------------------------------

        ear_values.append(
            average_ear
        )


        # --------------------------------
        # Update blink detector
        # --------------------------------

        blink_detector.update(
            average_ear
        )


        # --------------------------------
        # Update PERCLOS
        # --------------------------------

        perclos_tracker.update(
            current_time,
            average_ear
        )


        # --------------------------------
        # Current PERCLOS
        # --------------------------------

        perclos, observation_time = (
            perclos_tracker.calculate()
        )


        # --------------------------------
        # Display
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


        cv2.putText(
            frame,
            f"EAR: {average_ear:.3f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        stats = blink_detector.get_statistics()

        cv2.putText(
            frame,
            f"Blinks: {stats['blink_count']}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        if observation_time >= 10.0:

            cv2.putText(
                frame,
                f"PERCLOS: {perclos:.2f}%",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        elapsed_time = (
            current_time - window_start_time
        )

        remaining_time = max(
            0.0,
            WINDOW_DURATION - elapsed_time
        )


        cv2.putText(
            frame,
            f"Window remaining: {remaining_time:.1f}s",
            (30, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # --------------------------------
        # Complete feature window
        # --------------------------------

        if elapsed_time >= WINDOW_DURATION:

            stats = blink_detector.get_statistics()

            final_perclos, _ = (
                perclos_tracker.calculate()
            )


            features = calculate_features(
                ear_values=ear_values,
                blink_durations=stats["blink_durations"],
                perclos=final_perclos,
                window_duration=WINDOW_DURATION
            )


            print("\nFeature window completed.")

            print("\nHow tired do you feel right now?")
            print("0 - Not tired")
            print("1 - Tired")
            print("2 - Very tired")

            while True:
                try:
                    fatigue_label = int(
                        input("Enter your fatigue level (0/1/2): ")
                    )

                    if fatigue_label in [0, 1, 2]:
                        break

                    print("Please enter 0, 1, or 2.")

                except ValueError:
                    print("Please enter a valid number.")


            features["participant_id"] = PARTICIPANT_ID
            features["session_id"] = SESSION_ID
            features["window_id"] = window_id
            features["fatigue_label"] = fatigue_label

            save_features(features, fatigue_label)

            for name, value in features.items():

                if isinstance(value, (int, float)):
                    print(
                        f"{name}: {value:.4f}"
                    )
                else:
                    print(
                        f"{name}: {value}"
                    )


            print(
                f"\nSaved to: {DATASET_PATH}"
            )


            # Reset for next window
            ear_values.clear()

            blink_detector.reset_window()

            perclos_tracker.reset_window()

            window_id += 1

            window_start_time = time.time()


    cv2.imshow(
        "SmartEye - Data Collection",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
landmarker.close()
cv2.destroyAllWindows()