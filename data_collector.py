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
from temporal_analysis import TemporalAnalyzer
from fatigue_scoring import (
    load_profile,
    calculate_fatigue_score,
    classify_fatigue,
)


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



def save_features(
    features,
    fatigue_label,
    current_window_id,
    fatigue_score=None,
    predicted_status="Unavailable",
):
    os.makedirs(
        os.path.dirname(DATASET_PATH),
        exist_ok=True
    )

    fieldnames = [
        "participant_id",
        "session_id",
        "window_id",
        "mean_ear",
        "min_ear",
        "ear_std",
        "ear_range",
        "blink_rate",
        "mean_blink_duration",
        "max_blink_duration",
        "perclos",
        "fatigue_label",
        "fatigue_score",
        "predicted_status",
        "prediction_available",
    ]

    file_exists = os.path.exists(DATASET_PATH)
    file_is_empty = (
        not file_exists
        or os.path.getsize(DATASET_PATH) == 0
    )

    with open(DATASET_PATH, "a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        if file_is_empty:
            writer.writeheader()

        row = {
            "participant_id": PARTICIPANT_ID,
            "session_id": SESSION_ID,
            "window_id": current_window_id,
            **features,
            "fatigue_label": fatigue_label,
            "fatigue_score": fatigue_score,
            "predicted_status": predicted_status,
            "prediction_available": fatigue_score is not None,
        }

        writer.writerow(row)



# --------------------------------------------------


def main():
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

    temporal_analyzer = TemporalAnalyzer(
        window_duration=10.0,
        ear_threshold=EAR_THRESHOLD,
        long_closure_duration=0.80
    )

    # Load personalised calibration profile
    user_profile = load_profile("models/user_profile.json")

    print("Personalised calibration profile loaded.")
    print(f"Baseline EAR: {user_profile['baseline_ear']:.3f}")
    print(f"Baseline blink rate: {user_profile['baseline_blink_rate']:.1f}/min")


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

    try:
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

                blink_detected = blink_detector.update(
                    average_ear,
                    current_time
                )

                temporal_analyzer.update(
                    timestamp=current_time,
                    ear=average_ear,
                    blink_detected=blink_detected
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

                temporal_stats = temporal_analyzer.get_statistics(
                    current_time=current_time
                )


                # --------------------------------
                # Live Fatigue Scoring
                # --------------------------------

                fatigue_result = None
                fatigue_label = "Calibrating..."

                # Wait for enough valid observations before scoring.
                if observation_time >= 10.0:

                    fatigue_result = calculate_fatigue_score(
                        current_ear=temporal_stats["mean_ear"],
                        current_blink_rate=temporal_stats["blink_rate"],
                        perclos=perclos,
                        current_closure_duration=temporal_stats[
                            "current_closure_duration"
                        ],
                        profile=user_profile,
                        recent_mean_blink_duration=(
                            sum(stats["blink_durations"]) / len(stats["blink_durations"])
                            if stats["blink_durations"]
                            else None
                        ),
                    )

                    fatigue_label = classify_fatigue(
                        fatigue_result["fatigue_score"]
                    )

                    print(
                        f"\rScore: {fatigue_result['fatigue_score']:.3f} | "
                        f"EAR: {fatigue_result['components']['ear']:.3f} | "
                        f"Blink: {fatigue_result['components']['blink_rate']:.3f} | "
                        f"PERCLOS: {fatigue_result['components']['perclos']:.3f} | "
                        f"Closure: {fatigue_result['components']['closure_duration']:.3f} | "
                        f"Status: {fatigue_label}",
                        end="",
                        flush=True,
                    )

            
                # --------------------------------
                # Fatigue Dashboard
                # --------------------------------

                if fatigue_result is not None:

                    score = fatigue_result["fatigue_score"]

                    cv2.putText(
                        frame,
                        f"Fatigue Score: {score:.3f}",
                        (30, 275),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Status: {fatigue_label}",
                        (30, 310),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255) if fatigue_label == "High"
                        else (0, 165, 255) if fatigue_label == "Moderate"
                        else (0, 255, 0),
                        2,
                    )

                else:
                    cv2.putText(
                        frame,
                        "Fatigue Score: Collecting observations...",
                        (30, 275),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )


                cv2.putText(
                    frame,
                    f"Rolling EAR: {temporal_stats['mean_ear']:.3f}",
                    (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 200, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Rolling Blink Rate: {temporal_stats['blink_rate']:.1f}/min",
                    (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 200, 0),
                    2
                )

                closure_text = (
                    f"Long Closure: YES "
                    f"({temporal_stats['current_closure_duration']:.1f}s)"
                    if temporal_stats["long_closure"]
                    else "Long Closure: NO"
                )

                cv2.putText(
                    frame,
                    closure_text,
                    (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255) if temporal_stats["long_closure"] else (0, 255, 0),
                    2
                )

                # --------------------------------
                # Fatigue Alert Banner
                # --------------------------------

                if fatigue_result is not None:
                    if fatigue_label == "High":
                        alert_message = "WARNING: HIGH FATIGUE - TAKE A BREAK"
                        alert_color = (0, 0, 180)

                    elif temporal_stats["long_closure"]:
                        alert_message = "WARNING: LONG EYE CLOSURE"
                        alert_color = (0, 100, 255)

                    else:
                        alert_message = None

                    if alert_message:
                        banner_y = frame.shape[0] - 55

                        cv2.rectangle(
                            frame,
                            (0, banner_y),
                            (frame.shape[1], frame.shape[0]),
                            alert_color,
                            -1,
                        )

                        cv2.putText(
                            frame,
                            alert_message,
                            (15, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 255),
                            2,
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

                    save_features(
                        features,
                        fatigue_label,
                        window_id,
                        fatigue_score=(
                            fatigue_result["fatigue_score"]
                            if fatigue_result is not None
                            else None
                        ),
                        predicted_status=(
                            classify_fatigue(fatigue_result["fatigue_score"])
                            if fatigue_result is not None
                            else "Unavailable"
                        ),
                    )

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

                    temporal_analyzer = TemporalAnalyzer(
                        window_duration=10.0,
                        ear_threshold=EAR_THRESHOLD,
                        long_closure_duration=0.80
                    )

                    window_id += 1

                    window_start_time = time.time()


            cv2.imshow(
                "SmartEye - Data Collection",
                frame
            )


            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
