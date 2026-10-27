
import cv2
import mediapipe as mp
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from blink_detector import BlinkDetector
from calibration import calculate_calibration, save_profile

MODEL_PATH = "models/face_landmarker.task"
CALIBRATION_DURATION = 30.0

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def euclidean_distance(p1, p2):
    return math.dist(p1, p2)


def calculate_ear(points):
    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


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

landmarker = vision.FaceLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    landmarker.close()
    raise RuntimeError("Could not open webcam.")

ear_values = []
blink_detector = BlinkDetector(ear_threshold=0.20)

start_time = time.monotonic()
frame_timestamp_ms = 0
face_frames = 0
total_frames = 0

print("Calibration started.")
print("Look at the screen normally and remain alert.")
print("Press q to cancel.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read webcam frame.")
            break

        total_frames += 1
        now = time.monotonic()
        elapsed = now - start_time
        remaining = max(0.0, CALIBRATION_DURATION - elapsed)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )
        frame_timestamp_ms += 33

        if result.face_landmarks:
            face_frames += 1
            landmarks = result.face_landmarks[0]
            height, width = frame.shape[:2]

            left_points = [
                (
                    int(landmarks[i].x * width),
                    int(landmarks[i].y * height)
                )
                for i in LEFT_EYE
            ]

            right_points = [
                (
                    int(landmarks[i].x * width),
                    int(landmarks[i].y * height)
                )
                for i in RIGHT_EYE
            ]

            left_ear = calculate_ear(left_points)
            right_ear = calculate_ear(right_points)
            average_ear = (left_ear + right_ear) / 2.0

            ear_values.append(average_ear)
            blink_detector.update(average_ear, now)

            cv2.putText(
                frame,
                f"EAR: {average_ear:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "Face not detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.putText(
            frame,
            f"Calibration remaining: {remaining:.1f}s",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 200, 0),
            2
        )

        cv2.imshow("SmartEye - Personal Calibration", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Calibration cancelled.")
            break

        if elapsed >= CALIBRATION_DURATION:
            break

    if elapsed >= CALIBRATION_DURATION:
        if len(ear_values) < 2:
            print("Calibration failed: insufficient EAR observations.")
        elif face_frames / max(total_frames, 1) < 0.80:
            print("Calibration failed: face detection was too inconsistent.")
        else:
            stats = blink_detector.get_statistics()

            profile = calculate_calibration(
                ear_values=ear_values,
                blink_durations=stats["blink_durations"],
                duration=CALIBRATION_DURATION
            )

            profile["face_detection_rate"] = (
                face_frames / total_frames
            )

            save_profile(profile)

            print("\nCalibration complete!")
            for key, value in profile.items():
                print(f"{key}: {value}")

finally:
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
