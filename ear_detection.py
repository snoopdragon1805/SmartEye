import cv2
import mediapipe as mp
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/face_landmarker.task"

# Eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def euclidean_distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


def calculate_ear(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR).

    eye_points:
        [p1, p2, p3, p4, p5, p6]

    EAR = (vertical_1 + vertical_2) / (2 * horizontal)
    """

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)

    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


# MediaPipe setup
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


# Camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_timestamp_ms = 0

print("EAR detection started.")
print("Press 'q' to quit.")


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


        # Extract left eye points
        for index in LEFT_EYE:

            landmark = face[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            left_eye_points.append((x, y))


        # Extract right eye points
        for index in RIGHT_EYE:

            landmark = face[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            right_eye_points.append((x, y))


        # Calculate EAR
        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)

        # Average both eyes
        average_ear = (left_ear + right_ear) / 2.0


        # Draw eye landmarks
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


        # Display EAR
        cv2.putText(
            frame,
            f"Left EAR: {left_ear:.3f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Right EAR: {right_ear:.3f}",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Average EAR: {average_ear:.3f}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    cv2.imshow(
        "SmartEye - Eye Aspect Ratio",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
landmarker.close()
cv2.destroyAllWindows()