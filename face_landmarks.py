import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Path to the MediaPipe model
MODEL_PATH = "models/face_landmarker.task"


# Create model options
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


# Create the face landmarker
landmarker = vision.FaceLandmarker.create_from_options(options)


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


print("Face landmark detection started.")
print("Press 'q' to quit.")


frame_timestamp_ms = 0


while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break
        
    # OpenCV uses BGR, MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame into a MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect landmarks
    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )

    frame_timestamp_ms += 33

    # Draw landmarks manually
    if result.face_landmarks:

        for face in result.face_landmarks:

            height, width, _ = frame.shape

            for landmark in face:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )

    # Display
    cv2.imshow(
        "SmartEye - Face Landmarks",
        frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
landmarker.close()
cv2.destroyAllWindows()