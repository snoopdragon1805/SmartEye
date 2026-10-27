import unittest

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from eye_preprocessing import extract_eye_region, preprocess_eye


MODEL_PATH = "models/face_landmarker.task"
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


class EyePreprocessingTest(unittest.TestCase):
    def test_synthetic_eye_crop_is_resized_and_normalized(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        points = [
            (115, 120),
            (135, 110),
            (150, 108),
            (205, 120),
            (185, 130),
            (150, 132),
        ]

        crop = extract_eye_region(frame, points)
        processed = preprocess_eye(crop)

        self.assertIsNotNone(crop)
        self.assertEqual(processed.shape, (64, 128))
        self.assertTrue(np.isfinite(processed).all())
        self.assertGreaterEqual(float(processed.min()), 0.0)
        self.assertLessEqual(float(processed.max()), 1.0)


def run_webcam_demo():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap.release()
        landmarker.close()
        raise RuntimeError("Could not open webcam. Check camera permissions.")

    timestamp_ms = 0
    try:
        print("Eye preprocessing test started. Press q to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read webcam frame.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += 33

            if result.face_landmarks:
                face = result.face_landmarks[0]
                height, width = frame.shape[:2]

                for indices, window_name in [
                    (LEFT_EYE, "Left Eye Crop"),
                    (RIGHT_EYE, "Right Eye Crop"),
                ]:
                    points = [
                        (int(face[i].x * width), int(face[i].y * height))
                        for i in indices
                    ]
                    crop = extract_eye_region(frame, points)
                    processed = preprocess_eye(crop)

                    if crop is not None and processed is not None:
                        x_min = min(point[0] for point in points)
                        y_min = min(point[1] for point in points)
                        x_max = max(point[0] for point in points)
                        y_max = max(point[1] for point in points)
                        cv2.rectangle(
                            frame,
                            (x_min - 8, y_min - 8),
                            (x_max + 8, y_max + 8),
                            (0, 255, 0),
                            2,
                        )
                        display_eye = (processed * 255).astype("uint8")
                        cv2.imshow(window_name, display_eye)

            cv2.imshow("SmartEye - Preprocessing Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_demo()
