
import cv2
import numpy as np


def extract_eye_region(frame, eye_points, padding=8):
    """
    Crop an eye region from a BGR webcam frame.

    Args:
        frame: Original OpenCV image.
        eye_points: List of (x, y) eye landmark coordinates.
        padding: Extra pixels around the eye.

    Returns:
        Cropped eye image, or None if the crop is invalid.
    """
    if frame is None or len(eye_points) < 3:
        return None

    height, width = frame.shape[:2]
    points = np.asarray(eye_points, dtype=np.int32)

    x_min = max(0, int(points[:, 0].min()) - padding)
    y_min = max(0, int(points[:, 1].min()) - padding)
    x_max = min(width, int(points[:, 0].max()) + padding + 1)
    y_max = min(height, int(points[:, 1].max()) + padding + 1)

    if x_max <= x_min or y_max <= y_min:
        return None

    return frame[y_min:y_max, x_min:x_max].copy()


def preprocess_eye(eye_crop, output_size=(128, 64)):
    """
    Convert a cropped eye image to grayscale, resize it,
    and normalise pixel values to [0, 1].
    """
    if eye_crop is None or eye_crop.size == 0:
        return None

    grayscale = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(
        grayscale,
        output_size,
        interpolation=cv2.INTER_AREA
    )

    normalised = resized.astype(np.float32) / 255.0

    return normalised


if __name__ == "__main__":
    # Synthetic image to test preprocessing without a webcam.
    sample_frame = np.zeros((240, 320, 3), dtype=np.uint8)

    # Draw a simple bright region representing an eye.
    cv2.ellipse(
        sample_frame,
        (160, 120),
        (45, 15),
        0,
        0,
        360,
        (220, 220, 220),
        -1
    )

    sample_points = [
        (115, 120),
        (135, 110),
        (150, 108),
        (205, 120),
        (185, 130),
        (150, 132)
    ]

    crop = extract_eye_region(sample_frame, sample_points)
    processed = preprocess_eye(crop)

    if processed is not None:
        print("Preprocessing test passed")
        print("Output shape:", processed.shape)
        print("Minimum pixel:", float(processed.min()))
        print("Maximum pixel:", float(processed.max()))
    else:
        print("Preprocessing test failed")
