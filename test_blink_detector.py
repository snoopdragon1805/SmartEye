import time

from blink_detector import BlinkDetector


detector = BlinkDetector()

print("Initial:")
print(detector.get_statistics())

# Eyes open
detector.update(0.26)

# Eyes close
detector.update(0.10)

time.sleep(0.15)

# Eyes open again
blink_detected = detector.update(0.26)

print("\nAfter blink:")
print("Blink detected:", blink_detected)
print(detector.get_statistics())
