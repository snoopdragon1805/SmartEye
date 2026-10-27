import unittest

from blink_detector import BlinkDetector


class BlinkDetectorTest(unittest.TestCase):
    def test_detects_completed_blink(self):
        detector = BlinkDetector()

        self.assertFalse(detector.update(0.26, 0.0))
        self.assertFalse(detector.update(0.10, 0.1))
        self.assertTrue(detector.update(0.26, 0.25))
        self.assertEqual(detector.get_statistics()["blink_count"], 1)

    def test_invalid_sample_breaks_an_open_closure(self):
        detector = BlinkDetector()
        detector.update(0.10, 0.0)
        self.assertFalse(detector.update(float("nan"), 0.1))
        self.assertFalse(detector.update(0.26, 1.0))
        self.assertEqual(detector.get_statistics()["blink_count"], 0)


if __name__ == "__main__":
    unittest.main()
