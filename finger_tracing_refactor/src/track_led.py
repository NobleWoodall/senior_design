"""
LED Tracker - Combined red HSV + brightness tracking
Detects bright red LEDs while filtering out white ceiling lights
"""

import cv2
import numpy as np


class LEDTracker:
    """Track a bright red LED using combined HSV color + brightness filtering"""

    def __init__(self, hsv_low, hsv_high, brightness_threshold, morph_kernel=3, min_area=5):
        """
        Initialize LED tracker.

        Args:
            hsv_low: Lower HSV threshold (H, S, V) for red detection
            hsv_high: Upper HSV threshold (H, S, V) for red detection
            brightness_threshold: Minimum brightness (0-255) for detection
            morph_kernel: Kernel size for morphological operations
            min_area: Minimum blob area in pixels
        """
        self.hsv_low = np.array(hsv_low, dtype=np.uint8)
        self.hsv_high = np.array(hsv_high, dtype=np.uint8)
        self.brightness_threshold = brightness_threshold
        self.morph_kernel_size = morph_kernel
        self.min_area = min_area

    def track(self, frame):
        """
        Track LED in frame using combined red HSV + brightness.

        Args:
            frame: BGR color frame from camera

        Returns:
            (x, y) tuple of LED centroid, or None if not detected
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Red HSV mask - red wraps around in HSV color space
        # Check both low red (0-20) and high red (160-180)
        hsv_mask1 = cv2.inRange(hsv, self.hsv_low, self.hsv_high)
        hsv_mask2 = cv2.inRange(hsv, np.array([160, 100, 150]), np.array([180, 255, 255]))
        hsv_mask = cv2.bitwise_or(hsv_mask1, hsv_mask2)

        # Brightness mask - only very bright pixels
        _, brightness_mask = cv2.threshold(gray, self.brightness_threshold, 255, cv2.THRESH_BINARY)

        # Combine: pixel must be BOTH red AND bright
        mask = cv2.bitwise_and(hsv_mask, brightness_mask)

        # Morphological operations to clean up noise
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find LED position
        return self._find_led_centroid(mask)

    def _find_led_centroid(self, mask):
        """Find LED centroid from binary mask"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Filter by area and find largest
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_area]

        if not valid_contours:
            return None

        # Get largest contour (LED should be brightest blob)
        largest = max(valid_contours, key=cv2.contourArea)

        # Get centroid
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return (cx, cy)

    def close(self):
        """Cleanup resources (no-op for LED tracker, here for API consistency)"""
        pass
