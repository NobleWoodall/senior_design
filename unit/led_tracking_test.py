"""
LED Tracking Test - Standalone test for tracking a bright LED
Tests both HSV-based and brightness-based tracking methods
"""

import sys
import os
import cv2
import numpy as np

# Add parent directory to path for RealSense imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'finger_tracing_refactor', 'src'))

from io_rs import RealSenseIO


class LEDTracker:
    """Test different LED tracking methods"""

    def __init__(self):
        # Tracking mode: 'combined' (red HSV + brightness), 'hsv', or 'brightness'
        self.mode = 'combined'

        # HSV thresholds for red LED - wider range to catch all reds
        self.hsv_low = np.array([4, 0, 227])    # Lower saturation, lower value to catch more reds
        self.hsv_high = np.array([35, 255, 255])  # Wider hue range (0-20 catches orange-reds)

        # Brightness threshold (0-255) - only very bright objects
        self.brightness_threshold = 141

        # Morphological operations
        self.morph_kernel_size = 3
        self.min_area = 20

        # Display settings
        self.show_mask = True
        self.show_original = True

    def track_combined(self, frame):
        """Track LED using BOTH red HSV AND brightness - must be red AND bright"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Red HSV mask - wider range to catch all reds
        # Red wraps around in HSV, so we need two ranges
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

        return mask

    def track_hsv(self, frame):
        """Track LED using HSV color range only"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for red LED
        # Note: Red wraps around in HSV, so we need two ranges
        mask1 = cv2.inRange(hsv, self.hsv_low, self.hsv_high)
        mask2 = cv2.inRange(hsv, np.array([160, 100, 150]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations to clean up noise
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def track_brightness(self, frame):
        """Track LED using brightness/intensity only"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold for very bright pixels
        _, mask = cv2.threshold(gray, self.brightness_threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up noise
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def find_led_position(self, mask):
        """Find LED position from binary mask"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Filter by area and find largest
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_area]

        if not valid_contours:
            return None

        # Get largest contour (LED should be brightest/most saturated)
        largest = max(valid_contours, key=cv2.contourArea)

        # Get centroid
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return (cx, cy)


def main():
    print("="*60)
    print("LED Tracking Test")
    print("="*60)
    print("\nCONTROLS:")
    print("  'q' or ESC - Quit")
    print("  'm' - Cycle mode (Combined / HSV / Brightness)")
    print("\n  Combined Mode (Default - Recommended):")
    print("    Detects pixels that are BOTH red AND bright")
    print("    Filters out white ceiling lights!")
    print("\n  Adjustment Controls:")
    print("    'h'/'H' - Decrease/Increase Hue max (red range)")
    print("    's'/'S' - Decrease/Increase Saturation min")
    print("    'v'/'V' - Decrease/Increase Value min")
    print("    'b'/'B' - Decrease/Increase brightness threshold")
    print("    'k'/'K' - Decrease/Increase kernel size")
    print("    'a'/'A' - Decrease/Increase min area")
    print("\nStarting camera...")
    print("="*60 + "\n")

    # Initialize camera
    camera = RealSenseIO(
        width=1920,
        height=1080,
        fps=30,
        depth_width=640,
        depth_height=480,
        depth_fps=30,
        use_auto_exposure=False,
        exposure=30  # Low exposure makes LED stand out
    )

    try:
        camera.start()
        print("Camera started successfully!")

        # Warm up camera
        print("Warming up camera...")
        for i in range(10):
            color, _, _ = camera.get_aligned()
            if color is None:
                raise RuntimeError("Failed to get camera frames")

        print("Camera ready!\n")

        tracker = LEDTracker()

        # Create windows
        cv2.namedWindow("LED Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("LED Tracking", 1280, 480)

        while True:
            # Get frame
            color, _, _ = camera.get_aligned()
            if color is None:
                continue

            # Track LED
            if tracker.mode == 'combined':
                mask = tracker.track_combined(color)
            elif tracker.mode == 'hsv':
                mask = tracker.track_hsv(color)
            else:
                mask = tracker.track_brightness(color)

            # Find LED position
            led_pos = tracker.find_led_position(mask)

            # Create visualization
            vis_original = color.copy()
            vis_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Draw LED position
            if led_pos is not None:
                # Draw on original
                cv2.circle(vis_original, led_pos, 20, (0, 255, 0), 2)
                cv2.circle(vis_original, led_pos, 5, (0, 255, 0), -1)
                cv2.drawMarker(vis_original, led_pos, (255, 0, 255),
                              cv2.MARKER_CROSS, 30, 2)

                # Draw on mask
                cv2.circle(vis_mask, led_pos, 20, (0, 255, 0), 2)
                cv2.circle(vis_mask, led_pos, 5, (0, 255, 0), -1)

                # Show coordinates
                coord_text = f"LED: ({led_pos[0]}, {led_pos[1]})"
                cv2.putText(vis_original, coord_text, (led_pos[0] + 25, led_pos[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Add info overlay
            mode_text = f"Mode: {tracker.mode.upper()}"
            if tracker.mode == 'combined':
                mode_text += " (Red + Bright)"
            cv2.putText(vis_original, mode_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show settings
            hsv_text = f"HSV: H[{tracker.hsv_low[0]}-{tracker.hsv_high[0]}] S[{tracker.hsv_low[1]}-255] V[{tracker.hsv_low[2]}-255]"
            cv2.putText(vis_original, hsv_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            bright_text = f"Brightness: {tracker.brightness_threshold} | Kernel: {tracker.morph_kernel_size} | Min Area: {tracker.min_area}"
            cv2.putText(vis_original, bright_text, (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Detection status
            status = "DETECTED" if led_pos is not None else "NO DETECTION"
            status_color = (0, 255, 0) if led_pos is not None else (0, 0, 255)
            cv2.putText(vis_original, status, (10, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # Combine views side by side
            combined = np.hstack([vis_original, vis_mask])

            # Show
            cv2.imshow("LED Tracking", combined)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Quit
                break
            elif key == ord('m'):  # Cycle mode
                modes = ['combined', 'hsv', 'brightness']
                current_idx = modes.index(tracker.mode)
                tracker.mode = modes[(current_idx + 1) % len(modes)]
                print(f"Switched to {tracker.mode.upper()} mode")

            # HSV controls
            elif key == ord('h'):
                tracker.hsv_high[0] = max(0, tracker.hsv_high[0] - 5)
                print(f"Hue max: {tracker.hsv_high[0]}")
            elif key == ord('H'):
                tracker.hsv_high[0] = min(180, tracker.hsv_high[0] + 5)
                print(f"Hue max: {tracker.hsv_high[0]}")
            elif key == ord('s'):
                tracker.hsv_low[1] = max(0, tracker.hsv_low[1] - 10)
                print(f"Saturation min: {tracker.hsv_low[1]}")
            elif key == ord('S'):
                tracker.hsv_low[1] = min(255, tracker.hsv_low[1] + 10)
                print(f"Saturation min: {tracker.hsv_low[1]}")
            elif key == ord('v'):
                tracker.hsv_low[2] = max(0, tracker.hsv_low[2] - 10)
                print(f"Value min: {tracker.hsv_low[2]}")
            elif key == ord('V'):
                tracker.hsv_low[2] = min(255, tracker.hsv_low[2] + 10)
                print(f"Value min: {tracker.hsv_low[2]}")

            # Brightness controls
            elif key == ord('b'):
                tracker.brightness_threshold = max(0, tracker.brightness_threshold - 10)
                print(f"Brightness threshold: {tracker.brightness_threshold}")
            elif key == ord('B'):
                tracker.brightness_threshold = min(255, tracker.brightness_threshold + 10)
                print(f"Brightness threshold: {tracker.brightness_threshold}")
            elif key == ord('k'):
                tracker.morph_kernel_size = max(1, tracker.morph_kernel_size - 2)
                print(f"Kernel size: {tracker.morph_kernel_size}")
            elif key == ord('K'):
                tracker.morph_kernel_size = min(15, tracker.morph_kernel_size + 2)
                print(f"Kernel size: {tracker.morph_kernel_size}")
            elif key == ord('a'):
                tracker.min_area = max(5, tracker.min_area - 5)
                print(f"Min area: {tracker.min_area}")
            elif key == ord('A'):
                tracker.min_area = min(500, tracker.min_area + 5)
                print(f"Min area: {tracker.min_area}")

        print("\nFinal Settings:")
        print(f"Mode: {tracker.mode}")
        print("HSV Settings:")
        print(f"  HSV Low:  {tracker.hsv_low}")
        print(f"  HSV High: {tracker.hsv_high}")
        print("Brightness Settings:")
        print(f"  Brightness Threshold: {tracker.brightness_threshold}")
        print(f"  Kernel Size: {tracker.morph_kernel_size}")
        print(f"  Min Area: {tracker.min_area}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nCleaning up...")
        camera.stop()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()
