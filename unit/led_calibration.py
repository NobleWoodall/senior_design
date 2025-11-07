"""
LED Calibration Tool

Interactive calibration script to determine optimal HSV and brightness thresholds
for LED tracking. Click on the LED in different positions to sample values, and
the tool will calculate the min/max ranges needed to capture all samples.

Controls:
- Click: Sample HSV and brightness at cursor position
- 'r': Reset all samples
- 's': Save settings to config.yaml
- 'q': Quit and show final settings
"""

import sys
import os
import cv2
import numpy as np
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from finger_tracing_refactor.src.io_rs import RealSenseIO


class LEDCalibrator:
    def __init__(self, camera):
        self.camera = camera
        self.samples = []  # List of (h, s, v, brightness) tuples
        self.window_name = "LED Calibration"
        self.current_frame = None
        self.current_hsv = None
        self.mouse_pos = (0, 0)

        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        self.mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Sample the pixel at click position
            self.sample_pixel(x, y)

    def sample_pixel(self, x, y):
        """Sample HSV and brightness values at given pixel"""
        if self.current_frame is None or self.current_hsv is None:
            return

        h, w = self.current_frame.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return

        # Get HSV values
        hsv_pixel = self.current_hsv[y, x]
        h_val, s_val, v_val = hsv_pixel

        # Get brightness (V channel value)
        brightness = v_val

        # Store sample
        self.samples.append((h_val, s_val, v_val, brightness))
        print(f"Sampled point {len(self.samples)}: H={h_val}, S={s_val}, V={v_val}, Brightness={brightness}")

    def calculate_ranges(self):
        """Calculate min/max ranges from all samples"""
        if not self.samples:
            return None

        h_vals = [s[0] for s in self.samples]
        s_vals = [s[1] for s in self.samples]
        v_vals = [s[2] for s in self.samples]
        brightness_vals = [s[3] for s in self.samples]

        # Calculate ranges with some margin
        h_min = max(0, min(h_vals) - 5)
        h_max = min(180, max(h_vals) + 5)
        s_min = max(0, min(s_vals) - 20)
        s_max = min(255, max(s_vals) + 20)
        v_min = max(0, min(v_vals) - 20)
        v_max = 255  # Always max for V
        brightness_threshold = max(0, min(brightness_vals) - 10)

        return {
            'hsv_low': [int(h_min), int(s_min), int(v_min)],
            'hsv_high': [int(h_max), int(s_max), int(v_max)],
            'brightness_threshold': int(brightness_threshold)
        }

    def draw_overlay(self, frame):
        """Draw UI overlay on frame"""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw crosshair at mouse position
        mx, my = self.mouse_pos
        if 0 <= mx < w and 0 <= my < h:
            cv2.line(vis, (mx - 10, my), (mx + 10, my), (0, 255, 0), 1)
            cv2.line(vis, (mx, my - 10), (mx, my + 10), (0, 255, 0), 1)

            # Show HSV values at cursor
            if self.current_hsv is not None:
                hsv_pixel = self.current_hsv[my, mx]
                text = f"H:{hsv_pixel[0]} S:{hsv_pixel[1]} V:{hsv_pixel[2]}"
                cv2.putText(vis, text, (mx + 15, my - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw sampled points
        for i, (h, s, v, b) in enumerate(self.samples):
            # Find approximate pixel location (we don't store it, so just show count)
            pass

        # Draw info panel
        panel_height = 120
        cv2.rectangle(vis, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.rectangle(vis, (0, 0), (w, panel_height), (0, 255, 0), 2)

        # Draw text info
        y_offset = 20
        cv2.putText(vis, "LED Calibration Tool", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(vis, f"Samples: {len(self.samples)}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(vis, "Click: Sample | 'r': Reset | 's': Save | 'q': Quit",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Show current ranges if we have samples
        if self.samples:
            y_offset += 25
            ranges = self.calculate_ranges()
            cv2.putText(vis, f"HSV Low: {ranges['hsv_low']}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y_offset += 20
            cv2.putText(vis, f"HSV High: {ranges['hsv_high']}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        return vis

    def reset_samples(self):
        """Clear all samples"""
        self.samples = []
        print("\nSamples reset!")

    def save_to_config(self, config_path):
        """Save calculated settings to config.yaml"""
        if not self.samples:
            print("No samples to save!")
            return False

        ranges = self.calculate_ranges()

        try:
            # Read existing config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Update LED settings
            if 'led' not in config:
                config['led'] = {}

            config['led']['hsv_low'] = ranges['hsv_low']
            config['led']['hsv_high'] = ranges['hsv_high']
            config['led']['brightness_threshold'] = ranges['brightness_threshold']

            # Write back to file
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            print(f"\nSettings saved to {config_path}")
            return True

        except Exception as e:
            print(f"Error saving config: {e}")
            return False

    def print_final_settings(self):
        """Print final recommended settings"""
        if not self.samples:
            print("\nNo samples collected!")
            return

        ranges = self.calculate_ranges()

        print("\n" + "="*60)
        print("LED CALIBRATION RESULTS")
        print("="*60)
        print(f"\nTotal samples collected: {len(self.samples)}")
        print("\nSample values:")
        for i, (h, s, v, b) in enumerate(self.samples, 1):
            print(f"  {i}. H={h:3d}, S={s:3d}, V={v:3d}, Brightness={b:3d}")

        print("\n" + "-"*60)
        print("RECOMMENDED SETTINGS:")
        print("-"*60)
        print(f"hsv_low: {ranges['hsv_low']}")
        print(f"hsv_high: {ranges['hsv_high']}")
        print(f"brightness_threshold: {ranges['brightness_threshold']}")

        print("\n" + "-"*60)
        print("CONFIG.YAML FORMAT:")
        print("-"*60)
        print("led:")
        print(f"  hsv_low: {ranges['hsv_low']}")
        print(f"  hsv_high: {ranges['hsv_high']}")
        print(f"  brightness_threshold: {ranges['brightness_threshold']}")
        print("="*60 + "\n")

    def run(self):
        """Main calibration loop"""
        print("\n" + "="*60)
        print("LED CALIBRATION TOOL")
        print("="*60)
        print("\nInstructions:")
        print("1. Turn on your LED")
        print("2. Move it around and click on it in different positions")
        print("3. Sample at least 5-10 different locations")
        print("4. Press 'q' when done to see results")
        print("\nControls:")
        print("  Click: Sample pixel")
        print("  'r': Reset samples")
        print("  's': Save to config.yaml")
        print("  'q': Quit and show results")
        print("="*60 + "\n")

        while True:
            # Get frame
            color, depth, ts = self.camera.get_aligned()
            if color is None:
                print("Failed to get frame")
                break

            self.current_frame = color
            self.current_hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

            # Draw overlay
            vis = self.draw_overlay(color)

            # Show frame
            cv2.imshow(self.window_name, vis)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # Quit
                break
            elif key == ord('r'):
                # Reset samples
                self.reset_samples()
            elif key == ord('s'):
                # Save to config
                config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
                self.save_to_config(config_path)

        # Clean up
        cv2.destroyAllWindows()

        # Print final results
        self.print_final_settings()


def main():
    print("Initializing camera...")

    # Camera settings - use high exposure to see the scene clearly
    camera = RealSenseIO(
        width=1920,
        height=1080,
        fps=30,
        depth_width=640,
        depth_height=480,
        depth_fps=30,
        use_auto_exposure=False,
        exposure=10  # Higher exposure for better visibility during calibration
    )

    camera.start()

    # Warmup
    print("Warming up camera...")
    for _ in range(10):
        camera.get_aligned()

    print("Camera ready!\n")

    # Run calibrator
    calibrator = LEDCalibrator(camera)
    calibrator.run()

    # Clean up
    camera.stop()


if __name__ == "__main__":
    main()
