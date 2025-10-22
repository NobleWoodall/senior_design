"""
XReal Spiral Tracing with Hand Tracking
Displays a static spiral on black background with real-time fingertip tracking dot.
Optimized for XReal 1 Pro glasses in 2D screen mirroring mode.
"""

import sys
import os
import cv2
import numpy as np
import yaml
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'finger_tracing_refactor', 'src'))

from io_rs import RealSenseIO
from track_mp import MediaPipeTracker
from spiral import Spiral


class XRealSpiralTracing:
    """Main application for XReal spiral tracing with hand tracking."""

    def __init__(self, config_path="config.yaml"):
        """
        Initialize the XReal spiral tracing application.

        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Display settings for XReal 1 Pro
        self.display_width = 1920
        self.display_height = 1080

        # Camera settings
        cam_cfg = self.cfg['camera']
        self.camera = RealSenseIO(
            width=cam_cfg['width'],
            height=cam_cfg['height'],
            fps=cam_cfg['fps'],
            depth_width=cam_cfg['depth_width'],
            depth_height=cam_cfg['depth_height'],
            depth_fps=cam_cfg['depth_fps'],
            use_auto_exposure=cam_cfg['use_auto_exposure'],
            exposure=cam_cfg['exposure']
        )

        # Hand tracking settings
        mp_cfg = self.cfg['mediapipe']
        self.tracker = MediaPipeTracker(
            model_complexity=mp_cfg['model_complexity'],
            det_conf=mp_cfg['detection_confidence'],
            trk_conf=mp_cfg['tracking_confidence'],
            ema_alpha=mp_cfg['ema_alpha']
        )

        # Spiral settings
        spiral_cfg = self.cfg['spiral']
        self.spiral = Spiral(
            width=self.display_width,
            height=self.display_height,
            a=spiral_cfg['a'],
            b=spiral_cfg['b'],
            turns=spiral_cfg['turns'],
            theta_step=spiral_cfg['theta_step']
        )

        # Visual settings
        self.spiral_color = tuple(spiral_cfg['color_bgr'])
        self.spiral_thickness = spiral_cfg['line_thickness']
        self.dot_color = (255, 0, 255)  # Bright magenta
        self.dot_radius = 12

        # State flags
        self.show_spiral = True
        self.show_dot = True
        self.running = False

        # Color presets for cycling
        self.color_presets = [
            (0, 255, 255),    # Cyan
            (0, 255, 0),      # Green
            (255, 255, 0),    # Yellow
            (255, 128, 0),    # Orange
            (255, 0, 255),    # Magenta
            (255, 255, 255),  # White
        ]
        self.current_color_idx = 0

    def start(self):
        """Initialize camera and hand tracking."""
        print("Starting RealSense camera...")
        self.camera.start()
        print("Camera started successfully!")
        self.running = True

    def stop(self):
        """Clean up resources."""
        print("Stopping...")
        self.running = False
        self.camera.stop()
        self.tracker.close()
        cv2.destroyAllWindows()
        print("Stopped.")

    def create_display_frame(self, finger_pos=None):
        """
        Create the display frame with spiral and tracking dot.

        Args:
            finger_pos: Tuple of (x, y) in camera coordinates, or None

        Returns:
            Display frame (1920x1080, BGR)
        """
        # Create black canvas
        frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)

        # Draw spiral if enabled
        if self.show_spiral:
            self.spiral.draw(frame, color=self.spiral_color, thickness=self.spiral_thickness)

        # Draw tracking dot if finger detected and enabled
        if self.show_dot and finger_pos is not None:
            # Scale finger position from camera resolution to display resolution
            cam_w = self.cfg['camera']['width']
            cam_h = self.cfg['camera']['height']

            x_scaled = int(finger_pos[0] * self.display_width / cam_w)
            y_scaled = int(finger_pos[1] * self.display_height / cam_h)

            # Draw outer glow effect
            cv2.circle(frame, (x_scaled, y_scaled), self.dot_radius + 4, self.dot_color, 2)
            # Draw solid dot
            cv2.circle(frame, (x_scaled, y_scaled), self.dot_radius, self.dot_color, -1)

        return frame

    def handle_keyboard(self, key):
        """
        Handle keyboard input.

        Args:
            key: Key code from cv2.waitKey()

        Returns:
            True to continue, False to quit
        """
        if key == ord('q') or key == 27:  # q or ESC
            return False
        elif key == ord('r'):  # Reset
            print("Reset")
        elif key == ord('+') or key == ord('='):  # Increase thickness
            self.spiral_thickness = min(self.spiral_thickness + 1, 20)
            print(f"Spiral thickness: {self.spiral_thickness}")
        elif key == ord('-') or key == ord('_'):  # Decrease thickness
            self.spiral_thickness = max(self.spiral_thickness - 1, 1)
            print(f"Spiral thickness: {self.spiral_thickness}")
        elif key == ord('c'):  # Cycle colors
            self.current_color_idx = (self.current_color_idx + 1) % len(self.color_presets)
            self.spiral_color = self.color_presets[self.current_color_idx]
            print(f"Spiral color: BGR{self.spiral_color}")
        elif key == ord('d'):  # Toggle dot
            self.show_dot = not self.show_dot
            print(f"Dot: {'ON' if self.show_dot else 'OFF'}")
        elif key == ord('s'):  # Toggle spiral
            self.show_spiral = not self.show_spiral
            print(f"Spiral: {'ON' if self.show_spiral else 'OFF'}")
        elif key == ord('['):  # Decrease dot size
            self.dot_radius = max(self.dot_radius - 1, 3)
            print(f"Dot radius: {self.dot_radius}")
        elif key == ord(']'):  # Increase dot size
            self.dot_radius = min(self.dot_radius + 1, 30)
            print(f"Dot radius: {self.dot_radius}")

        return True

    def run(self):
        """Main application loop."""
        # Create fullscreen window
        window_name = "XReal Spiral Tracing"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print("\n" + "="*70)
        print("  XReal Spiral Tracing - Hand Tracking")
        print("="*70)
        print("\nCONTROLS:")
        print("  'q' or ESC - Quit")
        print("  'r' - Reset")
        print("  '+/-' - Adjust spiral thickness")
        print("  '[/]' - Adjust dot size")
        print("  'c' - Cycle spiral colors")
        print("  'd' - Toggle dot visibility")
        print("  's' - Toggle spiral visibility")
        print("\nStarting in 3 seconds...")
        print("="*70 + "\n")

        time.sleep(3)

        fps_history = []
        last_time = time.time()

        try:
            while self.running:
                # Get frame from camera
                color_frame, depth_frame, timestamp = self.camera.get_aligned()

                if color_frame is None:
                    continue

                # Track fingertip
                finger_pos = self.tracker.track(color_frame)

                # Create display frame
                display_frame = self.create_display_frame(finger_pos)

                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - last_time) if current_time > last_time else 0
                last_time = current_time
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)

                # Draw FPS counter (optional, small text in corner)
                cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

                # Display
                cv2.imshow(window_name, display_frame)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    if not self.handle_keyboard(key):
                        break

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()


def main():
    """Entry point for the application."""
    import argparse

    parser = argparse.ArgumentParser(description="XReal Spiral Tracing with Hand Tracking")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    args = parser.parse_args()

    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        print("Looking for config.yaml in project root...")

        # Try to find config.yaml in parent directories
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        config_path = os.path.join(parent_dir, "config.yaml")

        if os.path.exists(config_path):
            print(f"Found config at: {config_path}")
            args.config = config_path
        else:
            print("Error: Could not find config.yaml")
            sys.exit(1)

    app = XRealSpiralTracing(args.config)
    app.start()
    app.run()


if __name__ == "__main__":
    main()
