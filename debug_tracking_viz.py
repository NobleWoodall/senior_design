"""
Debug Tracking Visualization
Shows side-by-side:
- Left: Raw camera view with tracked point
- Right: Stereo display view with transformed coordinates

This helps visualize coordinate transformation and calibrate alignment.
"""

import sys
import os
import cv2
import numpy as np
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'finger_tracing_refactor', 'src'))

from io_rs import RealSenseIO
from track_led import LEDTracker
from track_mp import MediaPipeTracker
from spiral_3d import Spiral3D
from config import AppConfig


def draw_crosshair(img, x, y, color, size=30, thickness=2):
    """Draw crosshair at position"""
    x, y = int(x), int(y)
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)
    cv2.circle(img, (x, y), size//2, color, thickness)


def draw_info_text(img, lines, start_y=30):
    """Draw multi-line text info"""
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, start_y + i*25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    print("="*80)
    print("DEBUG TRACKING VISUALIZATION")
    print("="*80)
    print("\nShows:")
    print("  LEFT: Raw camera view (1920x1080) with tracked point in RED")
    print("  RIGHT: Stereo display view (3840x1080) with transformed point in MAGENTA")
    print("\nControls:")
    print("  'q' or ESC - Quit")
    print("  'm' - Toggle tracking method (MediaPipe / LED)")
    print("  's' - Toggle spiral visibility")
    print("  'f' - Toggle coordinate flip")
    print("  '+'/'-' - Adjust finger_scale_x")
    print("  '['/']' - Adjust finger_scale_y")
    print("\n" + "="*80)

    # Load config
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = AppConfig.from_dict(cfg_dict)

    # Initialize camera
    rsio = RealSenseIO(
        cfg.camera.width, cfg.camera.height, cfg.camera.fps,
        cfg.camera.depth_width, cfg.camera.depth_height, cfg.camera.depth_fps,
        cfg.camera.use_auto_exposure, cfg.camera.exposure
    )
    rsio.start()
    print("\n✓ Camera started")

    # Initialize trackers
    mp_tracker = MediaPipeTracker(
        cfg.mediapipe.model_complexity,
        cfg.mediapipe.detection_confidence,
        cfg.mediapipe.tracking_confidence,
        cfg.mediapipe.ema_alpha
    )
    led_tracker = LEDTracker(
        tuple(cfg.led.hsv_low), tuple(cfg.led.hsv_high),
        cfg.led.brightness_threshold, cfg.led.morph_kernel, cfg.led.min_area
    )

    # Current tracker
    use_led = True  # Start with LED
    tracker = led_tracker
    tracker_name = "LED"

    # Initialize spiral
    spiral = Spiral3D(
        cfg.spiral.a, cfg.spiral.b,
        cfg.spiral.turns, cfg.spiral.theta_step,
        target_depth_m=cfg.stereo_3d.target_depth_m,
        disparity_offset_px=cfg.stereo_3d.disparity_offset_px
    )
    print(f"✓ Spiral created: {spiral.disparity_px:.1f}px disparity")

    # Settings (mutable for real-time adjustment)
    settings = {
        'finger_scale_x': cfg.stereo_3d.finger_scale_x,
        'finger_scale_y': cfg.stereo_3d.finger_scale_y,
        'flip_x': cfg.stereo_3d.flip_x,
        'flip_y': cfg.stereo_3d.flip_y,
        'show_spiral': True
    }

    # Get first frame to determine dimensions
    color0, _, _ = rsio.get_aligned()
    if color0 is None:
        raise RuntimeError("Failed to get camera frame")

    cam_h, cam_w = color0.shape[:2]
    scale_x = 1920 / cam_w
    scale_y = 1080 / cam_h

    print(f"✓ Camera: {cam_w}x{cam_h}")
    print(f"✓ Scale factors: {scale_x:.3f}x, {scale_y:.3f}x")
    print("\nStarting visualization...\n")

    # Create window
    cv2.namedWindow("Debug Tracking Visualization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Debug Tracking Visualization", 1920, 540)  # Half height for side-by-side

    try:
        while True:
            # Get camera frame
            color, depth, t_now = rsio.get_aligned()
            if color is None:
                continue

            # === LEFT SIDE: RAW CAMERA VIEW ===
            camera_view = color.copy()

            # Track in camera space
            pt = tracker.track(color)

            if pt is not None:
                x_cam, y_cam = pt

                # Draw on camera view (RED = raw tracking)
                draw_crosshair(camera_view, x_cam, y_cam, (0, 0, 255), size=40, thickness=3)
                cv2.putText(camera_view, f"Camera: ({x_cam:.0f}, {y_cam:.0f})",
                           (int(x_cam) + 50, int(y_cam) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # === TRANSFORM TO DISPLAY SPACE ===

                # Step 1: Scale to display coordinates
                x = x_cam * scale_x
                y = y_cam * scale_y

                # Step 2: FOV adjustment scaling (center-based)
                cx = 1920 / 2.0
                cy = 1080 / 2.0
                x = cx + (x - cx) * settings['finger_scale_x']
                y = cy + (y - cy) * settings['finger_scale_y']

                # Step 3: Apply coordinate flipping
                if settings['flip_x']:
                    x = 1920 - x
                if settings['flip_y']:
                    y = 1080 - y

            else:
                x, y = None, None

            # === RIGHT SIDE: STEREO DISPLAY VIEW ===
            if settings['show_spiral']:
                stereo_view = spiral.draw_stereo(
                    color_bgr=tuple(cfg.spiral.color_bgr),
                    thickness=cfg.spiral.line_thickness
                )
            else:
                # Black background if no spiral
                stereo_view = np.zeros((1080, 3840, 3), dtype=np.uint8)

            # Draw transformed point on stereo view (MAGENTA)
            if pt is not None and x is not None:
                spiral.draw_point_on_spiral(stereo_view, x, y, color=(255, 0, 255), radius=20)

                # Also draw center reference point
                cv2.circle(stereo_view, (1920//2, 1080//2), 10, (0, 255, 255), 2)  # Center dot
                cv2.circle(stereo_view, (1920//2 + 1920, 1080//2), 10, (0, 255, 255), 2)  # Center dot (right eye)

            # === COMBINE VIEWS SIDE-BY-SIDE ===
            # Resize camera view to half width for comparison
            camera_view_resized = cv2.resize(camera_view, (960, 540))
            stereo_view_resized = cv2.resize(stereo_view, (1920, 540))

            # Stack horizontally
            combined = np.hstack([camera_view_resized, stereo_view_resized])

            # === OVERLAY INFO ===
            info_lines = [
                f"Tracker: {tracker_name}",
                f"Scale: {settings['finger_scale_x']:.2f}x, {settings['finger_scale_y']:.2f}x",
                f"Flip: X={settings['flip_x']}, Y={settings['flip_y']}",
            ]

            if pt is not None:
                info_lines.append(f"Camera XY: ({x_cam:.0f}, {y_cam:.0f})")
                info_lines.append(f"Display XY: ({x:.0f}, {y:.0f})")
            else:
                info_lines.append("NO TRACKING")

            draw_info_text(combined, info_lines, start_y=30)

            # Labels
            cv2.putText(combined, "RAW CAMERA VIEW", (10, combined.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined, "STEREO DISPLAY VIEW", (970, combined.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Show
            cv2.imshow("Debug Tracking Visualization", combined)

            # === KEYBOARD CONTROLS ===
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Quit
                break

            elif key == ord('m'):  # Toggle tracker
                use_led = not use_led
                tracker = led_tracker if use_led else mp_tracker
                tracker_name = "LED" if use_led else "MediaPipe"
                print(f"\n→ Switched to {tracker_name} tracking")

            elif key == ord('s'):  # Toggle spiral
                settings['show_spiral'] = not settings['show_spiral']
                print(f"\n→ Spiral: {'ON' if settings['show_spiral'] else 'OFF'}")

            elif key == ord('f'):  # Toggle flip X
                settings['flip_x'] = not settings['flip_x']
                print(f"\n→ Flip X: {settings['flip_x']}")

            elif key == ord('F'):  # Toggle flip Y
                settings['flip_y'] = not settings['flip_y']
                print(f"\n→ Flip Y: {settings['flip_y']}")

            elif key == ord('=') or key == ord('+'):  # Increase scale X
                settings['finger_scale_x'] += 0.05
                print(f"\n→ Scale X: {settings['finger_scale_x']:.2f}")

            elif key == ord('-') or key == ord('_'):  # Decrease scale X
                settings['finger_scale_x'] = max(0.1, settings['finger_scale_x'] - 0.05)
                print(f"\n→ Scale X: {settings['finger_scale_x']:.2f}")

            elif key == ord(']'):  # Increase scale Y
                settings['finger_scale_y'] += 0.05
                print(f"\n→ Scale Y: {settings['finger_scale_y']:.2f}")

            elif key == ord('['):  # Decrease scale Y
                settings['finger_scale_y'] = max(0.1, settings['finger_scale_y'] - 0.05)
                print(f"\n→ Scale Y: {settings['finger_scale_y']:.2f}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n\n" + "="*80)
        print("FINAL SETTINGS:")
        print("="*80)
        print(f"  finger_scale_x: {settings['finger_scale_x']:.2f}")
        print(f"  finger_scale_y: {settings['finger_scale_y']:.2f}")
        print(f"  flip_x: {settings['flip_x']}")
        print(f"  flip_y: {settings['flip_y']}")
        print("\nCopy these to config.yaml if they work well!")
        print("="*80)

        # Cleanup
        mp_tracker.close()
        rsio.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
