"""
Calibration routine for finger/LED tracking.

User traces a moving dot that follows the spiral path perfectly.
System collects (detected_position, ground_truth_position) pairs
and computes affine transformation to correct camera misalignment.
"""
import os
import time
import math
import numpy as np
import cv2
from typing import List, Tuple

from .config import AppConfig
from .io_rs import RealSenseIO
from .spiral_3d import Spiral3D
from .track_mp import MediaPipeTracker
from .track_led import LEDTracker
from .calibration_utils import compute_calibration, save_calibration, validate_calibration, apply_calibration


class CalibrationRunner:
    """Runs calibration routine with moving dot on stereo 3D spiral."""

    # XReal glasses dimensions
    FULL_WIDTH = 3840
    FULL_HEIGHT = 1080
    EYE_WIDTH = FULL_WIDTH // 2
    EYE_HEIGHT = FULL_HEIGHT

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def _draw_stereo_text(self, frame: np.ndarray, text: str, color: tuple, y_offset: int = 40):
        """Draw text on both eyes of stereo frame."""
        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, text, (self.EYE_WIDTH + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    def _get_dot_position_at_time(self, spiral: Spiral3D, t_elapsed: float,
                                  revolutions_per_sec: float) -> Tuple[float, float]:
        """
        Get dot position along spiral at given time.

        Args:
            spiral: Spiral3D object
            t_elapsed: Time elapsed since start (seconds)
            revolutions_per_sec: Speed in revolutions per second

        Returns:
            (x, y) in display coordinates (1920x1080 space)
        """
        # Calculate normalized position along spiral (0.0 to 1.0)
        progress = (t_elapsed * revolutions_per_sec) % 1.0  # Wrap around

        # Find corresponding point on spiral
        idx = int(progress * (len(spiral.spiral_points) - 1))
        idx = max(0, min(idx, len(spiral.spiral_points) - 1))

        # Get spiral point in relative coordinates
        x_rel = spiral.spiral_points[idx, 0]
        y_rel = spiral.spiral_points[idx, 1]

        # Transform to display coordinates
        cx = self.EYE_WIDTH / 2.0
        cy = self.EYE_HEIGHT / 2.0
        x = x_rel + cx
        y = y_rel + cy

        return x, y

    def test_calibration(self, rsio: RealSenseIO, spiral: Spiral3D, tracker,
                        calibration_matrix: np.ndarray, scale_x: float, scale_y: float) -> str:
        """
        Test calibration by showing tracked position with calibration applied.
        User can trace the spiral and see if calibration improves accuracy.

        Returns:
            "keep" to keep calibration, "redo" to redo, "cancel" to abort
        """
        print("\n=== TEST CALIBRATION ===")
        print("Trace the spiral to see if calibration is accurate")
        print(f"\nUsing calibration matrix:")
        print(calibration_matrix)
        print("\nControls:")
        print("  K = Keep this calibration and finish")
        print("  R = Redo calibration")
        print("  ESC = Cancel and exit")

        MAX_JUMP_PX = int(self.cfg.experiment.max_jump_px)
        last_xy = None

        while True:
            color, _, t_now = rsio.get_aligned()
            if color is None:
                continue

            view = spiral.draw_stereo(
                color_bgr=tuple(self.cfg.spiral.color_bgr),
                thickness=self.cfg.spiral.line_thickness
            )

            # Track finger/LED
            pt = tracker.track(color)

            # Apply jump gate (camera coordinates)
            if pt is not None:
                x_cam, y_cam = pt
                if last_xy is not None:
                    dx = x_cam - last_xy[0]
                    dy = y_cam - last_xy[1]
                    if (dx*dx + dy*dy)**0.5 > MAX_JUMP_PX:
                        pt = None
                if pt is not None:
                    last_xy = (x_cam, y_cam)
            else:
                last_xy = None

            if pt is not None:
                x_cam, y_cam = pt

                # Apply calibration transform
                x_cam_cal, y_cam_cal = apply_calibration(calibration_matrix, x_cam, y_cam)

                # Debug: Print first frame transformation (only once)
                if not hasattr(self, '_debug_printed'):
                    print(f"\n[DEBUG] First frame transformation:")
                    print(f"  Raw camera: ({x_cam:.1f}, {y_cam:.1f})")
                    print(f"  After calibration: ({x_cam_cal:.1f}, {y_cam_cal:.1f})")
                    print(f"  Offset: ({x_cam_cal - x_cam:.1f}, {y_cam_cal - y_cam:.1f})")
                    self._debug_printed = True

                # Scale to display coordinates
                x = x_cam_cal * scale_x
                y = y_cam_cal * scale_y

                # Apply FOV adjustment scaling
                cx = self.EYE_WIDTH / 2.0
                cy = self.EYE_HEIGHT / 2.0
                x = cx + (x - cx) * self.cfg.stereo_3d.finger_scale_x
                y = cy + (y - cy) * self.cfg.stereo_3d.finger_scale_y

                # Find nearest spiral point to show error
                xs, ys, s_sp, _ = spiral.nearest_point(x, y)
                err = math.hypot(x - xs, y - ys)

                # Draw calibrated finger position
                spiral.draw_point_on_spiral(view, x, y, color=(255, 0, 255), radius=20)

                # Draw nearest spiral point
                spiral.draw_point_on_spiral(view, xs, ys, color=(0, 255, 0), radius=8)

                self._draw_stereo_text(view, f"TEST MODE - Error: {err:.1f}px", (0, 255, 255), 40)
                self._draw_stereo_text(view, "Purple = Your finger | Green = Nearest spiral point", (200, 200, 200), 80)
            else:
                self._draw_stereo_text(view, "TEST MODE - No tracking", (0, 0, 255), 40)

            self._draw_stereo_text(view, "K=Keep | R=Redo | ESC=Cancel", (200, 200, 200), 120)

            cv2.imshow("Calibration", view)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                return "cancel"
            elif key == ord('k') or key == ord('K'):
                return "keep"
            elif key == ord('r') or key == ord('R'):
                return "redo"

    def run_calibration(self, method: str = "hsv", initial_matrix: np.ndarray = None) -> bool:
        """
        Run calibration routine.

        Args:
            method: Tracking method to use ("mp" or "hsv")
            initial_matrix: Optional initial calibration matrix to refine

        Returns:
            True if calibration succeeded, False otherwise, or "redo" to refine
        """
        cfg = self.cfg

        # Track if we're refining an existing calibration
        is_refinement = initial_matrix is not None
        if is_refinement:
            print("\n[Calibration] REFINEMENT MODE - Using existing calibration as baseline")
            print("Existing matrix:")
            print(initial_matrix)

        # Initialize camera
        print("\n=== Starting Calibration ===")
        print(f"Tracking method: {method.upper()}")

        rsio = RealSenseIO(
            cfg.camera.width, cfg.camera.height, cfg.camera.fps,
            cfg.camera.depth_width, cfg.camera.depth_height, cfg.camera.depth_fps,
            cfg.camera.use_auto_exposure, cfg.camera.exposure
        )
        rsio.start()

        # Initialize tracker
        if method == "mp":
            tracker = MediaPipeTracker(
                cfg.mediapipe.model_complexity,
                cfg.mediapipe.detection_confidence,
                cfg.mediapipe.tracking_confidence,
                cfg.mediapipe.ema_alpha
            )
        elif method == "hsv":
            tracker = LEDTracker(
                tuple(cfg.led.hsv_low),
                tuple(cfg.led.hsv_high),
                cfg.led.brightness_threshold,
                cfg.led.morph_kernel,
                cfg.led.min_area
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Create spiral
        spiral = Spiral3D(
            cfg.spiral.a, cfg.spiral.b,
            cfg.spiral.turns, cfg.spiral.theta_step,
            target_depth_m=cfg.stereo_3d.target_depth_m,
            disparity_offset_px=cfg.stereo_3d.disparity_offset_px
        )

        # Get camera frame to determine scaling
        color0, _, _ = rsio.get_aligned()
        if color0 is None:
            print("[Calibration] Error: Failed to read from camera")
            rsio.stop()
            return False

        cam_h, cam_w = color0.shape[:2]
        scale_x = self.EYE_WIDTH / cam_w
        scale_y = self.EYE_HEIGHT / cam_h

        print(f"\nCamera: {cam_w}x{cam_h}")
        print(f"Display: {self.FULL_WIDTH}x{self.FULL_HEIGHT} (side-by-side)")
        print(f"Scale: {scale_x:.2f}x, {scale_y:.2f}x")

        # Calibration parameters
        countdown_sec = cfg.calibration.countdown_sec
        num_traces = cfg.calibration.num_traces
        revolutions_per_sec = cfg.calibration.dot_speed_revolutions_per_sec
        trace_duration = num_traces / revolutions_per_sec

        print(f"\nCalibration settings:")
        print(f"  Countdown: {countdown_sec}s")
        print(f"  Number of traces: {num_traces}")
        print(f"  Dot speed: {revolutions_per_sec:.2f} revolutions/sec")
        print(f"  Total duration: {trace_duration:.1f}s")

        # Data collection
        src_points = []  # Detected finger positions (camera coords)
        dst_points = []  # Ground truth dot positions (display coords, scaled back to camera)

        MAX_JUMP_PX = int(cfg.experiment.max_jump_px)
        last_xy = None

        try:
            # Create window
            cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Calibration", self.FULL_WIDTH, self.FULL_HEIGHT)
            print("\nDrag window to XReal display and press 'f' for fullscreen")

            # Preview phase - show starting position
            print(f"\n=== PREVIEW PHASE ===")
            if is_refinement:
                print("Refinement mode: The purple dot shows your position WITH current calibration applied")
                print("Position yourself at the starting dot (center of spiral)")
            else:
                print("Position yourself to see the starting dot (center of spiral)")
            print("Press SPACE when ready to start countdown, or ESC to cancel")

            preview_ready = False
            while not preview_ready:
                color, _, t_now = rsio.get_aligned()
                if color is None:
                    continue

                view = spiral.draw_stereo(
                    color_bgr=tuple(cfg.spiral.color_bgr),
                    thickness=cfg.spiral.line_thickness
                )

                # Show starting dot position (t=0)
                dot_x, dot_y = self._get_dot_position_at_time(spiral, 0.0, revolutions_per_sec)
                spiral.draw_point_on_spiral(view, dot_x, dot_y, color=(0, 255, 255), radius=15)

                # Track and show current finger position
                pt = tracker.track(color)
                if pt is not None:
                    x_cam, y_cam = pt

                    # Apply existing calibration if refining
                    if initial_matrix is not None:
                        x_cam, y_cam = apply_calibration(initial_matrix, x_cam, y_cam)

                    x_display = x_cam * scale_x
                    y_display = y_cam * scale_y
                    spiral.draw_point_on_spiral(view, x_display, y_display, color=(255, 0, 255), radius=20)

                    status = "PREVIEW - Your finger is visible"
                    if initial_matrix is not None:
                        status += " (with current calibration)"
                    self._draw_stereo_text(view, status, (0, 255, 0), 40)
                else:
                    self._draw_stereo_text(view, "PREVIEW - No tracking", (0, 0, 255), 40)

                self._draw_stereo_text(view, "Yellow dot = Starting position", (200, 200, 200), 80)
                self._draw_stereo_text(view, "Press SPACE to start | ESC to cancel | F for fullscreen", (200, 200, 200), 120)

                cv2.imshow("Calibration", view)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC to cancel
                    print("\n[Calibration] Cancelled by user")
                    rsio.stop()
                    if method == "mp":
                        tracker.close()
                    cv2.destroyAllWindows()
                    return False
                elif key == ord(' '):  # SPACE to continue
                    preview_ready = True
                elif key == ord('f'):
                    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # Countdown phase
            print(f"\n=== COUNTDOWN PHASE ===")
            countdown_start = time.time()

            while True:
                color, _, t_now = rsio.get_aligned()
                if color is None:
                    continue

                view = spiral.draw_stereo(
                    color_bgr=tuple(cfg.spiral.color_bgr),
                    thickness=cfg.spiral.line_thickness
                )

                elapsed = time.time() - countdown_start
                remaining = max(0, countdown_sec - elapsed)

                if remaining > 0:
                    text = f"GET READY: {int(remaining) + 1}"
                    self._draw_stereo_text(view, text, (0, 255, 255), 40)
                    self._draw_stereo_text(view, "Trace the moving dot with your finger/LED", (200, 200, 200), 80)
                    self._draw_stereo_text(view, f"Duration: {trace_duration:.0f}s | Traces: {num_traces}", (200, 200, 200), 120)

                    # Show starting dot during countdown
                    dot_x, dot_y = self._get_dot_position_at_time(spiral, 0.0, revolutions_per_sec)
                    spiral.draw_point_on_spiral(view, dot_x, dot_y, color=(0, 255, 255), radius=15)
                else:
                    break

                cv2.imshow("Calibration", view)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC to cancel
                    print("\n[Calibration] Cancelled by user")
                    rsio.stop()
                    if method == "mp":
                        tracker.close()
                    cv2.destroyAllWindows()
                    return False
                elif key == ord('f'):
                    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # Collection phase
            print(f"\n=== COLLECTING DATA ===")
            collection_start = time.time()
            frame_count = 0
            tracked_count = 0

            while True:
                color, _, t_now = rsio.get_aligned()
                if color is None:
                    continue

                elapsed = time.time() - collection_start

                # Check if done
                if elapsed >= trace_duration:
                    break

                # Render spiral
                view = spiral.draw_stereo(
                    color_bgr=tuple(cfg.spiral.color_bgr),
                    thickness=cfg.spiral.line_thickness
                )

                # Get dot position
                dot_x, dot_y = self._get_dot_position_at_time(spiral, elapsed, revolutions_per_sec)

                # Draw moving target dot at spiral depth (YELLOW)
                spiral.draw_point_on_spiral(view, dot_x, dot_y, color=(0, 255, 255), radius=15)

                # Track finger/LED
                pt = tracker.track(color)

                # Apply jump gate (camera coordinates)
                if pt is not None:
                    x_cam, y_cam = pt
                    if last_xy is not None:
                        dx = x_cam - last_xy[0]
                        dy = y_cam - last_xy[1]
                        if (dx*dx + dy*dy)**0.5 > MAX_JUMP_PX:
                            pt = None
                    if pt is not None:
                        last_xy = (x_cam, y_cam)
                else:
                    last_xy = None

                # Collect data if tracking successful
                if pt is not None:
                    x_cam_original, y_cam_original = pt  # Store original for calibration data collection
                    x_cam, y_cam = pt

                    # Apply existing calibration if refining
                    if initial_matrix is not None:
                        x_cam, y_cam = apply_calibration(initial_matrix, x_cam, y_cam)

                    # Scale to display coordinates for visualization
                    x_display = x_cam * scale_x
                    y_display = y_cam * scale_y

                    # Draw tracked finger position (PURPLE/MAGENTA)
                    spiral.draw_point_on_spiral(view, x_display, y_display, color=(255, 0, 255), radius=20)

                    # Calculate and show error distance
                    error_px = math.hypot(x_display - dot_x, y_display - dot_y)

                    # Store calibration pair (camera coords for src, camera coords for dst)
                    # Convert dot position from display back to camera coordinates
                    dot_x_cam = dot_x / scale_x
                    dot_y_cam = dot_y / scale_y

                    # For refinement, use original raw position as source
                    # so we can compose the matrices later
                    src_points.append((x_cam_original, y_cam_original) if initial_matrix is not None else (x_cam, y_cam))
                    dst_points.append((dot_x_cam, dot_y_cam))
                    tracked_count += 1

                    # Draw status with error feedback
                    progress_pct = (elapsed / trace_duration) * 100
                    self._draw_stereo_text(view, f"CALIBRATING: {progress_pct:.0f}%", (0, 255, 0), 40)
                    self._draw_stereo_text(view, f"Points: {tracked_count} | Error: {error_px:.1f}px", (200, 200, 200), 80)
                    self._draw_stereo_text(view, "Yellow=Target | Purple=Your Finger", (200, 200, 200), 120)
                else:
                    # Draw status without tracking
                    progress_pct = (elapsed / trace_duration) * 100
                    self._draw_stereo_text(view, f"CALIBRATING: {progress_pct:.0f}%", (0, 255, 0), 40)
                    self._draw_stereo_text(view, f"Points: {tracked_count}", (200, 200, 200), 80)
                    self._draw_stereo_text(view, "NO TRACKING - Keep finger/LED visible", (0, 0, 255), 120)

                cv2.imshow("Calibration", view)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC to cancel
                    print("\n[Calibration] Cancelled by user")
                    rsio.stop()
                    if method == "mp":
                        tracker.close()
                    cv2.destroyAllWindows()
                    return False

                frame_count += 1

            # Compute calibration
            print(f"\n=== COMPUTING CALIBRATION ===")
            print(f"Total frames: {frame_count}")
            print(f"Points collected: {len(src_points)}")

            if len(src_points) < 10:
                print(f"[Calibration] Error: Not enough points collected (need at least 10, got {len(src_points)})")
                print("Try again with better tracking or slower dot speed")
                # Cleanup
                cv2.destroyAllWindows()
                rsio.stop()
                if method == "mp":
                    tracker.close()
                return False

            # Split data into train and test sets (80/20)
            n_train = int(len(src_points) * 0.8)
            train_src = src_points[:n_train]
            train_dst = dst_points[:n_train]
            test_src = src_points[n_train:]
            test_dst = dst_points[n_train:]

            # Compute calibration matrix
            matrix = compute_calibration(train_src, train_dst, use_ransac=True)

            if matrix is None:
                print("[Calibration] Failed to compute calibration matrix")
                # Cleanup
                cv2.destroyAllWindows()
                rsio.stop()
                if method == "mp":
                    tracker.close()
                return False

            # Validate on test set
            if len(test_src) > 0:
                print(f"\n=== VALIDATION ===")
                validation_results = validate_calibration(matrix, test_src, test_dst)
                print(f"Test set size: {validation_results['num_test_points']}")
                print(f"Mean error: {validation_results['mean_error_px']:.2f} px")
                print(f"Median error: {validation_results['median_error_px']:.2f} px")
                print(f"Max error: {validation_results['max_error_px']:.2f} px")

                if validation_results['mean_error_px'] > 50:
                    print("\n[Calibration] Warning: High validation error - calibration may be poor")
                    print("Consider testing and potentially redoing calibration")

            # Test calibration mode - let user decide to keep or redo
            test_result = self.test_calibration(rsio, spiral, tracker, matrix, scale_x, scale_y)

            # Cleanup resources
            cv2.destroyAllWindows()
            rsio.stop()
            if method == "mp":
                tracker.close()

            if test_result == "keep":
                # Save calibration
                output_dir = cfg.experiment.output_dir
                os.makedirs(output_dir, exist_ok=True)
                calibration_file = os.path.join(output_dir, cfg.calibration.calibration_file)

                save_calibration(matrix, calibration_file)

                print(f"\n=== CALIBRATION COMPLETE ===")
                print(f"Calibration saved to: {calibration_file}")
                print(f"Enable calibration in config.yaml:")
                print(f"  calibration:")
                print(f"    enabled: true")

                return True
            elif test_result == "redo":
                print("\n[Calibration] User chose to redo calibration")
                return "redo"  # Signal to redo
            else:  # cancel
                print("\n[Calibration] Cancelled by user during test")
                return False

        except Exception as e:
            print(f"\n[Calibration] Error: {e}")
            import traceback
            traceback.print_exc()

            try:
                cv2.destroyAllWindows()
                rsio.stop()
                if method == "mp":
                    tracker.close()
            except:
                pass

            return False
