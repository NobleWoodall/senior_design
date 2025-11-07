import os
import json
import math
import time
import numpy as np
import cv2
from dataclasses import asdict
from enum import Enum

from .config import AppConfig
from .io_rs import RealSenseIO
from .spiral_3d import Spiral3D
from .track_mp import MediaPipeTracker
from .track_led import LEDTracker
from .depth_utils import median_depth_mm
from .metrics import summarize_errors
from .save import RunSaver
from .signal_processing import analyze_after_runs
from .html_results_viewer import generate_html_report
import sys
import webbrowser
from .calibration_utils import load_calibration, apply_calibration
from .directional_analysis import analyze_directional_tremor
from .session_manager import get_baseline_for_patient, calculate_improvements


class TrialState(Enum):
    """State machine for dot-follow trial."""
    COUNTDOWN = "countdown"
    FOLLOWING = "following"
    END_WAIT = "end_wait"
    COMPLETE = "complete"

class ExperimentRunner:
    # XReal glasses dimensions (3840x1080 side-by-side stereo)
    FULL_WIDTH = 3840
    FULL_HEIGHT = 1080
    EYE_WIDTH = FULL_WIDTH // 2
    EYE_HEIGHT = FULL_HEIGHT

    def __init__(self, cfg:AppConfig):
        self.cfg = cfg
        self.calibration_matrix = None

        # Load calibration if enabled
        if cfg.calibration.enabled:
            calibration_path = os.path.join(cfg.experiment.output_dir, cfg.calibration.calibration_file)
            self.calibration_matrix = load_calibration(calibration_path)
            if self.calibration_matrix is None:
                print(f"[Warning] Calibration enabled but failed to load from: {calibration_path}")
                print("[Warning] Continuing without calibration")
            else:
                print(f"[Calibration] Loaded successfully")
        else:
            print("[Calibration] Disabled (set calibration.enabled=true in config to enable)")

    def _draw_stereo_text(self, frame: np.ndarray, text: str, color: tuple):
        """Draw text on both eyes of stereo frame."""
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, text, (self.EYE_WIDTH + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    def _draw_stereo_circle(self, frame: np.ndarray, x: int, y: int, radius: int, color: tuple, thickness: int, disparity_px: float = 0.0):
        """Draw circle on both eyes of stereo frame with stereo disparity.

        Args:
            frame: Stereo frame to draw on
            x, y: Center position in display coordinates (1920x1080 space)
            radius: Circle radius
            color: Circle color (BGR)
            thickness: Line thickness
            disparity_px: Stereo disparity in pixels (for 3D depth)
        """
        # Left eye: shift left by half disparity
        x_left = int(x - disparity_px / 2.0)
        y_pos = int(y)
        if 0 <= x_left < self.EYE_WIDTH and 0 <= y_pos < self.EYE_HEIGHT:
            cv2.circle(frame, (x_left, y_pos), radius, color, thickness)

        # Right eye: shift right by half disparity
        x_right = int(x + disparity_px / 2.0 + self.EYE_WIDTH)
        if self.EYE_WIDTH <= x_right < self.FULL_WIDTH and 0 <= y_pos < self.EYE_HEIGHT:
            cv2.circle(frame, (x_right, y_pos), radius, color, thickness)

    def _get_dot_position_at_time(self, spiral: Spiral3D, t_elapsed: float, total_time: float):
        """
        Calculate moving dot position along spiral based on elapsed time.

        Args:
            spiral: Spiral3D object
            t_elapsed: Time elapsed since dot started moving
            total_time: Total time for one complete spiral traversal

        Returns:
            (dot_x, dot_y): Position in display coordinates
        """
        # Calculate progress (0.0 to 1.0)
        progress = min(t_elapsed / total_time, 1.0)

        # Map progress to spiral point index
        idx = int(progress * (len(spiral.spiral_points) - 1))

        # Get point from precomputed spiral (spiral-relative coords)
        x_rel = spiral.spiral_points[idx, 0]
        y_rel = spiral.spiral_points[idx, 1]

        # Transform to display coordinates
        cx = self.EYE_WIDTH / 2.0
        cy = self.EYE_HEIGHT / 2.0
        return x_rel + cx, y_rel + cy

    def _run_single_trial(self, rsio:RealSenseIO, spiral:Spiral3D, saver:RunSaver,
                          method:str, tracker, depth_win:int, fps:int,
                          metronome_bpm:int, show_live_preview:bool):
        # Trial state machine
        state = TrialState.COUNTDOWN
        countdown_start_time = None
        following_start_time = None
        end_wait_start_time = None

        # Configuration
        countdown_sec = self.cfg.dot_follow.countdown_sec
        dot_speed_sec = self.cfg.dot_follow.dot_speed_sec_per_spiral
        end_wait_sec = self.cfg.dot_follow.end_wait_sec

        # Data recording arrays
        times = []
        errs = []
        depths = []
        depth_valid_flags = []
        path_xy = []
        path_s = []
        path_err = []
        dot_positions = []
        dot_distances = []

        # Buffer for CSV rows (write after trial to avoid I/O in hot loop)
        csv_buffer = []

        frame_idx = 0
        color0, _, _ = rsio.get_aligned()
        if color0 is None:
            raise RuntimeError("Failed to read from RealSense.")
        cam_h, cam_w = color0.shape[:2]

        # Calculate scaling factors (camera -> display)
        scale_x = self.EYE_WIDTH / cam_w
        scale_y = self.EYE_HEIGHT / cam_h

        print(f"\n[3D Stereo Mode - Dot Follow]")
        print(f"  Camera: {cam_w}x{cam_h}")
        print(f"  Display: {self.FULL_WIDTH}x{self.FULL_HEIGHT} (side-by-side)")
        print(f"  Scale: {scale_x:.2f}x, {scale_y:.2f}x")
        print(f"  Countdown: {countdown_sec}s, Dot speed: {dot_speed_sec}s/spiral, End wait: {end_wait_sec}s\n")

        if self.cfg.experiment.save_preview:
            saver.open_video(self.FULL_WIDTH, self.FULL_HEIGHT, fps)

        last_xy = None
        MAX_JUMP_PX = int(self.cfg.experiment.max_jump_px)
        trial_active = True
        current_depth = None

        # Display smoothing (for visualization only, doesn't affect recorded data)
        display_smooth_alpha = self.cfg.dot_follow.display_smooth_alpha
        display_smooth_pos = None

        # Pre-render base spiral views (one for each depth state)
        print("[Optimization] Pre-rendering spiral views...")
        base_spiral_normal = spiral.draw_stereo(
            color_bgr=tuple(self.cfg.spiral.color_bgr),
            thickness=self.cfg.spiral.line_thickness,
            depth_mm=None,
            depth_close_mm=self.cfg.dot_follow.depth_close_mm,
            depth_far_mm=self.cfg.dot_follow.depth_far_mm
        )
        print("[Optimization] Spiral pre-rendered, will reuse each frame")

        # Timing instrumentation
        frame_times = []
        slow_frames = 0

        while trial_active:
            t_frame_start = time.perf_counter()
            color, depth, t_now = rsio.get_aligned()
            if color is None:
                continue

            # State machine: Update state based on time
            if state == TrialState.COUNTDOWN:
                if countdown_start_time is None:
                    countdown_start_time = t_now
                countdown_elapsed = t_now - countdown_start_time

                if countdown_elapsed >= countdown_sec:
                    state = TrialState.FOLLOWING
                    following_start_time = t_now
                    print(f"[Trial] Countdown complete, dot starting to move")

            elif state == TrialState.FOLLOWING:
                following_elapsed = t_now - following_start_time

                if following_elapsed >= dot_speed_sec:
                    state = TrialState.END_WAIT
                    end_wait_start_time = t_now
                    print(f"[Trial] Dot reached end, waiting {end_wait_sec}s")

            elif state == TrialState.END_WAIT:
                end_wait_elapsed = t_now - end_wait_start_time

                if end_wait_elapsed >= end_wait_sec:
                    state = TrialState.COMPLETE
                    trial_active = False
                    print(f"[Trial] Trial complete")

            # Track finger/LED in camera coordinates
            pt = tracker.track(color)

            # Jump gate (in camera coordinates)
            if pt is not None:
                x_cam, y_cam = pt
                if last_xy is not None:
                    dx = x_cam - last_xy[0]
                    dy = y_cam - last_xy[1]
                    if (dx*dx+dy*dy)**0.5 > MAX_JUMP_PX:
                        pt = None
                if pt is not None:
                    last_xy = (x_cam, y_cam)
            else:
                last_xy = None

            # Get depth for spiral coloring
            if pt is not None:
                x_cam, y_cam = pt
                current_depth = median_depth_mm(depth, int(round(x_cam)), int(round(y_cam)), depth_win,
                                               self.cfg.camera.min_depth_mm, self.cfg.camera.max_depth_mm)
            else:
                current_depth = None

            # Use pre-rendered spiral (fast copy instead of re-rendering)
            view = base_spiral_normal.copy()

            # Process finger position and draw visuals
            if pt is not None:
                x_cam, y_cam = pt

                # Apply calibration transform if available (camera coords -> camera coords)
                if self.calibration_matrix is not None:
                    x_cam, y_cam = apply_calibration(self.calibration_matrix, x_cam, y_cam)

                # Scale to display coordinates
                x = x_cam * scale_x
                y = y_cam * scale_y

                # Apply FOV adjustment scaling (center-based for proportional movement)
                cx_transform = self.EYE_WIDTH / 2.0
                cy_transform = self.EYE_HEIGHT / 2.0
                x = cx_transform + (x - cx_transform) * self.cfg.stereo_3d.finger_scale_x
                y = cy_transform + (y - cy_transform) * self.cfg.stereo_3d.finger_scale_y

                # Apply coordinate flipping for head-mounted camera
                if self.cfg.stereo_3d.flip_x:
                    x = self.EYE_WIDTH - x
                if self.cfg.stereo_3d.flip_y:
                    y = self.EYE_HEIGHT - y

                # Get depth
                z = current_depth
                z_valid = float(not math.isnan(z)) if z is not None else 0.0

                # Apply smoothing to display position (does NOT affect recorded data)
                if display_smooth_pos is None:
                    display_smooth_pos = np.array([x, y], dtype=np.float32)
                else:
                    display_smooth_pos = display_smooth_alpha * np.array([x, y]) + (1 - display_smooth_alpha) * display_smooth_pos

                x_display = float(display_smooth_pos[0])
                y_display = float(display_smooth_pos[1])

                # Draw finger dot (using smoothed position for display)
                spiral.draw_point_on_spiral(view, x_display, y_display, color=(255, 0, 255), radius=15)

                # State-specific visualization and data recording
                if state == TrialState.COUNTDOWN:
                    # Show countdown
                    countdown_remaining = int(countdown_sec - (t_now - countdown_start_time))
                    if countdown_remaining > 0:
                        countdown_text = str(countdown_remaining)
                    else:
                        countdown_text = "GO!"
                    self._draw_stereo_text(view, countdown_text, (0, 255, 255))

                elif state == TrialState.FOLLOWING:
                    # Calculate dot position
                    dot_elapsed = t_now - following_start_time
                    dot_x, dot_y = self._get_dot_position_at_time(spiral, dot_elapsed, dot_speed_sec)

                    # Draw moving dot
                    spiral.draw_point_on_spiral(view, dot_x, dot_y, color=(0, 255, 255), radius=20)

                    # Calculate distance from finger to dot
                    dot_dist = math.hypot(x - dot_x, y - dot_y)

                    # Find nearest spiral point for reference
                    xs, ys, s_sp, _ = spiral.nearest_point(x, y)
                    err = math.hypot(x - xs, y - ys)

                    # Buffer data (write after trial to avoid I/O overhead)
                    # NOTE: Uses RAW unsmoothed positions (x, y) for accurate tremor analysis
                    csv_buffer.append([f"{t_now:.6f}", method, frame_idx,
                                      f"{x_cam:.2f}", f"{y_cam:.2f}", f"{z:.1f}" if z is not None else "nan",
                                      f"{dot_x:.2f}", f"{dot_y:.2f}", f"{dot_dist:.2f}",
                                      f"{xs:.2f}", f"{ys:.2f}", f"{s_sp:.2f}",
                                      f"{err:.2f}", depth_win, z_valid])
                    times.append(t_now)
                    errs.append(dot_dist)  # Now tracking distance to dot, not spiral
                    depths.append(z if z is not None else float('nan'))
                    depth_valid_flags.append(z_valid)
                    path_xy.append((x, y))
                    path_s.append(s_sp)
                    path_err.append(err)
                    dot_positions.append((dot_x, dot_y))
                    dot_distances.append(dot_dist)

                    # Show status
                    self._draw_stereo_text(view, "FOLLOW THE DOT", (0, 220, 0))

                elif state == TrialState.END_WAIT:
                    self._draw_stereo_text(view, "COMPLETE - WAIT", (255, 128, 0))

                elif state == TrialState.COMPLETE:
                    self._draw_stereo_text(view, "TRIAL COMPLETE", (0, 0, 255))

            else:
                # No tracking detected - reset smoothing
                display_smooth_pos = None

                if state == TrialState.COUNTDOWN:
                    countdown_remaining = int(countdown_sec - (t_now - countdown_start_time)) if countdown_start_time else countdown_sec
                    countdown_text = str(max(countdown_remaining, 0))
                    self._draw_stereo_text(view, f"{countdown_text} (no track)", (128, 128, 128))
                elif state == TrialState.FOLLOWING:
                    # Still show moving dot even without tracking
                    dot_elapsed = t_now - following_start_time
                    dot_x, dot_y = self._get_dot_position_at_time(spiral, dot_elapsed, dot_speed_sec)
                    spiral.draw_point_on_spiral(view, dot_x, dot_y, color=(0, 255, 255), radius=20)
                    # Record dot position even when tracking is lost
                    dot_positions.append((dot_x, dot_y))
                    self._draw_stereo_text(view, "FOLLOW DOT (no track)", (165, 165, 0))
                elif state == TrialState.END_WAIT:
                    self._draw_stereo_text(view, "COMPLETE (no track)", (255, 128, 0))
                else:
                    self._draw_stereo_text(view, "NO TRACKING", (128, 128, 128))

            # Save and display
            if self.cfg.experiment.save_preview:
                saver.write_video_frame(view)

            if show_live_preview:
                if frame_idx == 0:
                    cv2.namedWindow("XReal_3D_Tracking", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("XReal_3D_Tracking", self.FULL_WIDTH, self.FULL_HEIGHT)
                    print("Window created. Drag to XReal display and press 'f' for fullscreen.\n")

                # Downscale for display to reduce rendering overhead (50% size)
                view_display = cv2.resize(view, (self.FULL_WIDTH // 2, self.FULL_HEIGHT // 2), interpolation=cv2.INTER_LINEAR)
                cv2.imshow("XReal_3D_Tracking", view_display)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    trial_active = False
                elif key == ord('f'):  # Toggle fullscreen
                    try:
                        prop = cv2.getWindowProperty("XReal_3D_Tracking", cv2.WND_PROP_FULLSCREEN)
                        if prop == cv2.WINDOW_NORMAL:
                            cv2.setWindowProperty("XReal_3D_Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            print("Fullscreen: ON")
                        else:
                            cv2.setWindowProperty("XReal_3D_Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                            print("Fullscreen: OFF")
                    except Exception as e:
                        print(f"Could not toggle fullscreen: {e}")

            frame_idx += 1

            # Measure frame processing time
            t_frame_end = time.perf_counter()
            frame_time_ms = (t_frame_end - t_frame_start) * 1000
            frame_times.append(frame_time_ms)

            # Track slow frames (> 33ms budget for 30fps)
            if frame_time_ms > 33.0:
                slow_frames += 1
                if slow_frames <= 5:  # Only print first 5 to avoid spam
                    print(f"[TIMING] Frame {frame_idx} took {frame_time_ms:.1f}ms (> 33ms budget)")
                elif slow_frames == 6:
                    print(f"[TIMING] More slow frames detected... (suppressing further warnings)")

        if show_live_preview:
            try:
                cv2.destroyWindow("XReal_3D_Tracking")
            except Exception:
                pass

        # Print timing summary
        if frame_times:
            avg_time = np.mean(frame_times)
            max_time = np.max(frame_times)
            min_time = np.min(frame_times)
            p95_time = np.percentile(frame_times, 95)
            slow_pct = (slow_frames / len(frame_times)) * 100
            print(f"\n[TIMING SUMMARY]")
            print(f"  Frames processed: {len(frame_times)}")
            print(f"  Avg: {avg_time:.1f}ms | Min: {min_time:.1f}ms | Max: {max_time:.1f}ms | P95: {p95_time:.1f}ms")
            print(f"  Slow frames (>33ms): {slow_frames}/{len(frame_times)} ({slow_pct:.1f}%)")
            print(f"  Target: 33.3ms for 30fps\n")

        # Write buffered CSV data after trial completes
        print("[Data] Writing buffered CSV data...")
        for row in csv_buffer:
            saver.write_frame_row(row)
        print(f"[Data] Wrote {len(csv_buffer)} rows to CSV")

        summ = summarize_errors(times, errs, depths, depth_valid_flags)
        saver.save_summary(summ)
        path = {"xy": path_xy, "s": path_s, "err": path_err, "times": times, "dot_xy": dot_positions}
        return summ, path

    def run_all(self):
        cfg = self.cfg
        rsio = RealSenseIO(cfg.camera.width, cfg.camera.height, cfg.camera.fps,
                           cfg.camera.depth_width, cfg.camera.depth_height, cfg.camera.depth_fps,
                           cfg.camera.use_auto_exposure, cfg.camera.exposure)
        rsio.start()

        # Warmup: flush camera buffers and let exposure settle
        print("[Camera] Warming up (flushing buffers)...")
        for _ in range(30):  # 30 frames @ 30fps = 1 second
            rsio.get_aligned()
        print("[Camera] Ready")

        intr = rsio.get_intrinsics()

        # Create 3D stereo spiral
        spiral = Spiral3D(
            cfg.spiral.a, cfg.spiral.b,
            cfg.spiral.turns, cfg.spiral.theta_step,
            target_depth_m=cfg.stereo_3d.target_depth_m,
            disparity_offset_px=cfg.stereo_3d.disparity_offset_px
        )

        mp_tracker = MediaPipeTracker(cfg.mediapipe.model_complexity, cfg.mediapipe.detection_confidence,
                                      cfg.mediapipe.tracking_confidence)
        hsv_tracker = LEDTracker(tuple(cfg.led.hsv_low), tuple(cfg.led.hsv_high),
                                cfg.led.brightness_threshold, cfg.led.morph_kernel, cfg.led.min_area)

        order = cfg.experiment.methods_order
        trials = 1
        base_out = cfg.experiment.output_dir
        os.makedirs(base_out, exist_ok=True)

        with open(os.path.join(base_out, "intrinsics.json"), "w") as f:
            json.dump(intr, f, indent=2)

        all_summaries = {m: [] for m in order}
        all_paths = {m: None for m in order}
        all_session_dirs = {m: [] for m in order}  # Track session directories for directional analysis
        try:
            for method in order:
                for _ in range(trials):
                    saver = RunSaver(base_out, method, spiral_id="stereo_3d")
                    saver.save_config_snapshot(asdict(cfg))
                    saver.save_intrinsics(intr)
                    if method == "mp":
                        summ, path = self._run_single_trial(rsio, spiral, saver, "mp", mp_tracker,
                                                            cfg.camera.depth_window, cfg.camera.fps,
                                                            cfg.experiment.metronome_bpm, cfg.show_live_preview)
                    elif method == "hsv":
                        summ, path = self._run_single_trial(rsio, spiral, saver, "hsv", hsv_tracker,
                                                            cfg.camera.depth_window, cfg.camera.fps,
                                                            cfg.experiment.metronome_bpm, cfg.show_live_preview)
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    all_session_dirs[method].append(str(saver.run_dir))  # Store session directory
                    saver.close()
                    all_summaries[method].append(summ)
                    all_paths[method] = path
        finally:
            mp_tracker.close()
            rsio.stop()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        # Post-analysis: Generate paths overlay and signal analysis
        signal_data = {}
        try:
            from .signal_processing import analyze_after_runs

            # Transform spiral points to display coordinates for visualization
            spiral_pts_display = spiral.spiral_points.copy()
            cx = spiral.EYE_WIDTH / 2.0
            cy = spiral.EYE_HEIGHT / 2.0
            spiral_pts_display[:, 0] += cx
            spiral_pts_display[:, 1] += cy

            # Generate signal analysis and paths overlay (overlay is generated inside analyze_after_runs)
            signal_data = analyze_after_runs(
                all_paths,
                spiral_pts_display,
                self.EYE_WIDTH,
                self.EYE_HEIGHT,
                base_out,
                target_fs=cfg.camera.fps,
                tremor_band_low=cfg.tremor_analysis.band_low_hz,
                tremor_band_high=cfg.tremor_analysis.band_high_hz
            )
            print("Signal analysis completed")

        except Exception as e:
            print(f"Post-analysis failed: {e}")
            import traceback
            traceback.print_exc()

        # Directional tremor analysis
        directional_data = {}
        try:
            print("Running directional tremor analysis...")
            for method, session_dirs in all_session_dirs.items():
                if session_dirs:
                    # Use the most recent session for directional analysis
                    latest_session_dir = session_dirs[-1]
                    frames_csv = os.path.join(latest_session_dir, 'frames.csv')

                    if os.path.exists(frames_csv):
                        directional_results = analyze_directional_tremor(
                            frames_csv,
                            method=method,
                            tremor_band=(cfg.tremor_analysis.band_low_hz, cfg.tremor_analysis.band_high_hz)
                        )
                        directional_data[method] = directional_results
                        print(f"  {method}: Worst tremor at {directional_results['worst_angle']:.1f}° "
                              f"(power: {directional_results['worst_power']:.2f})")
        except Exception as e:
            print(f"Directional analysis failed: {e}")
            import traceback
            traceback.print_exc()

        # Calculate improvements from baseline (if applicable)
        improvement_data = {}
        try:
            if cfg.session_metadata.session_type != 'baseline':
                baseline_session = get_baseline_for_patient(
                    cfg.experiment.output_dir,
                    cfg.session_metadata.patient_id
                )

                if baseline_session:
                    baseline_results_path = os.path.join(baseline_session['path'], 'results.json')
                    if os.path.exists(baseline_results_path):
                        with open(baseline_results_path, 'r') as f:
                            baseline_results = json.load(f)

                        # Create temporary current results for comparison
                        temp_current = {
                            "signal_analysis": signal_data,
                            "directional_analysis": directional_data
                        }

                        for method in signal_data.keys():
                            improvements = calculate_improvements(
                                baseline_results,
                                temp_current,
                                method=method
                            )
                            improvement_data[method] = improvements

                            summary = improvements.get('summary', {})
                            reduction = summary.get('primary_metric_reduction_pct', 0)
                            print(f"  {method}: {reduction:.1f}% tremor reduction from baseline")
        except Exception as e:
            print(f"Improvement calculation failed: {e}")

        # Consolidated results - dynamically include only methods that were run
        consolidated_results = {
            "session_metadata": asdict(cfg.session_metadata),
            "trial_summaries": {
                m: all_summaries[m][0] if all_summaries.get(m) and all_summaries[m] else {}
                for m in all_summaries.keys()
            },
            "signal_analysis": signal_data,
            "directional_analysis": directional_data,
            "improvements": improvement_data,
            "stereo_3d": {
                "target_depth_m": spiral.target_depth_m,
                "disparity_px": spiral.disparity_px
            }
        }

        with open(os.path.join(base_out, "results.json"), "w") as f:
            json.dump(consolidated_results, f, indent=2)

        print("\n=== Results Summary ===")
        for m in all_summaries.keys():
            if all_summaries[m]:
                s = all_summaries[m][0]
                rmse = s.get("rmse_time_weighted", 0)
                median = s["err_px"].get("median", 0)
                p95 = s["err_px"].get("p95", 0)
                loss = s.get("tracking_loss_rate", 0)
                print(f"{m.upper():<4} | RMSE: {rmse:.2f}px | Median: {median:.2f}px | P95: {p95:.2f}px | Loss: {loss*100:.1f}%")

        print(f"\nSaved: {base_out}/results.json")

        # Generate HTML report and open in browser
        try:
            results_json_path = os.path.join(base_out, "results.json")
            html_path = generate_html_report(results_json_path)

            print(f"\n=== HTML Report Generated ===")
            print(f"Report saved to: {html_path}")
            print("Opening in browser...")

            # Open in browser
            webbrowser.open(f'file:///{os.path.abspath(html_path)}')

        except Exception as e:
            print(f"Could not generate HTML report: {e}")
            import traceback
            traceback.print_exc()

        return consolidated_results
