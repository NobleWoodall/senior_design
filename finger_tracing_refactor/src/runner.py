import os
import json
import math
import numpy as np
import cv2
from dataclasses import asdict

from .config import AppConfig
from .io_rs import RealSenseIO
from .spiral_3d import Spiral3D
from .dwell import DwellDetector
from .track_mp import MediaPipeTracker
from .track_led import LEDTracker
from .depth_utils import median_depth_mm
from .metrics import summarize_errors
from .save import RunSaver
from .signal_processing import analyze_after_runs
from .results_display import display_results
from .calibration_utils import load_calibration, apply_calibration

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
            output_dir = cfg.experiment.output_dir
            if not os.path.isabs(output_dir):
                output_dir = os.path.abspath(output_dir)

            calibration_path = os.path.join(output_dir, cfg.calibration.calibration_file)
            print(f"[Calibration] Attempting to load from: {calibration_path}")
            print(f"[Calibration] File exists: {os.path.exists(calibration_path)}")

            self.calibration_matrix = load_calibration(calibration_path)
            if self.calibration_matrix is None:
                print(f"[Warning] Calibration enabled but failed to load from: {calibration_path}")
                print("[Warning] Continuing without calibration")
            else:
                print(f"[Calibration] Loaded successfully!")
                print(f"[Calibration] Matrix:")
                print(self.calibration_matrix)
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

    def _run_single_trial(self, rsio:RealSenseIO, spiral:Spiral3D, saver:RunSaver,
                          method:str, tracker, depth_win:int, fps:int,
                          metronome_bpm:int, show_live_preview:bool):
        endpoints = spiral.endpoints()
        sx, sy = endpoints["start"]
        ex, ey = endpoints["end"]
        dwell = DwellDetector(self.cfg.dwell.StartRadiusPx, self.cfg.dwell.StartDwellSec,
                              self.cfg.dwell.EndRadiusPx, self.cfg.dwell.StopDwellSec, self.cfg.dwell.hysteresis_px)

        times=[]
        errs=[]
        depths=[]
        depth_valid_flags=[]
        path_xy=[]
        path_s=[]
        path_err=[]

        frame_idx=0
        color0, _, _ = rsio.get_aligned()
        if color0 is None:
            raise RuntimeError("Failed to read from RealSense.")
        cam_h, cam_w = color0.shape[:2]

        # Calculate scaling factors (camera -> display)
        scale_x = self.EYE_WIDTH / cam_w
        scale_y = self.EYE_HEIGHT / cam_h

        print(f"\n[3D Stereo Mode]")
        print(f"  Camera: {cam_w}x{cam_h}")
        print(f"  Display: {self.FULL_WIDTH}x{self.FULL_HEIGHT} (side-by-side)")
        print(f"  Scale: {scale_x:.2f}x, {scale_y:.2f}x\n")

        if self.cfg.experiment.save_preview:
            saver.open_video(self.FULL_WIDTH, self.FULL_HEIGHT, fps)

        last_xy = None
        MAX_JUMP_PX = int(self.cfg.experiment.max_jump_px)
        trial_active=True
        started=False

        while trial_active:
            color, depth, t_now = rsio.get_aligned()
            if color is None:
                continue

            # Render stereo 3D spiral (3840x1080)
            view = spiral.draw_stereo(
                color_bgr=tuple(self.cfg.spiral.color_bgr),
                thickness=self.cfg.spiral.line_thickness
            )

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

            if pt is not None:
                x_cam, y_cam = pt

                # Apply calibration transform if available (camera coords -> camera coords)
                if self.calibration_matrix is not None:
                    x_cam, y_cam = apply_calibration(self.calibration_matrix, x_cam, y_cam)

                # Scale to display coordinates
                x = x_cam * scale_x
                y = y_cam * scale_y

                # Apply FOV adjustment scaling (center-based for proportional movement)
                cx = self.EYE_WIDTH / 2.0
                cy = self.EYE_HEIGHT / 2.0
                x = cx + (x - cx) * self.cfg.stereo_3d.finger_scale_x
                y = cy + (y - cy) * self.cfg.stereo_3d.finger_scale_y

                # Apply coordinate flipping for head-mounted camera
                if self.cfg.stereo_3d.flip_x:
                    x = self.EYE_WIDTH - x
                if self.cfg.stereo_3d.flip_y:
                    y = self.EYE_HEIGHT - y

                # Find nearest spiral point
                xs, ys, s_sp, _ = spiral.nearest_point(x, y)
                err = math.hypot(x-xs, y-ys)

                # Get depth at camera coordinates
                z = median_depth_mm(depth, int(round(x_cam)), int(round(y_cam)), depth_win,
                                   self.cfg.camera.min_depth_mm, self.cfg.camera.max_depth_mm)
                z_valid = float(not math.isnan(z))

                # Dwell detection (display coordinates)
                d_start = math.hypot(x-sx, y-sy)
                d_end = math.hypot(x-ex, y-ey)
                st = dwell.update(t_now, d_start, d_end)
                started = started or st.recording

                # Draw finger dot at same spiral-relative position on both eyes
                spiral.draw_point_on_spiral(view, x, y, color=(255, 0, 255), radius=20)

                # Draw status and circles on both eyes
                if not started:
                    self._draw_stereo_text(view, "STANDBY", (128, 128, 128))
                    self._draw_stereo_circle(view, sx, sy, self.cfg.dwell.StartRadiusPx, (0, 255, 255), 2, spiral.disparity_px)
                elif started and not st.end_detected:
                    self._draw_stereo_text(view, "RECORDING", (0, 220, 0))
                else:
                    self._draw_stereo_text(view, "END DETECTED", (255, 0, 0))

                self._draw_stereo_circle(view, ex, ey, self.cfg.dwell.EndRadiusPx, (255, 0, 255), 1, spiral.disparity_px)

                # Save data during recording (camera coordinates for analysis consistency)
                if started and not st.end_detected:
                    saver.write_frame_row([f"{t_now:.6f}", method, frame_idx,
                                           f"{x_cam:.2f}", f"{y_cam:.2f}", f"{z:.1f}",
                                           f"{xs:.2f}", f"{ys:.2f}", f"{s_sp:.2f}",
                                           f"{err:.2f}", depth_win, z_valid])
                    times.append(t_now)
                    errs.append(err)
                    depths.append(z)
                    depth_valid_flags.append(z_valid)
                    path_xy.append((x, y))  # Display coords for visualization
                    path_s.append(s_sp)
                    path_err.append(err)

                if st.end_detected:
                    trial_active=False
            else:
                # No tracking
                st = dwell.update(t_now, 1e9, 1e9)
                if not started:
                    self._draw_stereo_text(view, "STANDBY (no track)", (128, 128, 128))
                    self._draw_stereo_circle(view, sx, sy, self.cfg.dwell.StartRadiusPx, (0, 255, 255), 2, spiral.disparity_px)
                else:
                    self._draw_stereo_text(view, "RECORDING (no track)", (0, 165, 255))
                    self._draw_stereo_circle(view, ex, ey, self.cfg.dwell.EndRadiusPx, (255, 0, 255), 1, spiral.disparity_px)

            # Save and display
            if self.cfg.experiment.save_preview:
                saver.write_video_frame(view)

            if show_live_preview:
                if frame_idx == 0:
                    cv2.namedWindow("XReal_3D_Tracking", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("XReal_3D_Tracking", self.FULL_WIDTH, self.FULL_HEIGHT)
                    print("Window created. Drag to XReal display and press 'f' for fullscreen.\n")

                cv2.imshow("XReal_3D_Tracking", view)

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

        if show_live_preview:
            try:
                cv2.destroyWindow("XReal_3D_Tracking")
            except Exception:
                pass

        summ = summarize_errors(times, errs, depths, depth_valid_flags)
        saver.save_summary(summ)
        path = {"xy": path_xy, "s": path_s, "err": path_err, "times": times}
        return summ, path

    def run_all(self):
        cfg = self.cfg
        rsio = RealSenseIO(cfg.camera.width, cfg.camera.height, cfg.camera.fps,
                           cfg.camera.depth_width, cfg.camera.depth_height, cfg.camera.depth_fps,
                           cfg.camera.use_auto_exposure, cfg.camera.exposure)
        rsio.start()
        intr = rsio.get_intrinsics()

        # Create 3D stereo spiral
        spiral = Spiral3D(
            cfg.spiral.a, cfg.spiral.b,
            cfg.spiral.turns, cfg.spiral.theta_step,
            target_depth_m=cfg.stereo_3d.target_depth_m,
            disparity_offset_px=cfg.stereo_3d.disparity_offset_px
        )

        mp_tracker = MediaPipeTracker(cfg.mediapipe.model_complexity, cfg.mediapipe.detection_confidence,
                                      cfg.mediapipe.tracking_confidence, cfg.mediapipe.ema_alpha)
        hsv_tracker = LEDTracker(tuple(cfg.led.hsv_low), tuple(cfg.led.hsv_high),
                                cfg.led.brightness_threshold, cfg.led.morph_kernel, cfg.led.min_area)

        order = cfg.experiment.methods_order
        trials = 1
        base_out = cfg.experiment.output_dir
        os.makedirs(base_out, exist_ok=True)

        with open(os.path.join(base_out, "intrinsics.json"), "w") as f:
            json.dump(intr, f, indent=2)

        all_summaries = {"mp": [], "hsv": []}
        all_paths = {"mp": None, "hsv": None}
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
            from .signal_processing import draw_paths_overlay, analyze_after_runs

            # Generate paths overlay image
            overlay_path = os.path.join(base_out, "paths_overlay.png")

            # Transform spiral points to display coordinates for visualization
            spiral_pts_display = spiral.spiral_points.copy()
            cx = spiral.EYE_WIDTH / 2.0
            cy = spiral.EYE_HEIGHT / 2.0
            spiral_pts_display[:, 0] += cx
            spiral_pts_display[:, 1] += cy

            # Get path data
            path_mp = all_paths.get("mp", {}).get("xy", [])
            path_hsv = all_paths.get("hsv", {}).get("xy", [])

            # Draw overlay
            draw_paths_overlay(self.EYE_WIDTH, self.EYE_HEIGHT, spiral_pts_display,
                             path_mp, path_hsv, overlay_path)
            print(f"Paths overlay saved: {overlay_path}")

            # Generate signal analysis (tremor, FFT, PSD)
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

        # Consolidated results
        consolidated_results = {
            "trial_summaries": {
                "mp": all_summaries["mp"][0] if all_summaries["mp"] else {},
                "hsv": all_summaries["hsv"][0] if all_summaries["hsv"] else {}
            },
            "signal_analysis": signal_data,
            "stereo_3d": {
                "target_depth_m": spiral.target_depth_m,
                "disparity_px": spiral.disparity_px
            }
        }

        with open(os.path.join(base_out, "results.json"), "w") as f:
            json.dump(consolidated_results, f, indent=2)

        print("\n=== Results Summary ===")
        for m in ["mp", "hsv"]:
            if all_summaries[m]:
                s = all_summaries[m][0]
                rmse = s.get("rmse_time_weighted", 0)
                median = s["err_px"].get("median", 0)
                p95 = s["err_px"].get("p95", 0)
                loss = s.get("tracking_loss_rate", 0)
                print(f"{m.upper():<4} | RMSE: {rmse:.2f}px | Median: {median:.2f}px | P95: {p95:.2f}px | Loss: {loss*100:.1f}%")

        print(f"\nSaved: {base_out}/results.json")

        # Display results screen
        try:
            overlay_path = os.path.join(base_out, "paths_overlay.png")
            display_results(consolidated_results, overlay_path, base_out)
        except Exception as e:
            print(f"Could not display results screen: {e}")

        return consolidated_results
