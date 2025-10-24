import os
import json
import math
import numpy as np
import cv2
from dataclasses import asdict

from .config import AppConfig
from .io_rs import RealSenseIO
from .spiral import Spiral
from .dwell import DwellDetector
from .track_mp import MediaPipeTracker
from .track_led import LEDTracker
from .depth_utils import median_depth_mm
from .metrics import summarize_errors
from .viz import draw_status, draw_circle, draw_meter, metronome_overlay
from .save import RunSaver
from .signal_processing import analyze_after_runs
from .results_display import display_results

class ExperimentRunner:
    def __init__(self, cfg:AppConfig):
        self.cfg = cfg

    def _run_single_trial(self, rsio:RealSenseIO, spiral:Spiral, saver:RunSaver,
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
        h,w = color0.shape[:2]

        zoom_factor = 2.0

        if self.cfg.experiment.save_preview:
            saver.open_video(w, h, fps)

        last_xy = None
        MAX_JUMP_PX = int(self.cfg.experiment.max_jump_px)
        trial_active=True
        started=False
        while trial_active:
            color, depth, t_now = rsio.get_aligned()
            if color is None:
                continue

            # Create black canvas for AR overlay
            view = np.zeros((h, w, 3), dtype=np.uint8)
            spiral.draw(view, color=tuple(self.cfg.spiral.color_bgr), thickness=self.cfg.spiral.line_thickness)

            # detect finger/LED
            pt = tracker.track(color)

            # jump gate BEFORE metrics
            if pt is not None:
                x,y = pt
                if last_xy is not None:
                    dx=x-last_xy[0]
                    dy=y-last_xy[1]
                    if (dx*dx+dy*dy)**0.5 > MAX_JUMP_PX:
                        pt = None
                if pt is not None:
                    last_xy=(x,y)
            else:
                last_xy=None

            if pt is not None:
                x,y = pt
                xs,ys,s_sp,_ = spiral.nearest_point(x,y)
                err = math.hypot(x-xs, y-ys)
                z = median_depth_mm(depth, int(round(x)), int(round(y)), depth_win,
                                   self.cfg.camera.min_depth_mm, self.cfg.camera.max_depth_mm)
                z_valid = float(not math.isnan(z))
                d_start = math.hypot(x-sx, y-sy)
                d_end   = math.hypot(x-ex, y-ey)
                st = dwell.update(t_now, d_start, d_end)
                started = started or st.recording

                if not started:
                    draw_status(view, "standby", (128,128,128))
                    draw_circle(view, sx, sy, self.cfg.dwell.StartRadiusPx, (0,255,255), 2)
                elif started and not st.end_detected:
                    draw_status(view, "recording", (0,220,0))
                else:
                    draw_status(view, "end-detected", (255,0,0))

                # Draw finger tracking dot with glow effect
                dot_color = (255, 0, 255)  # Bright magenta
                dot_radius = 4
                draw_circle(view, int(x), int(y), dot_radius + 4, dot_color, 2)
                draw_circle(view, int(x), int(y), dot_radius, dot_color, -1)
                draw_meter(view, err)
                draw_circle(view, sx, sy, self.cfg.dwell.StartRadiusPx, (0,255,255), 1)
                draw_circle(view, ex, ey, self.cfg.dwell.EndRadiusPx, (255,0,255), 1)
                metronome_overlay(view, t_now, metronome_bpm)

                if started and not st.end_detected:
                    saver.write_frame_row([f"{t_now:.6f}", method, frame_idx,
                                           f"{x:.2f}", f"{y:.2f}", f"{z:.1f}",
                                           f"{xs:.2f}", f"{ys:.2f}", f"{s_sp:.2f}",
                                           f"{err:.2f}", depth_win, z_valid])
                    times.append(t_now)
                    errs.append(err)
                    depths.append(z)
                    depth_valid_flags.append(z_valid)
                    path_xy.append((x,y))
                    path_s.append(s_sp)
                    path_err.append(err)

                if st.end_detected: 
                    trial_active=False
            else:
                st = dwell.update(t_now, 1e9, 1e9)
                if not started:
                    draw_status(view, "standby", (128,128,128))
                    draw_circle(view, sx, sy, self.cfg.dwell.StartRadiusPx, (0,255,255), 2)
                else:
                    draw_status(view, "recording (no-track)", (0,165,255))
                    draw_circle(view, ex, ey, self.cfg.dwell.EndRadiusPx, (255,0,255), 1)
                metronome_overlay(view, t_now, metronome_bpm)

            # Apply zoom effect by cropping and resizing
            if zoom_factor != 1.0:
                # Calculate crop region (center crop)
                crop_w = int(w / zoom_factor)
                crop_h = int(h / zoom_factor)
                start_x = (w - crop_w) // 2
                start_y = (h - crop_h) // 2
                end_x = start_x + crop_w
                end_y = start_y + crop_h

                # Crop the center region
                view = view[start_y:end_y, start_x:end_x]
                # Resize back to original size (this makes everything appear larger)
                view = cv2.resize(view, (w, h), interpolation=cv2.INTER_LINEAR)

            if self.cfg.experiment.save_preview: 
                saver.write_video_frame(view)

            if show_live_preview:
                if frame_idx==0:
                    cv2.namedWindow("Finger Trace", cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty("Finger Trace", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("Finger Trace", view)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    trial_active=False
                elif key == ord('['):  # Decrease zoom (farther away)
                    zoom_factor = max(1.0, zoom_factor - 0.1)
                    print(f"Zoom: {zoom_factor:.1f}x (appears {'closer' if zoom_factor > 1.0 else 'normal'})")
                elif key == ord(']'):  # Increase zoom (closer)
                    zoom_factor = min(4.0, zoom_factor + 0.1)
                    print(f"Zoom: {zoom_factor:.1f}x (appears closer)")

            frame_idx += 1

        if show_live_preview:
            try: 
                cv2.destroyWindow("Finger Trace")
            except Exception: 
                pass

        summ = summarize_errors(times, errs, depths, depth_valid_flags)
        saver.save_summary(summ)
        path = {"xy": path_xy, "s": path_s, "err": path_err, "times": times}
        return summ, path

    def  run_all(self):
        cfg = self.cfg
        rsio = RealSenseIO(cfg.camera.width, cfg.camera.height, cfg.camera.fps,
                           cfg.camera.depth_width, cfg.camera.depth_height, cfg.camera.depth_fps,
                           cfg.camera.use_auto_exposure, cfg.camera.exposure)
        rsio.start()
        intr = rsio.get_intrinsics()

        spiral = Spiral(cfg.camera.width, cfg.camera.height, cfg.spiral.a, cfg.spiral.b, cfg.spiral.turns, cfg.spiral.theta_step)

        mp_tracker = MediaPipeTracker(cfg.mediapipe.model_complexity, cfg.mediapipe.detection_confidence,
                                      cfg.mediapipe.tracking_confidence, cfg.mediapipe.ema_alpha)
        # Use LED tracker for HSV method (combined red + brightness)
        hsv_tracker = LEDTracker(tuple(cfg.led.hsv_low), tuple(cfg.led.hsv_high),
                                cfg.led.brightness_threshold, cfg.led.morph_kernel, cfg.led.min_area)

        order = cfg.experiment.methods_order
        trials = 1  # force exactly one per method
        base_out = cfg.experiment.output_dir
        os.makedirs(base_out, exist_ok=True)

        with open(os.path.join(base_out, "intrinsics.json"), "w") as f:
            json.dump(intr, f, indent=2)

        all_summaries = {"mp": [], "hsv": []}
        all_paths = {"mp": None, "hsv": None}
        try:
            for method in order:
                for _ in range(trials):
                    saver = RunSaver(base_out, method, spiral_id="default")
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

        # Post-analysis: overlay & temporal PSD/RMS with tremor band filtering
        signal_data = {}
        try:
            signal_data = analyze_after_runs(all_paths, getattr(spiral,"curve",None),
                                           cfg.camera.width, cfg.camera.height, base_out,
                                           target_fs=cfg.camera.fps,
                                           tremor_band_low=cfg.tremor_analysis.band_low_hz,
                                           tremor_band_high=cfg.tremor_analysis.band_high_hz)
        except Exception as e:
            print("Post-analysis failed:", e)

        # Consolidated results: merge per-trial summaries, comparison metrics, and signal analysis
        consolidated_results = {
            "trial_summaries": {
                "mp": all_summaries["mp"][0] if all_summaries["mp"] else {},
                "hsv": all_summaries["hsv"][0] if all_summaries["hsv"] else {}
            },
            "signal_analysis": signal_data
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

        print(f"\nSaved: {base_out}/results.json, {base_out}/paths_overlay.png")

        # Display results screen
        try:
            overlay_path = os.path.join(base_out, "paths_overlay.png")
            display_results(consolidated_results, overlay_path, base_out)
        except Exception as e:
            print(f"Could not display results screen: {e}")

        return consolidated_results
