import os, json, math, time
import numpy as np, cv2
from typing import Dict, Any
from dataclasses import asdict

from .config import AppConfig
from .io_rs import RealSenseIO
from .spiral import Spiral
from .dwell import DwellDetector
from .track_mp import MediaPipeTracker
from .track_hsv import HSVTracker
from .depth_utils import median_depth_mm
from .metrics import summarize_errors
from .viz import draw_status, draw_cross, draw_circle, draw_meter, metronome_overlay
from .save import RunSaver
from .signal_processing import analyze_after_runs

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
        color0, depth0, _ = rsio.get_aligned()
        if color0 is None: 
            raise RuntimeError("Failed to read from RealSense.")
        h,w = color0.shape[:2]
        if self.cfg.experiment.save_preview: 
            saver.open_video(w,h,fps)

        last_xy = None
        MAX_JUMP_PX = int(self.cfg.experiment.max_jump_px)
        trial_active=True
        started=False
        while trial_active:
            color, depth, t_now = rsio.get_aligned()
            if color is None: 
                continue
            view = color.copy()
            spiral.draw(view, color=tuple(self.cfg.spiral.color_bgr), thickness=self.cfg.spiral.line_thickness)

            # detect
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
                xs,ys,s_sp,theta_t = spiral.nearest_point(x,y)
                err = math.hypot(x-xs, y-ys)
                z = median_depth_mm(depth, int(round(x)), int(round(y)), depth_win)
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

                draw_circle(view, x, y, 6, (0,255,255), -1)
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

            if self.cfg.experiment.save_preview: 
                saver.write_video_frame(view)

            if show_live_preview:
                if frame_idx==0:
                    cv2.namedWindow("Finger Trace", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Finger Trace", 960, 540)
                cv2.imshow("Finger Trace", view)
                if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                    trial_active=False

            frame_idx += 1

        if show_live_preview:
            try: 
                cv2.destroyWindow("Finger Trace")
            except Exception: 
                pass

        summ = summarize_errors(times, errs, depths, depth_valid_flags)
        saver.save_summary(summ)
        path = {"xy": path_xy, "s": path_s, "err": path_err}
        return summ, path

    def run_all(self):
        cfg = self.cfg
        rsio = RealSenseIO(cfg.camera.width, cfg.camera.height, cfg.camera.fps,
                           cfg.camera.depth_width, cfg.camera.depth_height, cfg.camera.depth_fps,
                           cfg.camera.use_auto_exposure, cfg.camera.exposure)
        rsio.start()
        intr = rsio.get_intrinsics()

        spiral = Spiral(cfg.camera.width, cfg.camera.height, cfg.spiral.a, cfg.spiral.b, cfg.spiral.turns, cfg.spiral.theta_step)

        mp_tracker = MediaPipeTracker(cfg.mediapipe.model_complexity, cfg.mediapipe.detection_confidence,
                                      cfg.mediapipe.tracking_confidence, cfg.mediapipe.ema_alpha)
        hsv_tracker = HSVTracker(tuple(cfg.color.hsv_low), tuple(cfg.color.hsv_high),
                                 cfg.color.morph_kernel, cfg.color.min_area,
                                 cfg.color.use_flow_fallback, cfg.color.flow_win, cfg.color.flow_max_level)

        order = cfg.experiment.methods_order
        trials = 1  # force exactly one per method
        base_out = cfg.experiment.output_dir
        os.makedirs(base_out, exist_ok=True)

        with open(os.path.join(base_out, "intrinsics.json"), "w") as f: 
            json.dump(intr, f, indent=2)

        all_summaries = {"mp": [], "hsv": []}
        all_paths = {"mp": None, "hsv": None}
        cfg_txt = asdict(cfg)
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

        # comparison summary (averaged, though each has one)
        def extract(summ_list, metric_key):
            out=[] 
            for s in summ_list:
                if metric_key=="rmse": 
                    out.append(s.get("rmse_time_weighted", None))
                elif metric_key=="median": 
                    out.append(s["err_px"]["median"] if s["err_px"]["median"] is not None else None)
                elif metric_key=="p95": 
                    out.append(s["err_px"]["p95"] if s["err_px"]["p95"] is not None else None)
                elif metric_key=="loss": 
                    out.append(s.get("tracking_loss_rate", None))
            out=[v for v in out if v is not None]
            return float(np.mean(out)) if out else None

        comp = {
            "mp": {"RMSE_px": extract(all_summaries["mp"], "rmse"),
                   "median_px": extract(all_summaries["mp"], "median"),
                   "p95_px": extract(all_summaries["mp"], "p95"),
                   "tracking_loss_rate": extract(all_summaries["mp"], "loss")},
            "hsv":{"RMSE_px": extract(all_summaries["hsv"], "rmse"),
                   "median_px": extract(all_summaries["hsv"], "median"),
                   "p95_px": extract(all_summaries["hsv"], "p95"),
                   "tracking_loss_rate": extract(all_summaries["hsv"], "loss")}
        }
        with open(os.path.join(base_out, "comparison_summary.json"), "w") as f: 
            json.dump(comp, f, indent=2)
        print("=== Comparison (one trial each) ===")
        for m in ["mp","hsv"]:
            c=comp[m]
            print(f"{m.upper():<4} | RMSE: {c['RMSE_px']:.2f}px | Median: {c['median_px']:.2f}px | P95: {c['p95_px']:.2f}px | Loss: {c['tracking_loss_rate']*100:.1f}%")

        # Post-analysis: overlay & spatial PSD/RMS
        try:
            analyze_after_runs(all_paths, getattr(spiral,"curve",None), cfg.camera.width, cfg.camera.height, base_out, ds_px=2.0)
            print("Saved: runs/paths_overlay.png, runs/signal_summary.json, runs/psd_{mp,hsv}_spatial.csv")
        except Exception as e:
            print("Post-analysis failed:", e)
