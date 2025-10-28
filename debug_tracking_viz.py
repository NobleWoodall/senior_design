#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate Finger/LED → Stereo Display Mapping (XREAL + RealSense)

Two-window UI:
- "SBS Output" : full-size stereo buffer only (drag to the headset display)
- "Camera View": camera + crosshair + HUD (keep on laptop)

Linear mapping model (per-eye 1920×1080 or auto-detected per-eye WxH):
    x_disp = CX + s_x * (x_cam_rescaled - CX) + t_x
    y_disp = CY + s_y * (y_cam_rescaled - CY) + t_y
with optional X/Y flips (sign of s_x/s_y).

Keys:
  SPACE : capture sample at current target
  n/b   : next / prev target
  r     : refit mapping (least-squares) – needs ≥3 samples across ≥3 targets
  v     : toggle validation mode (auto-cycles targets)
  f/F   : flip X / Y (negates s_x / s_y)
  m     : toggle tracker (LED ↔ MediaPipe)
  p     : print current params
  w     : write params to config.yaml (safe section stereo_3d_calibration)
  q/ESC : quit

Dependencies:
- Your modules (preferred): io_rs.RealSenseIO, track_led.LEDTracker, track_mp.MediaPipeTracker,
  spiral_3d.Spiral3D, config.AppConfig
- Fallback (automatic): webcam + simple 2D spiral
- pip: opencv-python, numpy, pyyaml
"""

import os
import sys
import time
import math
import yaml
import cv2
import json
import inspect
import numpy as np
from collections import defaultdict

# -------------------- Project imports with fallbacks --------------------
HAVE_PROJECT = True
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'finger_tracing_refactor', 'src'))
    from io_rs import RealSenseIO
    from track_led import LEDTracker
    from track_mp import MediaPipeTracker
    from spiral_3d import Spiral3D
    from config import AppConfig
except Exception as e:
    HAVE_PROJECT = False
    print("[WARN] Project modules not found; running in fallback mode:", e)

    class AppConfig:
        def __init__(self, d):
            # Camera
            c = d.get("camera", {})
            self.camera = type("C", (), {})()
            self.camera.width  = c.get("width", 1280)
            self.camera.height = c.get("height", 720)
            self.camera.fps    = c.get("fps", 30)
            self.camera.depth_width  = c.get("depth_width", 640)
            self.camera.depth_height = c.get("depth_height", 480)
            self.camera.depth_fps    = c.get("depth_fps", 30)
            self.camera.use_auto_exposure = c.get("use_auto_exposure", True)
            self.camera.exposure = c.get("exposure", 100)
            # MediaPipe
            m = d.get("mediapipe", {})
            self.mediapipe = type("M", (), {})()
            self.mediapipe.model_complexity     = m.get("model_complexity", 0)
            self.mediapipe.detection_confidence = m.get("detection_confidence", 0.6)
            self.mediapipe.tracking_confidence  = m.get("tracking_confidence", 0.6)
            self.mediapipe.ema_alpha            = m.get("ema_alpha", 0.0)
            # LED
            l = d.get("led", {})
            self.led = type("L", (), {})()
            self.led.hsv_low   = l.get("hsv_low",  [20,120,120])
            self.led.hsv_high  = l.get("hsv_high", [35,255,255])
            self.led.brightness_threshold = l.get("brightness_threshold", 140)
            self.led.morph_kernel = l.get("morph_kernel", 3)
            self.led.min_area     = l.get("min_area", 12)
            # Spiral
            s = d.get("spiral", {})
            self.spiral = type("S", (), {})()
            self.spiral.a = s.get("a", 2.0)
            self.spiral.b = s.get("b", 4.0)
            self.spiral.turns = s.get("turns", 8)
            self.spiral.theta_step = s.get("theta_step", 0.05)
            self.spiral.color_bgr = s.get("color_bgr", [200,200,200])
            self.spiral.line_thickness = s.get("line_thickness", 2)
            # Stereo 3D
            t = d.get("stereo_3d", {})
            self.stereo_3d = type("T", (), {})()
            self.stereo_3d.target_depth_m = t.get("target_depth_m", 1.0)
            self.stereo_3d.disparity_offset_px = t.get("disparity_offset_px", 0)
            self.stereo_3d.finger_scale_x = t.get("finger_scale_x", 1.0)
            self.stereo_3d.finger_scale_y = t.get("finger_scale_y", 1.0)
            # offsets may not be defined in your dataclass; we won't write them under stereo_3d by default
            self.stereo_3d.offset_x = t.get("offset_x", 0.0)
            self.stereo_3d.offset_y = t.get("offset_y", 0.0)

        @classmethod
        def from_dict(cls, d): return cls(d)

    class RealSenseIO:
        """Fallback to single webcam; depth is None."""
        def __init__(self, w, h, fps, dw, dh, dfps, use_auto, exp):
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        def start(self): pass
        def stop(self):
            try: self.cap.release()
            except Exception: pass
        def get_aligned(self):
            ok, frame = self.cap.read()
            if not ok: return None, None, time.time()
            return frame, None, time.time()

    class LEDTracker:
        def __init__(self, hsv_low, hsv_high, bright_thr, k, min_area):
            self.hsv_low  = np.array(hsv_low,  np.uint8)
            self.hsv_high = np.array(hsv_high, np.uint8)
            self.min_area = min_area
        def track(self, bgr):
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: return None
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) < self.min_area: return None
            (x, y), r = cv2.minEnclosingCircle(c)
            return (float(x), float(y))

    class MediaPipeTracker:
        def __init__(self, *a, **k): pass
        def close(self): pass
        def track(self, bgr): return None  # not available in fallback

    class Spiral3D:
        """Fallback draws identical 2D spirals side-by-side."""
        def __init__(self, a, b, turns, theta_step, target_depth_m=1.0, disparity_offset_px=0):
            self.a, self.b, self.turns, self.theta_step = a, b, turns, theta_step
        def draw_stereo(self, color_bgr=(200,200,200), thickness=2, per_eye_w=1920, h=1080):
            W = per_eye_w * 2
            img = np.zeros((h, W, 3), np.uint8)
            for eye in [0,1]:
                ox = 0 if eye == 0 else per_eye_w
                pts = []
                cx, cy = per_eye_w // 2, h // 2
                t = 0.0
                while t < 2*math.pi*self.turns:
                    r = self.a + self.b * t
                    x = int(cx + r*math.cos(t))
                    y = int(cy + r*math.sin(t))
                    pts.append([x+ox, y])
                    t += self.theta_step
                pts = np.array(pts, np.int32).reshape(-1,1,2)
                cv2.polylines(img, [pts], False, tuple(map(int, color_bgr)), thickness, cv2.LINE_AA)
            return img

# -------------------- Drawing helpers --------------------
def put_text(img, text, xy, scale=0.6, color=(200,255,200)):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

def draw_crosshair(img, x, y, color=(255,0,255), size=28, thickness=2):
    if x is None or y is None: return
    x, y = int(x), int(y)
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)
    cv2.circle(img, (x, y), size//2, color, thickness)

def draw_target_both_eyes(stereo_img, tx, ty, color=(0,255,255), r=14, thick=3, per_eye_w=1920):
    cv2.circle(stereo_img, (int(tx), int(ty)), r, color, thick, cv2.LINE_AA)
    cv2.circle(stereo_img, (int(tx + per_eye_w), int(ty)), r, color, thick, cv2.LINE_AA)

def draw_mapped_both_eyes(stereo_img, Xd, Yd, color=(255,0,255), r=14, thick=3, per_eye_w=1920, per_eye_h=1080):
    xL = int(np.clip(Xd, 0, per_eye_w-1))
    y  = int(np.clip(Yd, 0, per_eye_h-1))
    xR = xL + per_eye_w
    cv2.circle(stereo_img, (xL, y), r, color, thick, cv2.LINE_AA)
    cv2.circle(stereo_img, (xR, y), r, color, thick, cv2.LINE_AA)

# -------------------- Safe wrapper for Spiral3D.draw_stereo --------------------
def safe_draw_stereo(spiral, color_bgr, thickness, per_eye_w, per_eye_h):
    """
    Call spiral.draw_stereo with only the args it supports (handles your repo + fallback).
    """
    try:
        sig = inspect.signature(spiral.draw_stereo)
        params = sig.parameters
        kwargs = {}
        if 'per_eye_w' in params: kwargs['per_eye_w'] = per_eye_w
        if 'h' in params: kwargs['h'] = per_eye_h
        if 'height' in params: kwargs['height'] = per_eye_h
        if 'width' in params:  kwargs['width']  = per_eye_w * 2
        return spiral.draw_stereo(color_bgr=color_bgr, thickness=thickness, **kwargs)
    except TypeError:
        return spiral.draw_stereo(color_bgr, thickness)

# -------------------- Targets / mapping --------------------
# Defaults; will be overridden after first stereo render
PER_EYE_W, PER_EYE_H = 1920, 1080
CX, CY = PER_EYE_W // 2, PER_EYE_H // 2
R = int(0.40 * (PER_EYE_W // 2))

def build_targets():
    global CX, CY, R, PER_EYE_W
    return [
        ("CENTER", (CX, CY)),
        ("RIGHT",  (CX + R, CY)),
        ("LEFT",   (CX - R, CY)),
        ("TOP",    (CX, CY - int(R*0.75))),
        ("BOTTOM", (CX, CY + int(R*0.75))),
        ("NE",     (CX + int(R*0.75), CY - int(R*0.75))),
        ("NW",     (CX - int(R*0.75), CY - int(R*0.75))),
        ("SE",     (CX + int(R*0.75), CY + int(R*0.75))),
        ("SW",     (CX - int(R*0.75), CY + int(R*0.75))),
    ]

TARGET_ORDER = build_targets()

def fit_linear_mapping(samples):
    """
    Fit s_x, t_x and s_y, t_y from least squares:
      (tx - CX) = s_x * (x1 - CX) + t_x
      (ty - CY) = s_y * (y1 - CY) + t_y
    where (x1,y1) are camera pixels rescaled to per-eye WxH.
    """
    if not samples:
        return 1.0, 0.0, 1.0, 0.0
    X  = np.array([s['x1'] - CX for s in samples], dtype=np.float32)
    Y  = np.array([s['y1'] - CY for s in samples], dtype=np.float32)
    XD = np.array([s['x_disp_t'] - CX for s in samples], dtype=np.float32)
    YD = np.array([s['y_disp_t'] - CY for s in samples], dtype=np.float32)

    Ax = np.vstack([X, np.ones_like(X)]).T
    Ay = np.vstack([Y, np.ones_like(Y)]).T
    sol_x, _, _, _ = np.linalg.lstsq(Ax, XD, rcond=None)
    sol_y, _, _, _ = np.linalg.lstsq(Ay, YD, rcond=None)
    s_x, t_x = float(sol_x[0]), float(sol_x[1])
    s_y, t_y = float(sol_y[0]), float(sol_y[1])
    return s_x, t_x, s_y, t_y

def apply_mapping(x_cam, y_cam, scale_x_px, scale_y_px, params):
    """
    Map raw camera coords to per-eye display coords.
    """
    x1 = x_cam * scale_x_px
    y1 = y_cam * scale_y_px
    s_x, t_x, s_y, t_y = params['s_x'], params['t_x'], params['s_y'], params['t_y']
    Xd = CX + s_x * (x1 - CX) + t_x
    Yd = CY + s_y * (y1 - CY) + t_y
    return Xd, Yd, x1, y1

# -------------------- Main --------------------
def main():
    # Load config (tolerate missing file)
    cfg_path = 'config.yaml'
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            cfg_dict = yaml.safe_load(f) or {}
    else:
        cfg_dict = {}
    cfg = AppConfig.from_dict(cfg_dict)

    # IO + trackers
    rsio = RealSenseIO(cfg.camera.width, cfg.camera.height, cfg.camera.fps,
                       cfg.camera.depth_width, cfg.camera.depth_height, cfg.camera.depth_fps,
                       cfg.camera.use_auto_exposure, cfg.camera.exposure)
    rsio.start()

    mp_tracker = MediaPipeTracker(
        getattr(cfg.mediapipe, 'model_complexity', 0),
        getattr(cfg.mediapipe, 'detection_confidence', 0.6),
        getattr(cfg.mediapipe, 'tracking_confidence', 0.6),
        getattr(cfg.mediapipe, 'ema_alpha', 0.0),
    )
    led_tracker = LEDTracker(
        tuple(getattr(cfg.led, 'hsv_low', [20,120,120])),
        tuple(getattr(cfg.led, 'hsv_high', [35,255,255])),
        getattr(cfg.led, 'brightness_threshold', 140),
        getattr(cfg.led, 'morph_kernel', 3),
        getattr(cfg.led, 'min_area', 12),
    )
    use_led = True
    tracker = led_tracker
    tracker_name = 'LED'

    spiral = Spiral3D(
        getattr(cfg.spiral, 'a', 2.0),
        getattr(cfg.spiral, 'b', 4.0),
        getattr(cfg.spiral, 'turns', 8),
        getattr(cfg.spiral, 'theta_step', 0.05),
        target_depth_m=getattr(cfg.stereo_3d, 'target_depth_m', 1.0),
        disparity_offset_px=getattr(cfg.stereo_3d, 'disparity_offset_px', 0)
    )

    params = {
        's_x': float(getattr(cfg.stereo_3d, 'finger_scale_x', 1.0)),
        's_y': float(getattr(cfg.stereo_3d, 'finger_scale_y', 1.0)),
        # Offsets kept in calibration section on save (won't break Stereo3DCfg):
        't_x': float(getattr(cfg.stereo_3d, 'offset_x', 0.0)),
        't_y': float(getattr(cfg.stereo_3d, 'offset_y', 0.0)),
    }

    # Get camera size & scaling
    color0, _, _ = rsio.get_aligned()
    if color0 is None:
        rsio.stop()
        raise RuntimeError("Failed to get first camera frame.")
    camH, camW = color0.shape[:2]
    scale_x_px = 1920.0 / float(camW)  # provisional; will be recomputed after stereo size
    scale_y_px = 1080.0 / float(camH)

    # --- Windows ---
    STEREO_WIN = 'SBS Output'   # drag to XREAL
    CAM_WIN    = 'Camera View'  # keep on laptop
    cv2.namedWindow(STEREO_WIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(CAM_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CAM_WIN, 960, 540)

    global PER_EYE_W, PER_EYE_H, CX, CY, R, TARGET_ORDER
    CX, CY = PER_EYE_W // 2, PER_EYE_H // 2
    R = int(0.40 * (PER_EYE_W // 2))
    # First stereo render; auto-detect actual per-eye size
    color_bgr = tuple(getattr(cfg.spiral, 'color_bgr', [200,200,200]))
    thickness = int(getattr(cfg.spiral, 'line_thickness', 2))
    stereo = safe_draw_stereo(spiral, color_bgr, thickness, PER_EYE_W, PER_EYE_H)
    sh, sw = stereo.shape[:2]
    PER_EYE_W, PER_EYE_H = sw // 2, sh
    # Update global per-eye dims & target layout
    TARGET_ORDER = build_targets()
    # Update camera scaling to match new per-eye dims
    scale_x_px = PER_EYE_W / float(camW)
    scale_y_px = PER_EYE_H / float(camH)
    # Size SBS window to exact stereo buffer
    cv2.resizeWindow(STEREO_WIN, sw, sh)

    # State
    samples = []
    samples_by_target = defaultdict(list)
    ti = 0
    validate_mode = False
    validate_idx = 0
    validate_timer = 0.0
    VALIDATE_HOLD = 0.8

    print("\n=== Calibration started ===")
    print("SPACE=capture  n/b=next/prev  r=fit  v=validate  f/F=flipX/Y  w=write  m=tracker  q/ESC=quit")

    try:
        while True:
            color, depth, t_now = rsio.get_aligned()
            if color is None:
                continue

            # ---- Camera window ----
            frame = color.copy()
            pt = tracker.track(color)
            x_cam = y_cam = None
            if pt is not None:
                x_cam, y_cam = pt
                draw_crosshair(frame, x_cam, y_cam, color=(0,0,255))
                put_text(frame, f'Cam: ({int(x_cam)}, {int(y_cam)})', (10, 30))
            put_text(frame, f'sx={params["s_x"]:.3f} tx={params["t_x"]:.1f}  sy={params["s_y"]:.3f} ty={params["t_y"]:.1f}  Tracker={tracker_name}', (10, 60), 0.55)
            cv2.imshow(CAM_WIN, cv2.resize(frame, (960, 540)))

            # ---- Stereo window ----
            stereo = safe_draw_stereo(spiral, color_bgr, thickness, PER_EYE_W, PER_EYE_H)

            # Pick target
            if validate_mode:
                name, (tx, ty) = TARGET_ORDER[validate_idx]
                if time.time() - validate_timer > VALIDATE_HOLD:
                    validate_idx = (validate_idx + 1) % len(TARGET_ORDER)
                    validate_timer = time.time()
            else:
                name, (tx, ty) = TARGET_ORDER[ti]

            # Targets in both eyes
            draw_target_both_eyes(stereo, tx, ty, color=(0,255,255), r=14, thick=3, per_eye_w=PER_EYE_W)
            cv2.putText(stereo, name, (int(tx+24), int(ty-12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(stereo, name, (int(tx+PER_EYE_W+24), int(ty-12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

            # Mapped fingertip in both eyes
            if x_cam is not None:
                Xd, Yd, x1, y1 = apply_mapping(x_cam, y_cam, scale_x_px, scale_y_px, params)
                draw_mapped_both_eyes(stereo, Xd, Yd, color=(255,0,255), r=14, thick=3, per_eye_w=PER_EYE_W, per_eye_h=PER_EYE_H)
                err = math.hypot(Xd - tx, Yd - ty)
                cv2.putText(stereo, f'|e|={err:5.1f}px', (int(tx+24), int(ty+24)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,255), 2, cv2.LINE_AA)
                cv2.putText(stereo, f'|e|={err:5.1f}px', (int(tx+PER_EYE_W+24), int(ty+24)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,255), 2, cv2.LINE_AA)

            # Show full-size SBS
            cv2.imshow(STEREO_WIN, stereo)

            # Keys (one waitKey handles both windows)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key in (ord('n'),):
                ti = (ti + 1) % len(TARGET_ORDER)
            elif key in (ord('b'),):
                ti = (ti - 1) % len(TARGET_ORDER)
            elif key == 32:  # SPACE
                if x_cam is not None:
                    Xd, Yd, x1, y1 = apply_mapping(x_cam, y_cam, scale_x_px, scale_y_px, params)
                    sample = {
                        'x_cam': float(x_cam), 'y_cam': float(y_cam),
                        'x1': float(x1), 'y1': float(y1),
                        'x_disp_t': float(tx), 'y_disp_t': float(ty),
                        'target': name,
                    }
                    samples.append(sample)
                    samples_by_target[name].append(sample)
                else:
                    print("No track to capture.")
            elif key in (8, 127):  # BACKSPACE/DEL: remove last sample for this target
                if samples_by_target[name]:
                    last = samples_by_target[name].pop()
                    for i in range(len(samples)-1, -1, -1):
                        if samples[i] is last:
                            samples.pop(i)
                            break
                else:
                    print("No samples to delete for this target.")
            elif key == ord('r'):
                distinct = sum(1 for k,v in samples_by_target.items() if v)
                if len(samples) >= 3 and distinct >= 3:
                    sx, tx_fit, sy, ty_fit = fit_linear_mapping(samples)
                    params['s_x'], params['t_x'], params['s_y'], params['t_y'] = sx, tx_fit, sy, ty_fit
                    print(f'Fitted: sx={sx:.4f} tx={tx_fit:.2f}  sy={sy:.4f} ty={ty_fit:.2f}  (N={len(samples)}; targets={distinct})')
                else:
                    print("Need ≥3 samples from ≥3 targets to fit.")
            elif key == ord('v'):
                validate_mode = not validate_mode
                validate_idx = 0
                validate_timer = time.time()
                print(f'Validate mode: {validate_mode}')
            elif key == ord('f'):
                params['s_x'] = -params['s_x']
                print(f'Flip X -> sx={params["s_x"]:.4f}')
            elif key == ord('F'):
                params['s_y'] = -params['s_y']
                print(f'Flip Y -> sy={params["s_y"]:.4f}')
            elif key == ord('m'):
                use_led = not use_led
                tracker = led_tracker if use_led else mp_tracker
                tracker_name = 'LED' if use_led else 'MediaPipe'
                print(f'Tracker -> {tracker_name}')
            elif key == ord('p'):
                print(f'Params: sx={params["s_x"]:.4f} tx={params["t_x"]:.2f}  sy={params["s_y"]:.4f} ty={params["t_y"]:.2f}')
            elif key == ord('w'):
                # SAFE WRITE: keep offsets away from stereo_3d to avoid breaking your dataclass
                cfg_dict.setdefault('stereo_3d', {})
                cfg_dict['stereo_3d']['finger_scale_x'] = float(params['s_x'])
                cfg_dict['stereo_3d']['finger_scale_y'] = float(params['s_y'])

                cfg_dict.setdefault('stereo_3d_calibration', {})
                cfg_dict['stereo_3d_calibration']['offset_x'] = float(params['t_x'])
                cfg_dict['stereo_3d_calibration']['offset_y'] = float(params['t_y'])
                cfg_dict['stereo_3d_calibration']['_note'] = {
                    'cam_w': int(camW), 'cam_h': int(camH),
                    'per_eye_w': int(PER_EYE_W), 'per_eye_h': int(PER_EYE_H),
                    'rescale_x': float(scale_x_px), 'rescale_y': float(scale_y_px),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                with open(cfg_path, 'w') as f:
                    yaml.safe_dump(cfg_dict, f, sort_keys=False)
                print(f'Wrote scales under stereo_3d and offsets under stereo_3d_calibration → {cfg_path}')

    except KeyboardInterrupt:
        pass
    finally:
        try:
            if hasattr(mp_tracker, 'close'):
                mp_tracker.close()
        except Exception:
            pass
        try:
            rsio.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
