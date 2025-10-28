"""
Debug Tracking Visualization (FOV‑Calibrated)

Shows side‑by‑side:
- Left: Raw camera view with tracked point (RGB from D435)
- Right: Stereo display view (3840×1080) with transformed coordinates

Key improvements vs prior version
- Computes finger_scale_x/y from true H‑FOV:  scale = H_FOV_display / H_FOV_camera
- Converts headset diagonal FOV (e.g., 57° for XREAL One Pro) → horizontal FOV using 16:9
- Profiles for common headsets (Air 2 Pro = 46° diag, One Pro = 57° diag)
- On‑screen calibration bar + quick hotkeys for fine tuning
- Safe defaults if config fields are missing

Dependencies
- Your project modules: io_rs.RealSenseIO, track_led.LEDTracker, track_mp.MediaPipeTracker,
  spiral_3d.Spiral3D, config.AppConfig
- OpenCV, numpy, pyyaml

Usage
- Put this file at project root (same level as config.yaml and finger_tracing_refactor/)
- Run:  python debug_tracking_visualization_fov_calibrated.py --headset one_pro
  or set display.diag_fov_deg in config.yaml; this script will prefer CLI > config > profile.

Hotkeys (additions)
  'g' - Cycle headset profiles (auto‑recompute scale)
  'c' - Recompute scale from FOVs currently set
  'p' - Print current FOVs and computed scale
  'r' - Reset scale/offset to computed defaults
Existing keys retained (m, s, f, F, +, -, [, ], u, d, q/ESC)
"""

import sys
import os
import math
import argparse
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


# ---------- FOV utilities ----------
ASPECT_W, ASPECT_H = 16, 9

HEADSET_PROFILES = [
    {"name": "air2_pro", "diag_fov_deg": 46.0},
    {"name": "one_pro",   "diag_fov_deg": 57.0},
]

# Intel RealSense D435 RGB typical horizontal FOV
# (Depth H‑FOV is wider; we track in RGB).
DEFAULT_RGB_HFOV_DEG = 69.4  # ± ~3° tolerance


def horizontal_fov_from_diagonal(diag_deg: float, aspect_w: int = ASPECT_W, aspect_h: int = ASPECT_H) -> float:
    """Convert diagonal FOV to horizontal FOV for a given aspect ratio using pinhole model."""
    r = aspect_w / aspect_h
    t_v = math.tan(math.radians(diag_deg / 2.0)) / math.sqrt(1 + r * r)
    t_h = r * t_v
    return math.degrees(2.0 * math.atan(t_h))


def compute_scale(hfov_display_deg: float, hfov_camera_deg: float) -> float:
    return float(hfov_display_deg) / float(hfov_camera_deg)


# ---------- Drawing helpers ----------

def draw_crosshair(img, x, y, color, size=30, thickness=2):
    x, y = int(x), int(y)
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)
    cv2.circle(img, (x, y), size // 2, color, thickness)


def draw_info_text(img, lines, start_y=30):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, start_y + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def draw_calibration_bar(img, disp_hfov_deg, cam_hfov_deg):
    """Overlay simple calibration bar/ticks for visual scale sanity checks."""
    h, w = img.shape[:2]
    cx, cy = w // 4 + 960, h // 2  # roughly center of the right panel (which is 1920 wide)
    # Bar across 60% width of right panel
    bar_w = int(0.5 * 1920)
    x0 = 960 + (1920 - bar_w) // 2
    x1 = x0 + bar_w
    y = h - 80

    cv2.line(img, (x0, y), (x1, y), (200, 200, 200), 2)
    cv2.circle(img, (x0, y), 4, (200, 200, 200), -1)
    cv2.circle(img, (x1, y), 4, (200, 200, 200), -1)

    # Mark ±10°, ±20° in display space
    def px_from_deg_display(deg):
        # per‑eye width is 1920 px
        px_per_deg = 1920.0 / disp_hfov_deg if disp_hfov_deg > 0 else 0.0
        return int(px_per_deg * deg)

    for deg in [10, 20]:
        dx = px_from_deg_display(deg)
        # right of center (use middle of right eye: base = 960 + 1920/2)
        base = 960 + 1920 // 2
        for sign in (-1, 1):
            x = base + sign * dx
            cv2.line(img, (x, y - 12), (x, y + 12), (0, 255, 255), 2)
            cv2.putText(img, f"{sign*deg:+d}°", (x - 20, y - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Labels
    cv2.putText(img, f"Display H‑FOV: {disp_hfov_deg:.1f}°  |  Camera H‑FOV: {cam_hfov_deg:.1f}°",
                (960 + 20, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--headset', choices=[p['name'] for p in HEADSET_PROFILES], default=None,
                        help='Headset profile (sets diagonal FOV). Overrides config if given.')
    parser.add_argument('--rgb_hfov_deg', type=float, default=None,
                        help='Override camera RGB horizontal FOV (deg). Default: D435 ≈ 69.4°')
    args = parser.parse_args()

    print('=' * 80)
    print('DEBUG TRACKING VISUALIZATION — FOV‑CALIBRATED')
    print('=' * 80)

    # Load config
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = AppConfig.from_dict(cfg_dict)

    # Resolve headset diag FOV (CLI > config > profile[0])
    diag_fov_cfg = None
    try:
        diag_fov_cfg = float(cfg.display.diag_fov_deg)
    except Exception:
        diag_fov_cfg = None

    if args.headset is not None:
        # Use specified profile
        profile = next(p for p in HEADSET_PROFILES if p['name'] == args.headset)
        diag_fov_deg = float(profile['diag_fov_deg'])
        headset_name = profile['name']
    elif diag_fov_cfg is not None:
        diag_fov_deg = diag_fov_cfg
        headset_name = 'custom_cfg'
    else:
        # Default to One Pro unless you prefer Air 2 Pro
        diag_fov_deg = next(p for p in HEADSET_PROFILES if p['name'] == 'one_pro')['diag_fov_deg']
        headset_name = 'one_pro(default)'

    # Camera H‑FOV
    cam_hfov_deg = float(args.rgb_hfov_deg) if args.rgb_hfov_deg is not None else \
                    (float(getattr(getattr(cfg, 'camera', object()), 'rgb_hfov_deg', DEFAULT_RGB_HFOV_DEG)))

    # Convert to display horizontal FOV
    disp_hfov_deg = horizontal_fov_from_diagonal(diag_fov_deg, ASPECT_W, ASPECT_H)
    scale_default = compute_scale(disp_hfov_deg, cam_hfov_deg)

    print(f"Headset: {headset_name} | diag FOV = {diag_fov_deg:.1f}° → display H‑FOV = {disp_hfov_deg:.2f}°")
    print(f"Camera RGB H‑FOV = {cam_hfov_deg:.2f}°  ⇒  default scale = {scale_default:.3f}")

    # Initialize camera
    rsio = RealSenseIO(
        cfg.camera.width, cfg.camera.height, cfg.camera.fps,
        cfg.camera.depth_width, cfg.camera.depth_height, cfg.camera.depth_fps,
        cfg.camera.use_auto_exposure, cfg.camera.exposure
    )
    rsio.start()
    print('\n✓ Camera started')

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
    use_led = True
    tracker = led_tracker
    tracker_name = 'LED'

    # Initialize spiral
    spiral = Spiral3D(
        cfg.spiral.a, cfg.spiral.b,
        cfg.spiral.turns, cfg.spiral.theta_step,
        target_depth_m=cfg.stereo_3d.target_depth_m,
        disparity_offset_px=cfg.stereo_3d.disparity_offset_px
    )
    print(f"✓ Spiral created: {spiral.disparity_px:.1f}px disparity")

    # Settings (mutable for real‑time adjustment)
    settings = {
        'finger_scale_x': scale_default if getattr(cfg.stereo_3d, 'finger_scale_x', None) is None else cfg.stereo_3d.finger_scale_x,
        'finger_scale_y': scale_default if getattr(cfg.stereo_3d, 'finger_scale_y', None) is None else cfg.stereo_3d.finger_scale_y,
        'flip_x': cfg.stereo_3d.flip_x,
        'flip_y': cfg.stereo_3d.flip_y,
        'offset_y': 0.0,
        'show_spiral': True
    }

    # Get first frame to determine dimensions
    color0, _, _ = rsio.get_aligned()
    if color0 is None:
        raise RuntimeError('Failed to get camera frame')

    cam_h, cam_w = color0.shape[:2]
    scale_x_px = 1920 / cam_w
    scale_y_px = 1080 / cam_h

    print(f"✓ Camera: {cam_w}x{cam_h}")
    print(f"✓ Pixel rescale factors (to 1920×1080 per eye): {scale_x_px:.3f}×, {scale_y_px:.3f}×\n")

    # Create window
    cv2.namedWindow('Debug Tracking Visualization', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Debug Tracking Visualization', 1920, 540)

    try:
        while True:
            color, depth, t_now = rsio.get_aligned()
            if color is None:
                continue

            # LEFT: camera view
            camera_view = color.copy()

            # Track in camera space
            pt = tracker.track(color)

            if pt is not None:
                x_cam, y_cam = pt
                draw_crosshair(camera_view, x_cam, y_cam, (0, 0, 255), size=40, thickness=3)
                cv2.putText(camera_view, f"Camera: ({x_cam:.0f}, {y_cam:.0f})",
                            (int(x_cam) + 50, int(y_cam) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # === Transform to display space ===
                # Step 1: pixel resize to per‑eye 1920×1080
                x = x_cam * scale_x_px
                y = y_cam * scale_y_px

                # Step 2: center‑based FOV scaling
                cx = 1920 / 2.0
                cy = 1080 / 2.0
                x = cx + (x - cx) * settings['finger_scale_x']
                y = cy + (y - cy) * settings['finger_scale_y']

                # Step 3: vertical offset
                y = y + settings['offset_y']

                # Step 4: optional flips
                if settings['flip_x']:
                    x = 1920 - x
                if settings['flip_y']:
                    y = 1080 - y
            else:
                x, y = None, None

            # RIGHT: stereo view
            if settings['show_spiral']:
                stereo_view = spiral.draw_stereo(
                    color_bgr=tuple(cfg.spiral.color_bgr),
                    thickness=cfg.spiral.line_thickness
                )
            else:
                stereo_view = np.zeros((1080, 3840, 3), dtype=np.uint8)

            # Draw transformed point (magenta) on each eye appropriately (helper already handles)
            if pt is not None and x is not None:
                spiral.draw_point_on_spiral(stereo_view, x, y, color=(255, 0, 255), radius=20)
                # Reference centers
                cv2.circle(stereo_view, (1920 // 2, 1080 // 2), 10, (0, 255, 255), 2)
                cv2.circle(stereo_view, (1920 // 2 + 1920, 1080 // 2), 10, (0, 255, 255), 2)

            # Compose views
            camera_view_resized = cv2.resize(camera_view, (960, 540))
            stereo_view_resized = cv2.resize(stereo_view, (1920, 540))
            combined = np.hstack([camera_view_resized, stereo_view_resized])

            # Overlays
            info_lines = [
                f"Tracker: {tracker_name}",
                f"Scale: {settings['finger_scale_x']:.3f} (x), {settings['finger_scale_y']:.3f} (y)",
                f"Offset Y: {settings['offset_y']:.0f}px",
                f"Flip: X={settings['flip_x']}  Y={settings['flip_y']}",
                f"Disp H‑FOV: {disp_hfov_deg:.1f}°  Cam H‑FOV: {cam_hfov_deg:.1f}°",
                f"Profile: {headset_name}  Diag: {diag_fov_deg:.1f}°",
            ]
            if pt is not None:
                info_lines.append(f"Camera XY: ({x_cam:.0f}, {y_cam:.0f})  →  Display XY: ({x:.0f}, {y:.0f})")
            else:
                info_lines.append("NO TRACKING")

            draw_info_text(combined, info_lines, start_y=30)
            draw_calibration_bar(combined, disp_hfov_deg, cam_hfov_deg)

            # Labels
            cv2.putText(combined, 'RAW CAMERA VIEW', (10, combined.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined, 'STEREO DISPLAY VIEW', (970, combined.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('Debug Tracking Visualization', combined)

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('m'):
                use_led = not use_led
                tracker = led_tracker if use_led else mp_tracker
                tracker_name = 'LED' if use_led else 'MediaPipe'
                print(f"\n→ Switched to {tracker_name} tracking")
            elif key == ord('s'):
                settings['show_spiral'] = not settings['show_spiral']
                print(f"\n→ Spiral: {'ON' if settings['show_spiral'] else 'OFF'}")
            elif key == ord('f'):
                settings['flip_x'] = not settings['flip_x']
                print(f"\n→ Flip X: {settings['flip_x']}")
            elif key == ord('F'):
                settings['flip_y'] = not settings['flip_y']
                print(f"\n→ Flip Y: {settings['flip_y']}")
            elif key in (ord('='), ord('+')):
                settings['finger_scale_x'] += 0.02
                print(f"\n→ Scale X: {settings['finger_scale_x']:.3f}")
            elif key in (ord('-'), ord('_')):
                settings['finger_scale_x'] = max(0.1, settings['finger_scale_x'] - 0.02)
                print(f"\n→ Scale X: {settings['finger_scale_x']:.3f}")
            elif key == ord(']'):
                settings['finger_scale_y'] += 0.02
                print(f"\n→ Scale Y: {settings['finger_scale_y']:.3f}")
            elif key == ord('['):
                settings['finger_scale_y'] = max(0.1, settings['finger_scale_y'] - 0.02)
                print(f"\n→ Scale Y: {settings['finger_scale_y']:.3f}")
            elif key == ord('u'):
                settings['offset_y'] += 10
                print(f"\n→ Offset Y: {settings['offset_y']:.0f}px")
            elif key == ord('d'):
                settings['offset_y'] -= 10
                print(f"\n→ Offset Y: {settings['offset_y']:.0f}px")
            elif key == ord('g'):
                # Cycle headset profiles
                curr_idx = next((i for i, p in enumerate(HEADSET_PROFILES) if p['name'] in headset_name), -1)
                next_idx = (curr_idx + 1) % len(HEADSET_PROFILES)
                profile = HEADSET_PROFILES[next_idx]
                diag_fov_deg = profile['diag_fov_deg']
                headset_name = profile['name'] + '(hot)'
                disp_hfov_deg = horizontal_fov_from_diagonal(diag_fov_deg, ASPECT_W, ASPECT_H)
                scale_default = compute_scale(disp_hfov_deg, cam_hfov_deg)
                settings['finger_scale_x'] = scale_default
                settings['finger_scale_y'] = scale_default
                print(f"\n→ Headset: {headset_name} | diag {diag_fov_deg:.1f}° → H {disp_hfov_deg:.2f}° | scale {scale_default:.3f}")
            elif key == ord('c'):
                # Recompute scale from current FOVs
                scale_default = compute_scale(disp_hfov_deg, cam_hfov_deg)
                settings['finger_scale_x'] = scale_default
                settings['finger_scale_y'] = scale_default
                print(f"\n→ Recomputed scale = {scale_default:.3f}")
            elif key == ord('p'):
                print(f"\n[FOV] display H {disp_hfov_deg:.2f}°, camera H {cam_hfov_deg:.2f}°, scale {settings['finger_scale_x']:.3f}")
            elif key == ord('r'):
                settings['finger_scale_x'] = scale_default
                settings['finger_scale_y'] = scale_default
                settings['offset_y'] = 0.0
                print(f"\n→ Reset to defaults: scale {scale_default:.3f}, offset_y 0")

    except KeyboardInterrupt:
        print('\nInterrupted by user')
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print('\n' + '=' * 80)
        print('FINAL SETTINGS:')
        print('=' * 80)
        print(f"  finger_scale_x: {settings['finger_scale_x']:.3f}")
        print(f"  finger_scale_y: {settings['finger_scale_y']:.3f}")
        print(f"  offset_y: {settings['offset_y']:.0f}")
        print(f"  flip_x: {settings['flip_x']}")
        print(f"  flip_y: {settings['flip_y']}")
        print('Copy these to config.yaml if they work well!')
        print('=' * 80)

        # Cleanup
        try:
            mp_tracker.close()
            rsio.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
