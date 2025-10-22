# hsv_probe_realsense.py
# Shows HSV (and depth) at the mouse pointer for an Intel RealSense D435.

import sys
import time
from collections import deque

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("pyrealsense2 not found. Install with: pip install pyrealsense2")
    sys.exit(1)

# -----------------------------
# Config
# -----------------------------
WIN_NAME = "D435 HSV Probe (move mouse / click to lock)"
COLOR_WIDTH, COLOR_HEIGHT, FPS = 1280, 720, 30
HSV_KERNEL = 3          # odd integer; median window around the pointer (to reduce noise)
DEPTH_KERNEL = 5        # odd integer; median window around the pointer for depth smoothing
SHOW_DEPTH = True       # toggle depth readout overlay by default
FONT = cv2.FONT_HERSHEY_SIMPLEX

# -----------------------------
# State
# -----------------------------
mouse_x, mouse_y = COLOR_WIDTH // 2, COLOR_HEIGHT // 2
locked_samples = deque(maxlen=12)  # store recent locked samples for on-screen display
lock_readout = False

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def on_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y, lock_readout, locked_samples
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
    elif event == cv2.EVENT_LBUTTONDOWN:
        lock_readout = True  # take a snapshot this frame

def median_window(arr, cx, cy, k=3):
    """Return median of a kxk window around (cx,cy) for a single or multi-channel image."""
    k = max(1, k | 1)  # ensure odd
    h, w = arr.shape[:2]
    x0 = clamp(cx - k // 2, 0, w - 1)
    y0 = clamp(cy - k // 2, 0, h - 1)
    x1 = clamp(x0 + k, 1, w)
    y1 = clamp(y0 + k, 1, h)
    patch = arr[y0:y1, x0:x1]
    if patch.ndim == 2:
        return np.median(patch)
    else:
        return np.median(patch.reshape(-1, patch.shape[-1]), axis=0)

def main():
    global lock_readout, SHOW_DEPTH

    # -----------------------------
    # RealSense setup
    # -----------------------------
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
    # Depth is required to report depth under the pointer
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)

    # Start pipeline
    profile = pipeline.start(cfg)

    # Align depth to color
    align = rs.align(rs.stream.color)

    # Optional: depth scale for converting to meters
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # meters per unit

    # Window & mouse
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1000, 560)
    cv2.setMouseCallback(WIN_NAME, on_mouse)

    fps_clock = deque(maxlen=32)
    last_time = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Convert to numpy
            color = np.asanyarray(color_frame.get_data())  # BGR
            depth = np.asanyarray(depth_frame.get_data())  # uint16 (depth in camera units)

            # Prepare HSV image once per frame
            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

            # Clamp pointer to frame
            x = clamp(mouse_x, 0, color.shape[1] - 1)
            y = clamp(mouse_y, 0, color.shape[0] - 1)

            # Smooth/median around the pointer
            hsv_med = median_window(hsv, x, y, HSV_KERNEL).astype(np.float32)
            bgr_med = median_window(color, x, y, HSV_KERNEL).astype(np.float32)

            # Depth smoothing
            depth_med = median_window(depth, x, y, DEPTH_KERNEL)
            # Convert to mm (depth_scale is meters/unit)
            depth_mm = float(depth_med) * depth_scale * 1000.0 if depth_med > 0 else 0.0

            # Compose readout text
            H, S, V = int(round(hsv_med[0])), int(round(hsv_med[1])), int(round(hsv_med[2]))
            B, G, R = int(round(bgr_med[0])), int(round(bgr_med[1])), int(round(bgr_med[2]))
            readout = f"HSV: ({H}, {S}, {V})   BGR: ({B}, {G}, {R})"
            if SHOW_DEPTH:
                readout += f"   Depth: {depth_mm:.0f} mm"

            # Draw a small crosshair at the pointer
            cv2.drawMarker(color, (x, y), (0, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
            cv2.drawMarker(color, (x, y), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1)

            # FPS
            now = time.time()
            fps_clock.append(1.0 / max(1e-6, (now - last_time)))
            last_time = now
            fps = sum(fps_clock) / len(fps_clock)

            # Overlay text
            cv2.rectangle(color, (8, 8), (8 + 590, 8 + 70), (0, 0, 0), thickness=-1)
            cv2.putText(color, readout, (20, 40), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(color, f"FPS: {fps:.1f}   (q=quit, d=toggle depth, c=clear locks)", (20, 70),
                        FONT, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

            # If user left-clicked, store a locked sample
            if lock_readout:
                locked_samples.append({
                    "x": x, "y": y,
                    "HSV": (H, S, V),
                    "BGR": (B, G, R),
                    "depth_mm": int(round(depth_mm))
                })
                lock_readout = False

            # Draw locked samples in a side panel
            y0 = 100
            if locked_samples:
                cv2.putText(color, "Locked samples (latest at bottom):", (20, y0), FONT, 0.6, (0, 255, 255), 2)
                y0 += 28
                for i, s in enumerate(list(locked_samples)[-8:]):
                    txt = f"{i+1}. xy=({s['x']},{s['y']})  HSV={s['HSV']}  Depth={s['depth_mm']}mm"
                    cv2.putText(color, txt, (20, y0), FONT, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
                    y0 += 22

            cv2.imshow(WIN_NAME, color)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                SHOW_DEPTH = not SHOW_DEPTH
            elif key == ord('c'):
                locked_samples.clear()

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
