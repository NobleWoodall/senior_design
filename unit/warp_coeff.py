import sys
import time
import json
import numpy as np
import cv2
import pyrealsense2 as rs

# =========================
# Distortion utilities
# =========================
def build_distortion_map(w, h, k1, k2, k3, cx, cy, scale):
    """
    Create cv2.remap grids to apply *pincushion pre-distortion*.
    (We map target pixels to source pixels.)
    """
    # Pixel grid (target)
    xs = np.linspace(0, w - 1, w, dtype=np.float32)
    ys = np.linspace(0, h - 1, h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    # Normalize around lens center to roughly [-1..1] using max dimension
    norm = max(w, h) * 0.5
    x = (grid_x - cx) / norm
    y = (grid_y - cy) / norm

    # Uniform zoom
    x *= scale
    y *= scale

    r2 = x * x + y * y
    f = 1.0 + k1 * r2 + k2 * (r2 * r2) + k3 * (r2 * r2 * r2)
    xd = x * f
    yd = y * f

    # Back to source pixel coords
    map_x = (xd * norm) + cx
    map_y = (yd * norm) + cy

    return map_x.astype(np.float32), map_y.astype(np.float32)

def draw_checkerboard(img, step=40, color=(80, 80, 80)):
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h - 1), color, 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w - 1, y), color, 1)

def put_overlay_text(img, params):
    h, w = img.shape[:2]
    x0, y0 = 10, 24
    dy = 24
    lines = [
        f"K1={params['k1']:.3f}  K2={params['k2']:.3f}  K3={params['k3']:.3f}",
        f"Scale={params['scale']:.3f}  Center=({int(params['center_x'])},{int(params['center_y'])})  IPDpx={int(params['ipd_px'])}",
        "Keys: [g]=grid  [s]=save  [q/Esc]=quit"
    ]
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x0, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# =========================
# Trackbar helpers
# =========================
def tb_get_float(win, name, scale, offset=0.0):
    # Returns (trackbar_value/scale) + offset
    v = cv2.getTrackbarPos(name, win)
    return (v / scale) + offset

def tb_get_int(win, name):
    return cv2.getTrackbarPos(name, win)

def create_controls_window(W_eye, H):
    win = "Controls"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 400, 320)

    # K1,K2,K3 in [-1.000 .. +1.000] via integer trackbar [-1000..1000]
    cv2.createTrackbar("K1 x1000", win,  -250, 2000, lambda v: None)  # -0.250 default
    cv2.createTrackbar("K2 x1000", win,    50, 2000, lambda v: None)  #  0.050 default
    cv2.createTrackbar("K3 x1000", win,   -20, 2000, lambda v: None)  # -0.020 default

    # Scale in [0.50 .. 2.00]  (trackbar 50..200)
    cv2.createTrackbar("Scale x100", win, 105,  300, lambda v: None)  # 1.05 default

    # Center X/Y in pixel coordinates of the *eye image*
    cv2.createTrackbar("CenterX", win, int(W_eye // 2), W_eye, lambda v: None)
    cv2.createTrackbar("CenterY", win, int(H // 2),     H,     lambda v: None)

    # IPD in pixels (shifts left/right lens centers horizontally)
    # Rough bound: up to 1/2 of eye width
    cv2.createTrackbar("IPD px", win, int(W_eye * 0.18), int(W_eye // 2), lambda v: None)
    return win

# =========================
# Main
# =========================
def main():
    # ---- RealSense color stream only ----
    pipeline = rs.pipeline()
    config = rs.config()

    # Base camera capture size (can be small; we’ll scale as needed)
    BASE_W, BASE_H, FPS = 640, 480, 30
    config.enable_stream(rs.stream.color, BASE_W, BASE_H, rs.format.bgr8, FPS)

    profile = pipeline.start(config)

    # ---- Eye buffer size (what each eye sees before warp) ----
    # You can increase this if you want more resolution per eye.
    W_eye, H_eye = BASE_W, BASE_H  # per-eye resolution
    # Output window will be (2*W_eye, H_eye)

    # Create controls window and defaults
    ctrl_win = create_controls_window(W_eye, H_eye)

    # State
    show_grid = False

    # Prebuilt maps cache
    last_params = None
    maps = {}  # ("L" or "R") -> (map_x, map_y)

    def get_params():
        k1 = tb_get_float(ctrl_win, "K1 x1000", 1000.0)
        k2 = tb_get_float(ctrl_win, "K2 x1000", 1000.0)
        k3 = tb_get_float(ctrl_win, "K3 x1000", 1000.0)
        scale = tb_get_float(ctrl_win, "Scale x100", 100.0)  # 0.50..3.00
        cx = float(tb_get_int(ctrl_win, "CenterX"))
        cy = float(tb_get_int(ctrl_win, "CenterY"))
        ipd_px = float(tb_get_int(ctrl_win, "IPD px"))
        return dict(k1=k1, k2=k2, k3=k3, scale=scale, center_x=cx, center_y=cy, ipd_px=ipd_px)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())

            # Resize/crop to eye size if needed
            if (color.shape[1], color.shape[0]) != (W_eye, H_eye):
                eye_src = cv2.resize(color, (W_eye, H_eye), interpolation=cv2.INTER_AREA)
            else:
                eye_src = color

            params = get_params()

            # Rebuild maps if params changed
            if last_params is None or any(abs(params[k] - last_params[k]) > 1e-6 for k in params):
                # Left/right centers from base center ± half IPD (in pixels)
                cx_base = params['center_x']
                cy_base = params['center_y']
                half_ipd = params['ipd_px'] * 0.5

                cx_L = cx_base - half_ipd
                cx_R = cx_base + half_ipd
                cy_L = cy_R = cy_base

                maps['L'] = build_distortion_map(
                    W_eye, H_eye,
                    params['k1'], params['k2'], params['k3'],
                    cx_L, cy_L, params['scale']
                )
                maps['R'] = build_distortion_map(
                    W_eye, H_eye,
                    params['k1'], params['k2'], params['k3'],
                    cx_R, cy_R, params['scale']
                )
                last_params = params.copy()

            # Apply warp (same source for both eyes in this viewer)
            map_x_L, map_y_L = maps['L']
            map_x_R, map_y_R = maps['R']

            eye_L = cv2.remap(eye_src, map_x_L, map_y_L, interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            eye_R = cv2.remap(eye_src, map_x_R, map_y_R, interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            # Optional checkerboard overlay (use it to tune straightness at edges)
            if show_grid:
                draw_checkerboard(eye_L, step=40)
                draw_checkerboard(eye_R, step=40)

            # Compose stereo side-by-side
            stereo = np.hstack((eye_L, eye_R))

            # HUD (draw once across stereo view)
            hud = stereo.copy()
            put_overlay_text(hud, params)

            cv2.imshow("VR Stereo Pre-distortion (Left | Right)", hud)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q or Esc
                break
            elif key == ord('g'):
                show_grid = not show_grid
            elif key == ord('s'):
                # Save to JSON
                save = params.copy()
                with open("warp_params.json", "w") as f:
                    json.dump(save, f, indent=2)
                print("[Saved] warp_params.json")

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.exit(main())
