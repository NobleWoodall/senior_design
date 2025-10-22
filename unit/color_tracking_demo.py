import argparse
import csv
import time
from collections import deque

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    RS_AVAILABLE = True
except Exception:
    RS_AVAILABLE = False


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h-low", type=int, default=0)
    ap.add_argument("--s-low", type=int, default=80)
    ap.add_argument("--v-low", type=int, default=80)
    ap.add_argument("--h-high", type=int, default=179)
    ap.add_argument("--s-high", type=int, default=255)
    ap.add_argument("--v-high", type=int, default=255)
    ap.add_argument("--median-k", type=int, default=5, help="Median blur kernel (odd).")
    ap.add_argument("--morph-k", type=int, default=3, help="Morphology kernel size.")
    ap.add_argument("--depth", action="store_true", help="Enable depth + alignment for depth readout.")
    ap.add_argument("--output", type=str, default="color_track.csv", help="CSV to save (t, x, y, depth?).")
    ap.add_argument("--max-pts", type=int, default=512, help="Tail to draw the last N positions.")
    ap.add_argument("--fps", type=int, default=30)
    return ap.parse_args()


# ---------- Trackbar helpers ----------
CTRL_WIN = "controls"

def _ensure_bounds(low_name, high_name, max_val):
    """Keep low <= high by nudging counterpart if needed."""
    low = cv2.getTrackbarPos(low_name, CTRL_WIN)
    high = cv2.getTrackbarPos(high_name, CTRL_WIN)
    changed = False
    if low > high:
        # push the one the user didn't touch last; just set low = high
        cv2.setTrackbarPos(low_name, CTRL_WIN, high)
        low = high
        changed = True
    if high < low:
        cv2.setTrackbarPos(high_name, CTRL_WIN, low)
        high = low
        changed = True
    # Clamp (shouldn't be necessary with OpenCV trackbars but safe)
    low = max(0, min(max_val, low))
    high = max(0, min(max_val, high))
    return low, high, changed


def init_trackbars(args):
    """Create trackbars with initial values from args."""
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WIN, 420, 350)

    # HSV low/high
    cv2.createTrackbar("H_low", CTRL_WIN, args.h_low, 179, lambda v: None)
    cv2.createTrackbar("H_high", CTRL_WIN, args.h_high, 179, lambda v: None)
    cv2.createTrackbar("S_low", CTRL_WIN, args.s_low, 255, lambda v: None)
    cv2.createTrackbar("S_high", CTRL_WIN, args.s_high, 255, lambda v: None)
    cv2.createTrackbar("V_low", CTRL_WIN, args.v_low, 255, lambda v: None)
    cv2.createTrackbar("V_high", CTRL_WIN, args.v_high, 255, lambda v: None)

    # Optional: tunables for smoothing/cleanup
    cv2.createTrackbar("Median_k(odd)", CTRL_WIN, args.median_k, 21, lambda v: None)   # odd 1..21
    cv2.createTrackbar("Morph_k", CTRL_WIN, args.morph_k, 15, lambda v: None)          # 1..15


def read_trackbars():
    """Read and sanitize all slider values."""
    h_low, h_high, _ = _ensure_bounds("H_low", "H_high", 179)
    s_low, s_high, _ = _ensure_bounds("S_low", "S_high", 255)
    v_low, v_high, _ = _ensure_bounds("V_low", "V_high", 255)

    median_k = cv2.getTrackbarPos("Median_k(odd)", CTRL_WIN)
    if median_k < 1:
        median_k = 1
    # force odd
    if median_k % 2 == 0:
        median_k += 1
        if median_k > 21:
            median_k = 21
        cv2.setTrackbarPos("Median_k(odd)", CTRL_WIN, median_k)

    morph_k = cv2.getTrackbarPos("Morph_k", CTRL_WIN)
    morph_k = max(1, morph_k)

    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, s_high, v_high], dtype=np.uint8)

    return lower, upper, median_k, morph_k


def main():
    args = parse_args()
    if not RS_AVAILABLE:
        raise RuntimeError("pyrealsense2 not available in this environment. Use a RealSense-enabled machine to run.")

    # Configure RealSense
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, args.fps)
    if args.depth:
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, args.fps)

    prof = pipe.start(cfg)

    align = None
    depth_scale = None
    if args.depth:
        align = rs.align(rs.stream.color)
        dev = prof.get_device()
        depth_sensor = dev.first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

    # CSV logging
    fout = open(args.output, "w", newline="")
    writer = csv.writer(fout)
    header = ["t", "x", "y"]
    if args.depth:
        header.append("z_m")
    writer.writerow(header)

    pts = deque(maxlen=args.max_pts)

    # --- GUI controls ---
    init_trackbars(args)

    print("[INFO] Press 'q' to quit. Press 'p' to print current HSV/morph settings.")
    try:
        t0 = time.time()
        while True:
            frames = pipe.wait_for_frames()
            if args.depth:
                frames = align.process(frames)

            c = frames.get_color_frame()
            if not c:
                continue
            color = np.asanyarray(c.get_data())

            d = None
            depth_img = None
            if args.depth:
                d = frames.get_depth_frame()
                if d:
                    depth_img = np.asanyarray(d.get_data())

            # ---- Read sliders ----
            lower, upper, median_k, morph_k = read_trackbars()
            morph_kernel = np.ones((morph_k, morph_k), dtype=np.uint8)

            # HSV threshold
            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)

            if median_k > 1:
                mask = cv2.medianBlur(mask, median_k)

            # Morph open+close to clean
            if morph_k > 1:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel, iterations=1)

            # Largest contour as target
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cx = cy = None
            if cnts:
                largest = max(cnts, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] > 1e-3:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

            t = time.time() - t0
            depth_val_m = None
            if cx is not None and cy is not None:
                pts.append((cx, cy))
                if depth_img is not None:
                    z = depth_img[cy, cx].astype(np.float32) * depth_scale
                    depth_val_m = float(z) if z > 0 else None

                row = [f"{t:.6f}", cx, cy]
                if args.depth:
                    row.append("" if depth_val_m is None else f"{depth_val_m:.4f}")
                writer.writerow(row)

                # Draw point and trail
                cv2.circle(color, (cx, cy), 7, (0, 0, 255), -1)
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    cv2.line(color, pts[i - 1], pts[i], (255, 255, 255), 2)

            # Display
            cv2.imshow("color", color)
            cv2.imshow("mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                print(f"HSV lower={lower.tolist()} upper={upper.tolist()} median_k={median_k} morph_k={morph_k}")

    finally:
        pipe.stop()
        fout.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
