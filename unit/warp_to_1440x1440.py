# square_crop_realsense_1440_locked.py
# Requires: pyrealsense2, opencv-python, numpy

import cv2
import numpy as np
import pyrealsense2 as rs

TARGET = 1440
COLOR_W, COLOR_H, FPS = 1920, 1080, 30
SQUARE = min(COLOR_W, COLOR_H)  # 1080

def clamp(v, lo, hi): return max(lo, min(hi, v))

def compute_square_roi(cx, cy, w, h, side):
    """Square ROI centered at (cx,cy) but clamped to image bounds."""
    half = side // 2
    left = int(round(cx - half))
    top  = int(round(cy - half))
    left = clamp(left, 0, w - side)
    top  = clamp(top,  0, h - side)
    return left, top, side, side

def adjust_intrinsics_for_crop_and_scale(K, roi, scale):
    """Adjust intrinsics (fx, fy, cx, cy) after crop then scale."""
    left, top, _, _ = roi
    fx = K.fx * scale
    fy = K.fy * scale
    cx = (K.ppx - left) * scale
    cy = (K.ppy - top)  * scale
    return fx, fy, cx, cy

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        print("Failed to start RealSense pipeline:", e)
        return

    # Get initial color intrinsics
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    K = color_stream.get_intrinsics()

    # ROI: square centered at camera principal point (ppx, ppy)
    roi = compute_square_roi(K.ppx, K.ppy, COLOR_W, COLOR_H, SQUARE)
    scale = TARGET / float(SQUARE)

    printed_intrinsics = False
    win = "RealSense color (1440x1440 square crop)"
    # AUTOSIZE locks window size to image
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue

            img = np.asanyarray(color.get_data())  # HxWx3 BGR

            # Crop to square ROI
            left, top, w, h = roi
            crop = img[top:top+h, left:left+w]

            # Resize to 1440x1440
            out = cv2.resize(crop, (TARGET, TARGET), interpolation=cv2.INTER_LINEAR)

            if not printed_intrinsics:
                fx, fy, cx, cy = adjust_intrinsics_for_crop_and_scale(K, roi, scale)
                print("=== Adjusted intrinsics for 1440x1440 (after crop+scale) ===")
                print(f"fx: {fx:.4f}, fy: {fy:.4f}, cx: {cx:.4f}, cy: {cy:.4f}")
                printed_intrinsics = True

            cv2.imshow(win, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or q
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
