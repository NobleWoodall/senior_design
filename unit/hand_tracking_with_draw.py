import sys
import time
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
MAX_NUM_HANDS = 1
DET_CONF = .9
TRK_CONF = .9
MODEL_COMPLEXITY = 1

DEPTH_KERNEL = 5
EMA_ALPHA = 0.35
SHOW_FPS = True

# Drawing params
LINE_THICKNESS = 3
LINE_COLOR = (0, 255, 255)  # BGR (yellow)
MAX_GAP_PX = 60

# Auto-clear every N seconds (set to 0 to disable by default; toggle with 't')
CLEAR_INTERVAL_SEC = 10.0

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def median_of_valid(arr):
    valid = arr[arr > 0]
    return float(np.median(valid)) if valid.size > 0 else 0.0

def depth_at_pixel_averaged(depth_frame, x, y, k=5):
    h, w = depth_frame.get_height(), depth_frame.get_width()
    half = k // 2
    x0, x1 = max(0, x - half), min(w - 1, x + half)
    y0, y1 = max(0, y - half), min(h - 1, y + half)

    depth = np.asanyarray(depth_frame.get_data())
    window = depth[y0:y1+1, x0:x1+1]
    if window.size == 0:
        return 0.0
    return median_of_valid(window.astype(np.float32))

def deproject_pixel_to_point(intrinsics, pixel, depth_m):
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [float(pixel[0]), float(pixel[1])], float(depth_m))
    return float(X), float(Y), float(Z)

class ExponentialSmoother:
    def __init__(self, alpha):
        self.alpha = alpha
        self.prev = {}

    def update(self, key, xyz):
        if key not in self.prev:
            self.prev[key] = xyz
            return xyz
        px, py, pz = self.prev[key]
        x, y, z = xyz
        nx = self.alpha * x + (1 - self.alpha) * px
        ny = self.alpha * y + (1 - self.alpha) * py
        nz = self.alpha * z + (1 - self.alpha) * pz
        self.prev[key] = (nx, ny, nz)
        return self.prev[key]

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    # RealSense setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_stream.get_intrinsics()

    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=DET_CONF,
        min_tracking_confidence=TRK_CONF,
        model_complexity=MODEL_COMPLEXITY
    )

    # Helpers
    fps_hist = deque(maxlen=30)
    t_prev = time.time()
    smoother = ExponentialSmoother(EMA_ALPHA)

    INDEX_TIP_IDX = 8  # pointer fingertip

    drawing_enabled = True
    prev_point = None
    save_count = 0
    scribble_layer = None

    auto_clear_enabled = CLEAR_INTERVAL_SEC > 0
    last_clear_time = time.time()

    print("Keys: p=pause/resume, c=clear, s=save, t=toggle auto-clear, q=quit")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            h, w, _ = color_image.shape

            if scribble_layer is None:
                scribble_layer = np.zeros_like(color_image)

            # Auto-clear timer
            now = time.time()
            if auto_clear_enabled and CLEAR_INTERVAL_SEC > 0 and (now - last_clear_time) >= CLEAR_INTERVAL_SEC:
                scribble_layer[:] = 0
                prev_point = None
                last_clear_time = now

            # Hand tracking (only pointer finger)
            rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            fingertip_px = None

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark[INDEX_TIP_IDX]
                u = int(np.clip(lm.x * w, 0, w - 1))
                v = int(np.clip(lm.y * h, 0, h - 1))
                fingertip_px = (u, v)

                # depth/Z (optional)
                depth_units = depth_at_pixel_averaged(depth_frame, u, v, k=DEPTH_KERNEL)
                depth_m = depth_units * depth_scale if depth_units > 0 else 0.0
                if depth_m == 0.0 and DEPTH_KERNEL < 9:
                    depth_units = depth_at_pixel_averaged(depth_frame, u, v, k=9)
                    depth_m = depth_units * depth_scale if depth_units > 0 else 0.0
                if depth_m > 0:
                    X, Y, Z = deproject_pixel_to_point(color_intr, (u, v), depth_m)
                    X, Y, Z = smoother.update(INDEX_TIP_IDX, (X, Y, Z))
                    cv2.putText(color_image, f"Z={Z/.0254:.5f}m", (u+6, v-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                # fingertip marker only
                cv2.circle(color_image, (u, v), 6, (0, 255, 255), -1)
            else:
                prev_point = None  # break stroke when hand is lost

            # Draw line following index fingertip
            if drawing_enabled and fingertip_px is not None:
                if prev_point is not None:
                    dx = fingertip_px[0] - prev_point[0]
                    dy = fingertip_px[1] - prev_point[1]
                    if (dx*dx + dy*dy) ** 0.5 < MAX_GAP_PX:
                        cv2.line(scribble_layer, prev_point, fingertip_px, LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA)
                prev_point = fingertip_px
            elif fingertip_px is not None:
                prev_point = fingertip_px

            # FPS
            if SHOW_FPS:
                t_now = time.time()
                fps = 1.0 / max(1e-6, (t_now - t_prev))
                t_prev = t_now
                fps_hist.append(fps)
                fps_smoothed = sum(fps_hist) / len(fps_hist)
                cv2.putText(color_image, f"FPS: {fps_smoothed:5.1f}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Auto-clear countdown overlay
            if auto_clear_enabled and CLEAR_INTERVAL_SEC > 0:
                remaining = max(0.0, CLEAR_INTERVAL_SEC - (now - last_clear_time))
                cv2.putText(color_image, f"Auto-clear in: {remaining:4.1f}s  (t to toggle)",
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Composite
            composed = cv2.addWeighted(color_image, 1.0, scribble_layer, 1.0, 0)
            cv2.imshow("Pointer-Finger Drawing — RealSense + MediaPipe", composed)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                scribble_layer[:] = 0
                prev_point = None
                last_clear_time = time.time()
            elif key == ord('p'):
                drawing_enabled = not drawing_enabled
            elif key == ord('s'):
                out_name = f"drawing_{save_count:03d}.png"
                cv2.imwrite(out_name, scribble_layer)
                print(f"Saved: {out_name}")
                save_count += 1
            elif key == ord('t'):
                auto_clear_enabled = not auto_clear_enabled
                last_clear_time = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.exit(main())
