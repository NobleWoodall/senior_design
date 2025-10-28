import sys
import time
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs

# Optional, but recommended for joint detection
import mediapipe as mp

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
MAX_NUM_HANDS = 2
DET_CONF = 0.6
TRK_CONF = 0.6
MODEL_COMPLEXITY = 1  # 0=faster, 1=balanced, 2=better

DRAW_TEXT = True
DEPTH_KERNEL = 5   # odd integer; window for averaging depth around each joint
EMA_ALPHA = 0.35   # smooth 3D outputs a bit (0=no smoothing, 1=instant)
SHOW_FPS = True

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def median_of_valid(arr):
    """Return median of >0 entries, or 0 if none."""
    valid = arr[arr > 0]
    return float(np.median(valid)) if valid.size > 0 else 0.0

def depth_at_pixel_averaged(depth_frame, x, y, k=5):
    """
    Get a more stable depth estimate by averaging within a kxk window
    centered on (x, y). Uses median of valid (>0) depths to reduce noise.
    """
    h, w = depth_frame.get_height(), depth_frame.get_width()
    half = k // 2
    x0, x1 = max(0, x - half), min(w - 1, x + half)
    y0, y1 = max(0, y - half), min(h - 1, y + half)

    # Convert the window to a numpy array of uint16 depth units
    depth = np.asanyarray(depth_frame.get_data())
    window = depth[y0:y1+1, x0:x1+1]
    if window.size == 0:
        return 0.0

    return median_of_valid(window.astype(np.float32))  # still in depth units

def deproject_pixel_to_point(intrinsics, pixel, depth_m):
    """
    Given camera intrinsics, a pixel (u, v) in the color frame, and depth in meters,
    compute the 3D point (X, Y, Z) in meters in the camera coordinate system.
    """
    # rs2_deproject_pixel_to_point expects depth in meters and pixel in [u, v]
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [float(pixel[0]), float(pixel[1])], float(depth_m))
    return float(X), float(Y), float(Z)

class ExponentialSmoother:
    """Simple EMA smoother for per-landmark 3D points."""
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
    # ---------------- RealSense setup ----------------
    pipeline = rs.pipeline()
    config = rs.config()

    # Depth + color (widely supported modes)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # Enable auto-exposure on color sensor for proper brightness
    color_sensor = profile.get_device().first_color_sensor()
    color_sensor.set_option(rs.option.enable_auto_exposure, True)
    # Optionally set a fixed higher exposure instead (uncomment if auto doesn't work well):
    # color_sensor.set_option(rs.option.enable_auto_exposure, False)
    # color_sensor.set_option(rs.option.exposure, 300)

    # Align depth to color frame for pixel-wise correspondence
    align = rs.align(rs.stream.color)

    # Get depth scale (to convert raw units to meters)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # e.g., ~0.001 for meters per unit

    # We'll need color intrinsics (after streaming started)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_stream.get_intrinsics()

    # ---------------- MediaPipe Hands ----------------
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=DET_CONF,
        min_tracking_confidence=TRK_CONF,
        model_complexity=MODEL_COMPLEXITY
    )
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # ---------------- Helpers ----------------
    fps_hist = deque(maxlen=30)
    t_prev = time.time()
    smoother = ExponentialSmoother(EMA_ALPHA)

    landmark_names = [
        "WRIST",
        "THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
        "INDEX_MCP","INDEX_PIP","INDEX_DIP","INDEX_TIP",
        "MIDDLE_MCP","MIDDLE_PIP","MIDDLE_DIP","MIDDLE_TIP",
        "RING_MCP","RING_PIP","RING_DIP","RING_TIP",
        "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP"
    ]

    print("Press 'q' to quit.")
    try:
        while True:
            # ----- Acquire frames -----
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            h, w, _ = color_image.shape

            # Run MediaPipe
            rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            joints_world = {}  # {name: (X,Y,Z) in meters}

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Draw connections for visualization
                    mp_draw.draw_landmarks(
                        color_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

                    for idx, lm in enumerate(hand_landmarks.landmark):
                        # Convert normalized coords to pixel coords
                        u = int(np.clip(lm.x * w, 0, w - 1))
                        v = int(np.clip(lm.y * h, 0, h - 1))

                        # Robust depth lookup (median in small window), in raw units
                        depth_units = depth_at_pixel_averaged(depth_frame, u, v, k=DEPTH_KERNEL)

                        # Convert to meters
                        depth_m = depth_units * depth_scale if depth_units > 0 else 0.0

                        # If no depth found (0), try a slightly larger window once
                        if depth_m == 0.0 and DEPTH_KERNEL < 9:
                            depth_units = depth_at_pixel_averaged(depth_frame, u, v, k=9)
                            depth_m = depth_units * depth_scale if depth_units > 0 else 0.0

                        # Deproject to 3D (if we have a valid depth)
                        if depth_m > 0:
                            X, Y, Z = deproject_pixel_to_point(color_intr, (u, v), depth_m)
                            X, Y, Z = smoother.update(idx, (X, Y, Z))
                            joints_world[landmark_names[idx]] = (X, Y, Z)

                            # Overlay marker + Z
                            cv2.circle(color_image, (u, v), 4, (0, 255, 0), -1)
                            if DRAW_TEXT:
                                z_txt = f"Z={Z:.2f}m"
                                cv2.putText(color_image, z_txt, (u+6, v-6),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                        else:
                            # Mark with a different color if depth invalid
                            cv2.circle(color_image, (u, v), 4, (0, 0, 255), -1)

                    # One hand only (MAX_NUM_HANDS controls count)
                    break

            # FPS display
            if SHOW_FPS:
                t_now = time.time()
                fps = 1.0 / max(1e-6, (t_now - t_prev))
                t_prev = t_now
                fps_hist.append(fps)
                fps_smoothed = sum(fps_hist) / len(fps_hist)
                cv2.putText(color_image, f"FPS: {fps_smoothed:5.1f}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Print 3D dictionary each frame (comment out if too chatty)
            if joints_world:
                # Example: {'WRIST': (X,Y,Z), 'THUMB_CMC': (...), ...}
                print(joints_world)

            cv2.imshow("RealSense Color (with 3D Hand Joints)", color_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.exit(main())
