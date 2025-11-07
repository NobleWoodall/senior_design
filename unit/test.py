# realsense_color_1080p.py
# Show Intel RealSense D435 COLOR stream at 1920x1080 @ 30 fps

import sys
import time
import pyrealsense2 as rs
import numpy as np
import cv2

TARGET_W, TARGET_H, FPS = 1920, 1080, 30

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    # Only enable the color stream (RGB) in BGR8 for OpenCV
    config.enable_stream(rs.stream.color, TARGET_W, TARGET_H, rs.format.bgr8, FPS)

    try:
        pipeline_profile = pipeline.start(config)
    except rs.error as e:
        print("[!] Failed to start RealSense pipeline at 1920x1080. "
              "Make sure your D435 is connected and supports this mode.\n"
              f"RealSense error: {e}")
        sys.exit(1)

    print(f"[OK] Streaming COLOR {TARGET_W}x{TARGET_H} @ {FPS} fps. Press 'q' to quit.")

    # Simple FPS meter (optional)
    last_t = time.time()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            frame_count += 1
            now = time.time()
            if now - last_t >= 1.0:
                fps = frame_count / (now - last_t)
                last_t = now
                frame_count = 0

            # Draw FPS text in the top-left
            cv2.putText(color_image, f"{fps:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow("D435 Color 1080p", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
