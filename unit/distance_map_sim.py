import pyrealsense2 as rs
import numpy as np
import cv2
def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            # Now depth_frame is a rs.depth_frame and supports get_width/height
            w = depth_frame.get_width()
            h = depth_frame.get_height()
            cx, cy = w // 2, h // 2

            # Sample distance at the center
            dist_m = depth_frame.get_distance(cx, cy)

            if 0.0 < dist_m <= 1.0:
                print(f"{dist_m:.3f} m")

            # Quick preview
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_vis = cv2.convertScaleAbs(depth_image, alpha=255/1000.0)  # normalize 0–1m
            cv2.circle(depth_vis, (cx, cy), 4, (255, 255, 255), 1)
            cv2.imshow("Depth (0–1 m scaled)", depth_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
