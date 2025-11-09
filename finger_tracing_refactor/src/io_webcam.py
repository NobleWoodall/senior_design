"""
Webcam fallback for testing without RealSense camera.
"""

import time
import cv2
import numpy as np
from typing import Tuple, Dict, Any


class WebcamIO:
    """
    Simple webcam wrapper that mimics RealSenseIO interface.
    Used for testing when RealSense camera is not available.
    """

    def __init__(self, width, height, fps, depth_width=None, depth_height=None, depth_fps=None,
                 use_auto_exposure=True, exposure=100):
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.start_time = None

    def start(self):
        """Start webcam capture."""
        print("[WebcamIO] RealSense not found - using computer webcam for testing")
        self.cap = cv2.VideoCapture(0)  # Default webcam

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.start_time = time.monotonic()

        # Warm up
        for _ in range(5):
            self.cap.read()

        print(f"[WebcamIO] Started at {self.width}x{self.height}@{self.fps}fps")

    def stop(self):
        """Stop webcam capture."""
        if self.cap:
            self.cap.release()
        print("[WebcamIO] Stopped")

    def get_aligned(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get color frame and fake depth.

        Returns:
            (color_frame, depth_frame, timestamp)
            - color_frame: BGR image from webcam
            - depth_frame: Fake depth (all 400mm) for compatibility
            - timestamp: Current time
        """
        if not self.cap or not self.cap.isOpened():
            return None, None, time.monotonic()

        ret, color = self.cap.read()
        t = time.monotonic()

        if not ret or color is None:
            return None, None, t

        # Resize if needed
        if color.shape[1] != self.width or color.shape[0] != self.height:
            color = cv2.resize(color, (self.width, self.height))

        # Create fake depth map (constant 400mm for all pixels)
        # This allows depth-based features to work without crashing
        depth = np.full((self.height, self.width), 400, dtype=np.uint16)

        return color, depth, t

    def get_intrinsics(self) -> Dict[str, Any]:
        """
        Get fake camera intrinsics for compatibility.

        Returns approximate webcam intrinsics.
        """
        # Typical webcam FOV ~60 degrees
        fx = fy = self.width / (2 * np.tan(np.radians(60) / 2))
        cx = self.width / 2
        cy = self.height / 2

        return {
            "color": {
                "width": self.width,
                "height": self.height,
                "fx": fx,
                "fy": fy,
                "ppx": cx,
                "ppy": cy,
                "model": "webcam_approximation",
                "coeffs": [0.0] * 5
            },
            "depth": {
                "width": self.width,
                "height": self.height,
                "fx": fx,
                "fy": fy,
                "ppx": cx,
                "ppy": cy,
                "model": "fake_depth",
                "coeffs": [0.0] * 5,
                "depth_scale": 0.001  # 1mm per unit
            }
        }


def create_camera_io(width, height, fps, depth_width, depth_height, depth_fps,
                     use_auto_exposure=True, exposure=100):
    """
    Factory function to create camera IO - tries RealSense first, falls back to webcam.

    Args:
        width, height, fps: Color stream settings
        depth_width, depth_height, depth_fps: Depth stream settings
        use_auto_exposure: Enable auto exposure
        exposure: Manual exposure value

    Returns:
        Camera IO object (RealSenseIO or WebcamIO)
    """
    try:
        import pyrealsense2 as rs
        from .io_rs import RealSenseIO

        # Check if RealSense is connected
        ctx = rs.context()
        if len(ctx.devices) > 0:
            print("[Camera] RealSense detected - using RealSense")
            return RealSenseIO(width, height, fps, depth_width, depth_height, depth_fps,
                             use_auto_exposure, exposure)
        else:
            print("[Camera] No RealSense detected - falling back to webcam")
            return WebcamIO(width, height, fps, depth_width, depth_height, depth_fps,
                          use_auto_exposure, exposure)

    except Exception as e:
        print(f"[Camera] RealSense initialization failed ({e}) - falling back to webcam")
        return WebcamIO(width, height, fps, depth_width, depth_height, depth_fps,
                      use_auto_exposure, exposure)
