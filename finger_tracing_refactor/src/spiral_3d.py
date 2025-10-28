"""
3D Stereoscopic Spiral for XReal Glasses
Archimedean spiral with stereoscopic rendering at configurable depth.
"""
import numpy as np
import cv2
import math
from typing import Tuple, Dict


class Spiral3D:
    """3D stereoscopic spiral renderer for XReal glasses (3840x1080)."""

    # XReal Air/Air Pro parameters
    FULL_WIDTH = 3840
    FULL_HEIGHT = 1080
    EYE_WIDTH = FULL_WIDTH // 2  # 1920 per eye
    EYE_HEIGHT = FULL_HEIGHT     # 1080

    IPD_MM = 63.0                # Inter-pupillary distance
    FOV_HORIZONTAL_DEG = 46.0    # Horizontal field of view

    def __init__(self, a: float, b: float, turns: float, theta_step: float, target_depth_m: float = 0.5):
        """
        Initialize 3D stereoscopic spiral.

        Args:
            a, b: Archimedean spiral parameters (r = a + b*theta)
            turns: Number of spiral turns
            theta_step: Angular step for spiral generation
            target_depth_m: Perceived depth in meters (default 0.5m)
        """
        self.a = a
        self.b = b
        self.turns = turns
        self.theta_step = theta_step
        self.target_depth_m = target_depth_m

        # Calculate stereoscopic parameters
        self.focal_length_px = self.EYE_WIDTH / (2.0 * math.tan(math.radians(self.FOV_HORIZONTAL_DEG / 2.0)))
        self.disparity_px = (self.IPD_MM * self.focal_length_px) / (self.target_depth_m * 1000.0)

        # Generate spiral
        self._generate_spiral()

        print(f"[Spiral3D] Target depth: {self.target_depth_m}m, Disparity: {self.disparity_px:.1f}px")

    def _generate_spiral(self):
        """Generate Archimedean spiral points."""
        theta_max = 2 * np.pi * self.turns
        thetas = np.arange(0, theta_max + self.theta_step, self.theta_step, dtype=np.float32)
        r = self.a + self.b * thetas
        xs = r * np.cos(thetas)
        ys = r * np.sin(thetas)

        self.spiral_points = np.stack([xs, ys], axis=1)
        ds = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        self.s = np.concatenate([[0.0], np.cumsum(ds)])
        self.thetas = thetas

    def depth_to_disparity(self, depth_mm: float) -> float:
        """Convert depth in mm to stereo disparity in pixels."""
        if depth_mm <= 0 or math.isnan(depth_mm):
            return self.disparity_px  # Use default
        return (self.IPD_MM * self.focal_length_px) / depth_mm

    def draw_stereo(self, color_bgr: Tuple[int, int, int] = (0, 255, 255), thickness: int = 4) -> np.ndarray:
        """
        Draw side-by-side stereo spiral.

        Returns:
            3840x1080 BGR image with left and right eye views
        """
        sbs = np.zeros((self.EYE_HEIGHT, self.FULL_WIDTH, 3), dtype=np.uint8)

        # Draw left and right eyes
        for eye, shift_sign in [("L", -1), ("R", 1)]:
            shift_x = shift_sign * self.disparity_px / 2.0
            center_x = self.EYE_WIDTH / 2.0 + shift_x
            center_y = self.EYE_HEIGHT / 2.0

            # Offset for right eye in SBS frame
            x_offset = 0 if eye == "L" else self.EYE_WIDTH

            # Transform points
            points = self.spiral_points.copy()
            points[:, 0] += center_x + x_offset
            points[:, 1] += center_y

            # Draw
            points_int = points.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(sbs, [points_int], False, color_bgr, thickness, cv2.LINE_AA)

        return sbs

    def draw_point_3d(self, sbs_frame: np.ndarray, x: float, y: float, depth_mm: float,
                      color: Tuple[int, int, int] = (255, 0, 255), radius: int = 5):
        """
        Draw a point with stereo disparity based on depth.

        Args:
            sbs_frame: 3840x1080 side-by-side frame to draw on
            x, y: Position in display coordinates (1920x1080 space)
            depth_mm: Actual depth in millimeters
            color: Point color (BGR)
            radius: Point radius
        """
        disparity = self.depth_to_disparity(depth_mm)

        # Left eye: shift right
        x_left = int(x - disparity / 2.0)
        y_pos = int(y)

        if 0 <= x_left < self.EYE_WIDTH and 0 <= y_pos < self.EYE_HEIGHT:
            cv2.circle(sbs_frame, (x_left, y_pos), radius + 4, color, 2)
            cv2.circle(sbs_frame, (x_left, y_pos), radius, color, -1)

        # Right eye: shift left
        x_right = int(x + disparity / 2.0 + self.EYE_WIDTH)

        if self.EYE_WIDTH <= x_right < self.FULL_WIDTH and 0 <= y_pos < self.EYE_HEIGHT:
            cv2.circle(sbs_frame, (x_right, y_pos), radius + 4, color, 2)
            cv2.circle(sbs_frame, (x_right, y_pos), radius, color, -1)

    def nearest_point(self, x: float, y: float) -> Tuple[float, float, float, float]:
        """
        Find nearest spiral point.

        Args:
            x, y: Query point in display coordinates (1920x1080 space)

        Returns:
            (xs, ys, s, theta_tan): Nearest point coords, arc length, tangent angle
        """
        cx = self.EYE_WIDTH / 2.0
        cy = self.EYE_HEIGHT / 2.0

        x_rel = x - cx
        y_rel = y - cy

        dx = self.spiral_points[:, 0] - x_rel
        dy = self.spiral_points[:, 1] - y_rel
        i = int(np.argmin(dx*dx + dy*dy))

        xs = float(self.spiral_points[i, 0]) + cx
        ys = float(self.spiral_points[i, 1]) + cy
        s = float(self.s[i])

        if i < len(self.spiral_points) - 1:
            tx = self.spiral_points[i+1, 0] - self.spiral_points[i, 0]
            ty = self.spiral_points[i+1, 1] - self.spiral_points[i, 1]
        else:
            tx = self.spiral_points[i, 0] - self.spiral_points[i-1, 0]
            ty = self.spiral_points[i, 1] - self.spiral_points[i-1, 1]

        theta_tan = float(np.arctan2(ty, tx))

        return xs, ys, s, theta_tan

    def endpoints(self) -> Dict[str, Tuple[int, int]]:
        """Get spiral start and end points in display coordinates."""
        cx = self.EYE_WIDTH / 2.0
        cy = self.EYE_HEIGHT / 2.0

        start = (int(self.spiral_points[0, 0] + cx), int(self.spiral_points[0, 1] + cy))
        end = (int(self.spiral_points[-1, 0] + cx), int(self.spiral_points[-1, 1] + cy))

        return {"start": start, "end": end}
