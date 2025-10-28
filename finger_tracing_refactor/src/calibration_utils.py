"""
Calibration utilities for affine transform correction.

Provides functions to compute, save, load, and apply affine transformations
that correct for camera misalignment, FOV mismatch, and distortions.
"""
import numpy as np
import cv2
import json
import os
from typing import Tuple, List, Optional


def compute_calibration(src_points: List[Tuple[float, float]],
                        dst_points: List[Tuple[float, float]],
                        use_ransac: bool = True) -> Optional[np.ndarray]:
    """
    Compute affine transformation matrix from source to destination points.

    Uses OpenCV's estimateAffinePartial2D which allows rotation, uniform scaling,
    and translation (but not shear). For full affine with shear, use estimateAffine2D.

    Args:
        src_points: List of (x, y) tuples - detected finger positions in camera coords
        dst_points: List of (x, y) tuples - corresponding ground truth dot positions
        use_ransac: Whether to use RANSAC to reject outliers (recommended)

    Returns:
        2x3 affine transformation matrix, or None if computation fails

    Example:
        src = [(100, 100), (200, 150), ...]  # Camera detected positions
        dst = [(110, 105), (215, 160), ...]  # Actual dot positions
        matrix = compute_calibration(src, dst)
    """
    if len(src_points) < 3 or len(dst_points) < 3:
        print(f"[Calibration] Error: Need at least 3 points, got {len(src_points)}")
        return None

    if len(src_points) != len(dst_points):
        print(f"[Calibration] Error: Source and destination point counts don't match")
        return None

    src_array = np.array(src_points, dtype=np.float32).reshape(-1, 1, 2)
    dst_array = np.array(dst_points, dtype=np.float32).reshape(-1, 1, 2)

    if use_ransac:
        # Use RANSAC to reject outliers (tracking errors, occlusions, etc.)
        matrix, inliers = cv2.estimateAffinePartial2D(
            src_array, dst_array,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,  # Max pixel error for inlier
            maxIters=2000,
            confidence=0.99
        )

        if matrix is None:
            print("[Calibration] Error: RANSAC failed to find affine transform")
            return None

        inlier_count = np.sum(inliers) if inliers is not None else 0
        inlier_ratio = inlier_count / len(src_points)
        print(f"[Calibration] RANSAC: {inlier_count}/{len(src_points)} inliers ({inlier_ratio*100:.1f}%)")

        if inlier_ratio < 0.5:
            print("[Calibration] Warning: Low inlier ratio, calibration may be poor")
    else:
        # Use all points (least squares fit)
        matrix, _ = cv2.estimateAffinePartial2D(src_array, dst_array, method=cv2.LMEDS)

        if matrix is None:
            print("[Calibration] Error: Failed to compute affine transform")
            return None

    print(f"[Calibration] Computed affine matrix:\n{matrix}")
    return matrix


def save_calibration(matrix: np.ndarray, filepath: str):
    """
    Save calibration matrix to JSON file.

    Args:
        matrix: 2x3 numpy array containing affine transformation
        filepath: Path to save calibration file (e.g., 'calibration.json')
    """
    if matrix is None:
        raise ValueError("Cannot save None matrix")

    if matrix.shape != (2, 3):
        raise ValueError(f"Matrix must be 2x3, got {matrix.shape}")

    calibration_data = {
        "matrix": matrix.tolist(),
        "version": "1.0",
        "transform_type": "affine_partial_2d"
    }

    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(calibration_data, f, indent=2)

    print(f"[Calibration] Saved to: {filepath}")


def load_calibration(filepath: str) -> Optional[np.ndarray]:
    """
    Load calibration matrix from JSON file.

    Args:
        filepath: Path to calibration file

    Returns:
        2x3 numpy array, or None if file doesn't exist or is invalid
    """
    if not os.path.exists(filepath):
        print(f"[Calibration] File not found: {filepath}")
        return None

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        matrix = np.array(data["matrix"], dtype=np.float64)

        if matrix.shape != (2, 3):
            print(f"[Calibration] Error: Invalid matrix shape {matrix.shape}, expected (2, 3)")
            return None

        print(f"[Calibration] Loaded from: {filepath}")
        print(f"[Calibration] Matrix:\n{matrix}")
        return matrix

    except Exception as e:
        print(f"[Calibration] Error loading file: {e}")
        return None


def apply_calibration(matrix: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    """
    Apply affine transformation to a single point.

    Args:
        matrix: 2x3 affine transformation matrix
        x, y: Input point coordinates

    Returns:
        (x_transformed, y_transformed): Calibrated coordinates

    Example:
        matrix = load_calibration('calibration.json')
        x_cal, y_cal = apply_calibration(matrix, x_raw, y_raw)
    """
    if matrix is None:
        return x, y

    # Affine transform: [x'] = [a00  a01  a02] * [x]
    #                   [y']   [a10  a11  a12]   [y]
    #                                            [1]
    point = np.array([x, y, 1.0], dtype=np.float64)
    transformed = matrix @ point

    return float(transformed[0]), float(transformed[1])


def apply_calibration_batch(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply affine transformation to multiple points efficiently.

    Args:
        matrix: 2x3 affine transformation matrix
        points: Nx2 numpy array of (x, y) coordinates

    Returns:
        Nx2 numpy array of transformed coordinates
    """
    if matrix is None:
        return points

    # Add homogeneous coordinate
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    points_homogeneous = np.hstack([points, ones])

    # Apply transform
    transformed = (matrix @ points_homogeneous.T).T

    return transformed


def validate_calibration(matrix: np.ndarray,
                        test_src: List[Tuple[float, float]],
                        test_dst: List[Tuple[float, float]]) -> dict:
    """
    Validate calibration quality using test points.

    Args:
        matrix: Calibration matrix to validate
        test_src: Test source points (camera coordinates)
        test_dst: Test destination points (ground truth)

    Returns:
        Dictionary with validation metrics (mean_error, median_error, max_error)
    """
    if len(test_src) == 0 or len(test_dst) == 0:
        return {"error": "No test points provided"}

    errors = []
    for (sx, sy), (dx, dy) in zip(test_src, test_dst):
        cx, cy = apply_calibration(matrix, sx, sy)
        error = np.sqrt((cx - dx)**2 + (cy - dy)**2)
        errors.append(error)

    errors = np.array(errors)

    return {
        "mean_error_px": float(np.mean(errors)),
        "median_error_px": float(np.median(errors)),
        "std_error_px": float(np.std(errors)),
        "max_error_px": float(np.max(errors)),
        "min_error_px": float(np.min(errors)),
        "num_test_points": len(errors)
    }
