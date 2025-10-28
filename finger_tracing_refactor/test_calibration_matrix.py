"""
Quick test to verify calibration matrix is working correctly.
"""
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.calibration_utils import load_calibration, apply_calibration

def test_calibration():
    """Test if calibration file exists and matrix works."""

    calibration_file = "../runs/calibration.json"

    print("="*60)
    print("CALIBRATION MATRIX TEST")
    print("="*60)

    # Check if file exists
    if not os.path.exists(calibration_file):
        print(f"\n❌ Calibration file not found: {calibration_file}")
        print("\nPlease run calibration first:")
        print("  py -m calibrate_main --config ../config.yaml --method hsv")
        return False

    # Load calibration
    print(f"\n✓ Calibration file found: {calibration_file}")
    matrix = load_calibration(calibration_file)

    if matrix is None:
        print("\n❌ Failed to load calibration matrix")
        return False

    print(f"\n✓ Calibration matrix loaded successfully:")
    print(matrix)

    # Test transformation
    print("\n" + "="*60)
    print("TESTING TRANSFORMATION")
    print("="*60)

    # Test some sample points
    test_points = [
        (960, 540),   # Center of 1920x1080
        (0, 0),       # Top-left corner
        (1920, 1080), # Bottom-right corner
        (480, 270),   # Quarter point
    ]

    print("\nApplying calibration to test points:")
    print(f"{'Input Point':<20} {'Output Point':<20} {'Offset':<20}")
    print("-"*60)

    for x, y in test_points:
        x_cal, y_cal = apply_calibration(matrix, x, y)
        offset_x = x_cal - x
        offset_y = y_cal - y
        print(f"({x:4.0f}, {y:4.0f}){'':<10} ({x_cal:7.1f}, {y_cal:7.1f}){'':<5} ({offset_x:+7.1f}, {offset_y:+7.1f})")

    # Check if it's identity-like (no calibration effect)
    center_x, center_y = apply_calibration(matrix, 960, 540)
    offset = np.sqrt((center_x - 960)**2 + (center_y - 540)**2)

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    if offset < 1.0:
        print(f"\n⚠️  WARNING: Calibration offset at center is very small ({offset:.2f}px)")
        print("This suggests the calibration matrix is close to identity (no effect).")
        print("\nPossible reasons:")
        print("  1. Camera is already well-aligned (calibration not needed)")
        print("  2. Tracing during calibration was perfect (unlikely)")
        print("  3. Calibration data had issues (too few points, high error)")
    else:
        print(f"\n✓ Calibration has effect: {offset:.1f}px offset at center")
        print("This looks like a valid calibration matrix")

    # Check matrix determinant (should be non-zero for valid transform)
    det = np.linalg.det(matrix[:, :2])
    print(f"\n✓ Matrix determinant: {det:.6f}")
    if abs(det) < 0.01:
        print("  ⚠️  WARNING: Determinant very close to zero - matrix may be degenerate")
    elif abs(det - 1.0) < 0.01:
        print("  ✓ Matrix preserves area (determinant ≈ 1.0)")

    return True

if __name__ == "__main__":
    success = test_calibration()
    print("\n" + "="*60 + "\n")
    sys.exit(0 if success else 1)
