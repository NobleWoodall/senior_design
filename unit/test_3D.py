# XReal Stereoscopic Spiral Display
# Displays an Archimedean spiral at 0.5m perceived depth using proper stereo disparity
# Requires: pip install opencv-python numpy
import cv2 as cv
import numpy as np
import time
import math
import yaml
from pathlib import Path

# --- Load config ---
config_path = Path(__file__).parent.parent / "config.yaml"
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from {config_path}")
else:
    config = {}
    print("Warning: config.yaml not found, using defaults")

# --- Display setup ---
FULL_W, FULL_H = 3840, 1080      # Full-SBS canvas
EYE_W, EYE_H   = FULL_W // 2, FULL_H

# --- XReal glasses stereoscopic parameters ---
# Based on XReal Air/Air Pro specs
IPD_MM = 63.0                     # Inter-pupillary distance (average adult: 63mm)
FOV_HORIZONTAL_DEG = 46.0         # XReal horizontal field of view

# Load from config
stereo_config = config.get('stereo_3d', {})
TARGET_DEPTH_M = stereo_config.get('target_depth_m', 0.5)
DISPARITY_OFFSET_PX = stereo_config.get('disparity_offset_px', 0.0)

# Calculate disparity for target depth
# Formula: disparity_px = (IPD_mm * focal_length_px) / depth_mm
# Where focal_length_px ≈ screen_width_px / (2 * tan(FOV/2))
FOCAL_LENGTH_PX = EYE_W / (2.0 * math.tan(math.radians(FOV_HORIZONTAL_DEG / 2.0)))
BASE_DISPARITY_PX = (IPD_MM * FOCAL_LENGTH_PX) / (TARGET_DEPTH_M * 1000.0)
DISPARITY_PX = BASE_DISPARITY_PX + DISPARITY_OFFSET_PX

print("Stereoscopic parameters:")
print(f"  Target depth: {TARGET_DEPTH_M}m")
print(f"  Focal length: {FOCAL_LENGTH_PX:.1f}px")
print(f"  Base disparity: {BASE_DISPARITY_PX:.1f}px")
print(f"  Disparity offset: {DISPARITY_OFFSET_PX:.1f}px")
print(f"  Final disparity: {DISPARITY_PX:.1f}px")

# --- Spiral parameters from config ---
spiral_config = config.get('spiral', {})
SPIRAL_A = spiral_config.get('a', 30.0)
SPIRAL_B = spiral_config.get('b', 35.0)
SPIRAL_TURNS = spiral_config.get('turns', 2.0)
SPIRAL_THETA_STEP = spiral_config.get('theta_step', 0.01)
SPIRAL_THICKNESS = spiral_config.get('line_thickness', 30)
SPIRAL_COLOR_BGR = tuple(spiral_config.get('color_bgr', [40, 220, 40]))

print("\nSpiral parameters:")
print(f"  a: {SPIRAL_A}")
print(f"  b: {SPIRAL_B}")
print(f"  turns: {SPIRAL_TURNS}")
print(f"  thickness: {SPIRAL_THICKNESS}px")

# If your XREAL is the second monitor to the RIGHT of your main display,
# set this to your main display width in pixels so the window opens there.
# Otherwise, leave at 0 and just drag the window to the XREAL screen.
SECOND_MONITOR_X = 1920  # adjust or set to 0


def generate_archimedean_spiral():
    """
    Generate Archimedean spiral points using config parameters.
    Uses formula: r = a + b*theta (matching the actual system)

    Returns:
        numpy array of (x, y) points relative to center (0, 0)
    """
    theta_max = 2.0 * np.pi * SPIRAL_TURNS
    thetas = np.arange(0, theta_max + SPIRAL_THETA_STEP, SPIRAL_THETA_STEP, dtype=np.float32)

    # Archimedean spiral: r = a + b * theta
    radii = SPIRAL_A + SPIRAL_B * thetas
    xs = radii * np.cos(thetas)
    ys = radii * np.sin(thetas)

    return np.stack([xs, ys], axis=1)


def draw_spiral(img, spiral_points, center_x, center_y, color, thickness):
    """
    Draw the spiral on the image.

    Args:
        img: Image to draw on
        spiral_points: Nx2 array of (x, y) points relative to origin
        center_x, center_y: Center position in image coordinates
        color: BGR color tuple
        thickness: Line thickness
    """
    # Translate spiral points to image coordinates
    points = spiral_points.copy()
    points[:, 0] += center_x
    points[:, 1] += center_y

    # Convert to integer coordinates for OpenCV
    points_int = points.astype(np.int32).reshape(-1, 1, 2)

    # Draw the spiral
    cv.polylines(img, [points_int], False, color, thickness, cv.LINE_AA)


def make_eye_frame(spiral_points, eye="L", disparity_mult=1.0):
    """
    Create a 1920x1080 frame for one eye with the spiral.

    Args:
        spiral_points: Precomputed spiral points
        eye: "L" for left, "R" for right
        disparity_mult: Multiplier to reverse disparity direction if needed

    Returns:
        1920x1080 BGR image
    """
    img = np.zeros((EYE_H, EYE_W, 3), np.uint8)

    # Calculate center position with stereo disparity
    # Left eye: shift spiral to the RIGHT (negative disparity)
    # Right eye: shift spiral to the LEFT (positive disparity)
    # This creates convergence at the target depth
    effective_disparity = DISPARITY_PX * disparity_mult
    if eye == "L":
        shift = -effective_disparity / 2.0
    else:
        shift = effective_disparity / 2.0

    center_x = EYE_W / 2.0 + shift
    center_y = EYE_H / 2.0

    # Draw the spiral
    draw_spiral(img, spiral_points, center_x, center_y, SPIRAL_COLOR_BGR, SPIRAL_THICKNESS)

    # Draw small center dot at convergence point (for debugging)
    dot_x = int(center_x)
    dot_y = int(center_y)
    cv.circle(img, (dot_x, dot_y), 3, (0, 255, 0), -1)

    # Add eye label and info (top-left)
    cv.putText(img, f"{eye}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2, cv.LINE_AA)
    cv.putText(img, f"Depth: {TARGET_DEPTH_M}m", (20, 80),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv.LINE_AA)
    cv.putText(img, f"Disparity: {DISPARITY_PX:.0f}px", (20, 110),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv.LINE_AA)

    # Add large eye label in center for SBS verification (bright and obvious)
    eye_label = f"{eye} EYE"
    if eye == "L":
        label_color = (0, 100, 255)  # Orange for left
    else:
        label_color = (255, 100, 0)  # Blue for right
    cv.putText(img, eye_label, (EYE_W//2 - 200, EYE_H - 50),
               cv.FONT_HERSHEY_SIMPLEX, 2.0, label_color, 4, cv.LINE_AA)

    return img

# --- Main loop ---
def main():
    """Main application loop."""
    print("\n" + "="*70)
    print("  XReal Stereoscopic Spiral Display - Test")
    print("="*70)
    print("\nThis will display a 3-coil Archimedean spiral at 0.5m perceived depth.")
    print("\nIMPORTANT: This creates a 3840x1080 window (too wide for 1920px monitors!)")
    print("You MUST move the window to your XReal display (extended monitor).")
    print("\nCONTROLS:")
    print("  'q' or ESC      - Quit")
    print("  'f'             - Toggle fullscreen mode")
    print("  '+' / '-'       - Adjust disparity ±10px (change perceived depth)")
    print("  'r'             - Reverse disparity direction (if backwards)")
    print("  '[' / ']'       - Adjust spiral size")
    print("\nTIP: Keep pressing '+' to increase disparity until spiral feels close!")
    print("\nStarting in windowed mode. DRAG WINDOW TO XREAL, then press 'f'.")
    print("="*70 + "\n")

    # Allow runtime adjustment of parameters
    global DISPARITY_PX, SPIRAL_B

    # Track whether to reverse disparity (in case our calculation is backwards)
    disparity_multiplier = 1.0

    # Generate spiral points once (static spiral)
    spiral_points = generate_archimedean_spiral()

    # Create window in normal (windowed) mode first
    print("Creating window...")
    window_name = "XREAL_SBS"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    # Set initial window size to 3840x1080
    cv.resizeWindow(window_name, FULL_W, FULL_H)

    # Generate and show first frame to make window visible
    print("Rendering initial frame...")
    left  = make_eye_frame(spiral_points, "L")
    right = make_eye_frame(spiral_points, "R")
    sbs = np.hstack((left, right))
    cv.imshow(window_name, sbs)

    # Wait a moment for window to be created and displayed
    cv.waitKey(50)

    # Try to move window to second monitor (if configured)
    if SECOND_MONITOR_X > 0:
        try:
            print(f"Attempting to move window to second monitor at x={SECOND_MONITOR_X}...")
            cv.moveWindow(window_name, SECOND_MONITOR_X, 0)
            cv.waitKey(100)  # Give Windows time to move the window
            print("Window moved to second monitor successfully!")
            print("If you don't see it on your XReal, manually drag the window.")
        except Exception as e:
            print(f"Could not move window automatically: {e}")
            print("\nMANUAL SETUP REQUIRED:")
            print("  1. You should see a wide window on your main monitor (showing only half)")
            print("  2. Click and DRAG the window title bar to your XReal display")
            print("  3. Once on XReal, press 'f' to enter fullscreen mode")
    else:
        print("SECOND_MONITOR_X is set to 0 - automatic positioning disabled")
        print("\nMANUAL SETUP REQUIRED:")
        print("  1. The window will appear on your main monitor")
        print("  2. DRAG the window to your XReal display")
        print("  3. Press 'f' to enter fullscreen on XReal")

    # Don't auto-fullscreen on startup - let user position window first
    is_fullscreen = False
    print("\nWindow created in WINDOWED mode.")
    print("Once the window is on your XReal display, press 'f' for fullscreen.\n")

    fps_timer, frames = time.time(), 0
    while True:
        # Generate left and right eye frames
        left  = make_eye_frame(spiral_points, "L", disparity_multiplier)
        right = make_eye_frame(spiral_points, "R", disparity_multiplier)

        # Combine into side-by-side format
        sbs = np.hstack((left, right))  # 3840x1080
        cv.imshow(window_name, sbs)

        # FPS counter
        frames += 1
        if time.time() - fps_timer > 1.0:
            print(f"FPS: {frames}")
            frames, fps_timer = 0, time.time()

        # Handle keyboard input
        key = cv.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q to quit
            break
        elif key == ord('f'):  # Toggle fullscreen
            is_fullscreen = not is_fullscreen
            try:
                if is_fullscreen:
                    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
                    print("Fullscreen mode: ON")
                else:
                    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
                    print("Fullscreen mode: OFF (windowed)")
            except Exception as e:
                print(f"Could not toggle fullscreen: {e}")
        elif key == ord('+') or key == ord('='):
            # Increase disparity (objects appear closer) - small step
            DISPARITY_PX += 10
            if DISPARITY_PX > 0:
                approx_depth_m = (IPD_MM * FOCAL_LENGTH_PX) / (DISPARITY_PX * 1000.0)
                print(f"Disparity: {DISPARITY_PX:.1f}px → ~{approx_depth_m:.2f}m (CLOSER)")
            else:
                print(f"Disparity: {DISPARITY_PX:.1f}px")
        elif key == ord('_'):
            # Shift + '-' = fast decrease (farther)
            DISPARITY_PX -= 50
            if DISPARITY_PX > 0:
                approx_depth_m = (IPD_MM * FOCAL_LENGTH_PX) / (DISPARITY_PX * 1000.0)
                print(f"Disparity: {DISPARITY_PX:.1f}px → ~{approx_depth_m:.2f}m (FARTHER - fast)")
            elif DISPARITY_PX < 0:
                print(f"Disparity: {DISPARITY_PX:.1f}px (NEGATIVE - divergent!)")
            else:
                print(f"Disparity: {DISPARITY_PX:.1f}px (screen plane)")
        elif key == ord('-'):
            # Regular '-' = small decrease (farther)
            DISPARITY_PX -= 10
            if DISPARITY_PX > 0:
                approx_depth_m = (IPD_MM * FOCAL_LENGTH_PX) / (DISPARITY_PX * 1000.0)
                print(f"Disparity: {DISPARITY_PX:.1f}px → ~{approx_depth_m:.2f}m (farther)")
            elif DISPARITY_PX < 0:
                print(f"Disparity: {DISPARITY_PX:.1f}px (NEGATIVE - divergent!)")
            else:
                print(f"Disparity: {DISPARITY_PX:.1f}px (screen plane)")
        elif key == ord('r'):
            # Reverse disparity direction
            disparity_multiplier *= -1.0
            if disparity_multiplier > 0:
                print("Disparity direction: NORMAL (L=right shift, R=left shift)")
            else:
                print("Disparity direction: REVERSED (L=left shift, R=right shift)")
        elif key == ord(']'):
            # Increase spiral size (increase b parameter)
            SPIRAL_B += 5
            spiral_points = generate_archimedean_spiral()
            print(f"Spiral B parameter increased: {SPIRAL_B:.1f}")
        elif key == ord('['):
            # Decrease spiral size (decrease b parameter)
            SPIRAL_B = max(10, SPIRAL_B - 5)
            spiral_points = generate_archimedean_spiral()
            print(f"Spiral B parameter decreased: {SPIRAL_B:.1f}")

    cv.destroyAllWindows()
    print("\nExiting...")


if __name__ == "__main__":
    main()
