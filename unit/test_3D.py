# XReal Stereoscopic Spiral Display
# Displays an Archimedean spiral at 0.5m perceived depth using proper stereo disparity
# Requires: pip install opencv-python numpy
import cv2 as cv
import numpy as np
import time
import math

# --- Display setup ---
FULL_W, FULL_H = 3840, 1080      # Full-SBS canvas
EYE_W, EYE_H   = FULL_W // 2, FULL_H

# --- XReal glasses stereoscopic parameters ---
# Based on XReal Air/Air Pro specs
IPD_MM = 63.0                     # Inter-pupillary distance (average adult: 63mm)
SCREEN_WIDTH_MM = 201.0           # XReal effective screen width at 4m (approximately)
FOV_HORIZONTAL_DEG = 46.0         # XReal horizontal field of view
TARGET_DEPTH_M = 0.5              # Target perceived depth (matching Unity's requiredZDistance)

# Calculate pixels per degree
PIXELS_PER_DEGREE = EYE_W / FOV_HORIZONTAL_DEG

# Calculate disparity for target depth
# Formula: disparity_px = (IPD_mm * focal_length_px) / depth_mm
# Where focal_length_px ≈ screen_width_px / (2 * tan(FOV/2))
FOCAL_LENGTH_PX = EYE_W / (2.0 * math.tan(math.radians(FOV_HORIZONTAL_DEG / 2.0)))
DISPARITY_PX = (IPD_MM * FOCAL_LENGTH_PX) / (TARGET_DEPTH_M * 1000.0)

print("Stereoscopic parameters:")
print(f"  Target depth: {TARGET_DEPTH_M}m")
print(f"  Focal length: {FOCAL_LENGTH_PX:.1f}px")
print(f"  Disparity: {DISPARITY_PX:.1f}px")
print(f"  Pixels per degree: {PIXELS_PER_DEGREE:.1f}")

# --- Unity spiral parameters ---
# Matching Unity's XRSpiralOverlay settings
SPIRAL_SIZE_M = 0.15              # spiralSize in Unity (increased from 9cm to 15cm diameter)
NUMBER_OF_COILS = 3               # numberOfCoils in Unity
SPIRAL_POINTS = 1000              # spiralPoints in Unity
SPIRAL_COLOR_BGR = (0, 255, 255) # Yellow (Unity's spiralColor)
SPIRAL_THICKNESS = 4              # Line thickness (increased from 3 to 4)

# Calculate spiral size in pixels at target depth
# Angular size = 2 * arctan(physical_size / (2 * distance))
ANGULAR_SIZE_RAD = 2.0 * math.atan(SPIRAL_SIZE_M / (2.0 * TARGET_DEPTH_M))
ANGULAR_SIZE_DEG = math.degrees(ANGULAR_SIZE_RAD)
SPIRAL_RADIUS_PX = ANGULAR_SIZE_DEG * PIXELS_PER_DEGREE / 2.0

print("\nSpiral parameters:")
print(f"  Physical size: {SPIRAL_SIZE_M}m ({SPIRAL_SIZE_M*100}cm)")
print(f"  Angular size: {ANGULAR_SIZE_DEG:.2f}°")
print(f"  Screen radius: {SPIRAL_RADIUS_PX:.1f}px")

# If your XREAL is the second monitor to the RIGHT of your main display,
# set this to your main display width in pixels so the window opens there.
# Otherwise, leave at 0 and just drag the window to the XREAL screen.
SECOND_MONITOR_X = 1920  # adjust or set to 0


def generate_archimedean_spiral(num_points=SPIRAL_POINTS, num_coils=NUMBER_OF_COILS, max_radius=SPIRAL_RADIUS_PX):
    """
    Generate Archimedean spiral points.
    Matches Unity's spiral generation: r = b * theta

    Returns:
        numpy array of (x, y) points relative to center (0, 0)
    """
    theta_max = num_coils * 2.0 * np.pi
    thetas = np.linspace(0, theta_max, num_points)

    # Archimedean spiral: r = b * theta
    # We want max radius at theta_max
    b = max_radius / theta_max

    radii = b * thetas
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
    global DISPARITY_PX, SPIRAL_RADIUS_PX

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
            # Increase spiral size
            SPIRAL_RADIUS_PX += 10
            spiral_points = generate_archimedean_spiral()
            print(f"Spiral radius increased: {SPIRAL_RADIUS_PX:.1f}px")
        elif key == ord('['):
            # Decrease spiral size
            SPIRAL_RADIUS_PX = max(50, SPIRAL_RADIUS_PX - 10)
            spiral_points = generate_archimedean_spiral()
            print(f"Spiral radius decreased: {SPIRAL_RADIUS_PX:.1f}px")

    cv.destroyAllWindows()
    print("\nExiting...")


if __name__ == "__main__":
    main()
