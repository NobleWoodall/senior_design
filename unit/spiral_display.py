"""
Spiral Display for XReal 1 Pro Glasses
Displays a spiral on a black background for XReal 1 Pro screen mirroring.
Black areas will appear transparent on the glasses.
"""

import cv2
import numpy as np
import math

def create_spiral_image(width=1920, height=1080, num_turns=5, thickness=3, color=(255, 255, 255)):
    """
    Create an image with a spiral drawn on black background.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        num_turns: Number of spiral turns
        thickness: Line thickness for the spiral
        color: Color of the spiral (BGR format)

    Returns:
        Image with spiral on black background
    """
    # Create black background
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Center of the image
    center_x = width // 2
    center_y = height // 2

    # Maximum radius (to fit in the image)
    max_radius = min(center_x, center_y) * 0.9

    # Generate spiral points
    points = []
    num_points = 1000

    for i in range(num_points):
        # Parameter t goes from 0 to num_turns * 2π
        t = (i / num_points) * num_turns * 2 * math.pi

        # Radius increases linearly with angle
        r = (t / (num_turns * 2 * math.pi)) * max_radius

        # Convert polar to cartesian coordinates
        x = int(center_x + r * math.cos(t))
        y = int(center_y + r * math.sin(t))

        points.append((x, y))

    # Draw the spiral
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], color, thickness)

    return img

def main():
    """
    Main function to display the spiral for XReal 1 Pro glasses.
    Press 'q' to quit, 'r' to regenerate with random color.
    """
    # XReal 1 Pro typical display resolution when screen mirroring
    width = 1920
    height = 1080

    # Create window
    window_name = "XReal 1 Pro Spiral Display"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Initial spiral with white color
    spiral_img = create_spiral_image(width, height, num_turns=5, thickness=3, color=(255, 255, 255))

    print("Spiral Display for XReal 1 Pro")
    print("Controls:")
    print("  'q' - Quit")
    print("  'r' - Random color spiral")
    print("  '+' - Increase thickness")
    print("  '-' - Decrease thickness")
    print("  'w' - White spiral (default)")
    print("  'ESC' - Exit fullscreen/Quit")

    thickness = 3
    num_turns = 5
    color = (255, 255, 255)  # White

    while True:
        cv2.imshow(window_name, spiral_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('r'):  # Random color
            color = (
                np.random.randint(100, 256),
                np.random.randint(100, 256),
                np.random.randint(100, 256)
            )
            spiral_img = create_spiral_image(width, height, num_turns, thickness, color)
            print(f"New color: BGR{color}")
        elif key == ord('w'):  # White
            color = (255, 255, 255)
            spiral_img = create_spiral_image(width, height, num_turns, thickness, color)
            print("Color: White")
        elif key == ord('+') or key == ord('='):  # Increase thickness
            thickness = min(thickness + 1, 20)
            spiral_img = create_spiral_image(width, height, num_turns, thickness, color)
            print(f"Thickness: {thickness}")
        elif key == ord('-') or key == ord('_'):  # Decrease thickness
            thickness = max(thickness - 1, 1)
            spiral_img = create_spiral_image(width, height, num_turns, thickness, color)
            print(f"Thickness: {thickness}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
