"""
Calibration entry point for finger/LED tracking.

Runs the calibration routine where user traces a moving dot
to compute affine transformation matrix.

Usage:
    python -m calibrate_main --config config.yaml --method hsv
    python -m calibrate_main --config config.yaml --method mp
"""
import argparse
import yaml
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import AppConfig
from src.calibrate import CalibrationRunner


def main():
    parser = argparse.ArgumentParser(description="Run calibration routine for finger/LED tracking")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--method", type=str, default="hsv", choices=["mp", "hsv"],
                       help="Tracking method: 'mp' (MediaPipe) or 'hsv' (LED)")

    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    cfg = AppConfig.from_dict(cfg_dict)

    # Run calibration
    print("\n" + "="*60)
    print("CALIBRATION MODE")
    print("="*60)
    print(f"\nConfig: {args.config}")
    print(f"Method: {args.method.upper()}")
    print(f"\nInstructions:")
    print("  1. Position your camera to view the XReal glasses display")
    print("  2. When countdown finishes, trace the moving YELLOW dot")
    print("  3. Keep your finger/LED visible throughout the calibration")
    print("  4. Press ESC to cancel at any time")
    print("  5. Press 'f' to toggle fullscreen")
    print("\n" + "="*60 + "\n")

    runner = CalibrationRunner(cfg)
    success = runner.run_calibration(method=args.method)

    if success:
        print("\n" + "="*60)
        print("SUCCESS: Calibration complete!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Enable calibration in config.yaml:")
        print("       calibration:")
        print("         enabled: true")
        print("  2. Run experiments as normal - calibration will be applied automatically")
        print("\n" + "="*60 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("FAILED: Calibration did not complete")
        print("="*60)
        print("\nTroubleshooting:")
        print("  - Ensure your finger/LED is clearly visible")
        print("  - Check camera exposure and tracking settings")
        print("  - Try slower dot speed in config.yaml")
        print("  - Ensure good lighting conditions")
        print("\n" + "="*60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
