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

    # Loop to allow iterative calibration refinement
    max_iterations = 20  # Allow many refinement iterations
    iteration = 1
    current_matrix = None  # Will hold the accumulated calibration matrix

    # Output directory for calibration file
    output_dir = cfg.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    calibration_file = os.path.join(output_dir, cfg.calibration.calibration_file)

    while iteration <= max_iterations:
        print(f"\n{'='*60}")
        if iteration == 1:
            print(f"INITIAL CALIBRATION")
        else:
            print(f"REFINEMENT ITERATION #{iteration - 1}")
        print(f"{'='*60}\n")

        # Run calibration (with current matrix if refining)
        if current_matrix is not None:
            print(f"\n[DEBUG] Passing calibration matrix to iteration {iteration}:")
            print(current_matrix)
            print()

        result = runner.run_calibration(method=args.method, initial_matrix=current_matrix)

        if result == True:
            # User pressed K (keep) - this returns the new matrix
            # We need to update run_calibration to return the matrix instead of True
            # For now, load it from the saved file
            from src.calibration_utils import load_calibration

            # Load the newly saved matrix
            new_matrix = load_calibration(calibration_file)

            if new_matrix is not None:
                current_matrix = new_matrix

                print("\n" + "="*60)
                print(f"ITERATION #{iteration} SAVED")
                print("="*60)
                print("\nOptions:")
                print("  - Press CTRL+C to finish and keep this calibration")
                print("  - Or wait 3 seconds to continue refining...")
                print("\n" + "="*60 + "\n")

                # Give user time to press CTRL+C or continue
                try:
                    import time
                    for i in range(3, 0, -1):
                        print(f"Continuing in {i}...")
                        time.sleep(1)
                    print("\nStarting next refinement iteration...\n")
                    iteration += 1
                    continue
                except KeyboardInterrupt:
                    print("\n\n" + "="*60)
                    print("CALIBRATION COMPLETE")
                    print("="*60)
                    print(f"\nFinal calibration saved to: {calibration_file}")
                    print(f"Total iterations: {iteration}")
                    print("\nNext steps:")
                    print("  1. Enable calibration in config.yaml:")
                    print("       calibration:")
                    print("         enabled: true")
                    print("  2. Run experiments as normal - calibration will be applied automatically")
                    print("\n" + "="*60 + "\n")
                    sys.exit(0)
            else:
                print("\n[ERROR] Failed to load saved calibration matrix")
                sys.exit(1)

        elif result == "redo":
            # User pressed R - redo current iteration (don't save, start over)
            print(f"\nRedoing iteration #{iteration}...")
            continue
        else:
            # User cancelled (ESC) or error occurred
            if current_matrix is not None:
                print("\n" + "="*60)
                print("CALIBRATION CANCELLED")
                print("="*60)
                print(f"\nLast successful calibration (iteration #{iteration - 1}) is still saved")
                print(f"File: {calibration_file}")
                print("\n" + "="*60 + "\n")
                sys.exit(0)
            else:
                print("\n" + "="*60)
                print("CALIBRATION CANCELLED")
                print("="*60)
                print("\nNo calibration was saved")
                print("\n" + "="*60 + "\n")
                sys.exit(1)

    # Too many iterations
    print("\n" + "="*60)
    print(f"MAXIMUM ITERATIONS REACHED ({max_iterations})")
    print("="*60)
    print(f"\nCalibration saved with {iteration - 1} refinement iterations")
    print(f"File: {calibration_file}")
    print("\n" + "="*60 + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
