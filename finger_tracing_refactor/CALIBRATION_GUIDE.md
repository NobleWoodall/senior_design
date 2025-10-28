# Calibration System Guide

## Overview

The calibration system corrects for camera misalignment, FOV mismatch, and distortions by having you trace a moving dot that follows the spiral path perfectly. The system learns the mapping between what the camera sees vs. where things actually are.

## How It Works

1. **Moving Dot**: A yellow dot moves along the spiral path at a known position
2. **User Traces**: You follow the dot with your finger/LED (not the spiral itself!)
3. **Data Collection**: System records (camera_position, actual_dot_position) pairs
4. **Calibration**: Computes an affine transformation matrix using RANSAC
5. **Application**: Transform is automatically applied to all future tracking

## Quick Start

### Step 1: Run Calibration

```bash
cd finger_tracing_refactor
py -m calibrate_main --config ../config.yaml --method hsv
```

**Options:**
- `--method hsv` - Use LED tracking (red light)
- `--method mp` - Use MediaPipe hand tracking

### Step 2: Follow Instructions

**Preview Phase:**
1. Window opens showing the spiral with a **yellow dot** at the center (starting position)
2. Position yourself so you can reach the starting point comfortably
3. You'll see your finger/LED tracked in real-time (purple dot)
4. Press **SPACE** when ready to begin

**Countdown Phase:**
5. 3-second countdown begins
6. Get your finger/LED positioned at the starting dot

**Calibration Phase:**
7. Yellow dot starts moving around the spiral
8. **Trace the yellow dot** (not the spiral!) with your finger/LED
9. Keep your finger/LED visible at all times
10. System collects data for 2 complete traces (configurable)

**Test Phase:**
11. After calibration is computed, you enter TEST MODE
12. Trace the spiral freely to see how accurate the calibration is
13. Purple dot = your calibrated position
14. Green dot = nearest point on spiral
15. Error in pixels is shown at the top
16. **Press K to Keep** the calibration and finish
17. **Press R to Redo** the calibration if you're not satisfied
18. **Press ESC to Cancel** and exit

### Step 3: Enable Calibration (if you kept it)

The calibration is automatically saved when you press K. Now edit `config.yaml`:

```yaml
calibration:
  enabled: true  # Change from false to true
```

### Step 4: Run Experiments

Run experiments as normal - calibration is automatically applied:

```bash
py -m src.main --config ../config.yaml
```

## Configuration Options

In `config.yaml`:

```yaml
calibration:
  enabled: false             # Enable/disable calibration
  calibration_file: "calibration.json"  # Where to save/load calibration
  dot_speed_revolutions_per_sec: 0.15   # Dot speed (lower = slower, easier to trace)
  num_traces: 2              # Number of complete spiral traces
  countdown_sec: 3           # Countdown before starting
```

### Adjusting Dot Speed

If the dot moves too fast and you can't keep up:
- Decrease `dot_speed_revolutions_per_sec` (e.g., 0.10 for slower movement)
- Increase `num_traces` to collect more data (e.g., 3 or 4)

### Troubleshooting

**Problem: "Not enough points collected"**
- Ensure your finger/LED is clearly visible
- Check camera exposure settings
- Try slower dot speed
- Improve lighting conditions

**Problem: "High validation error"**
- Calibration worked but accuracy is poor
- Try again and trace more carefully
- Ensure consistent tracking throughout
- Check for reflections or interference

**Problem: "RANSAC failed to find affine transform"**
- Too many tracking errors/outliers
- Improve tracking settings (HSV thresholds, exposure, etc.)
- Test tracking first before running calibration

**Problem: Calibration enabled but not loading**
- Check that `runs/calibration.json` exists
- Ensure `experiment.output_dir` is set to "runs" in config
- Try absolute path in `calibration_file` if needed

## How Calibration is Applied

The calibration transform is applied in the processing pipeline:

```
1. Camera detects finger/LED → (x_raw, y_raw) in camera coordinates
2. Apply calibration transform → (x_calibrated, y_calibrated)
3. Scale to display coordinates → (x_display, y_display)
4. Apply FOV adjustment → final position
```

This means calibration corrects the raw camera detection BEFORE any other transformations.

## Technical Details

### Affine Transform

The calibration uses an **affine transformation** which can correct:
- Translation (shifting)
- Rotation (camera angle)
- Uniform scaling (size differences)
- Shear (skewing)

The 2x3 transformation matrix is stored in `calibration.json`:

```json
{
  "matrix": [
    [a00, a01, a02],
    [a10, a11, a12]
  ],
  "version": "1.0",
  "transform_type": "affine_partial_2d"
}
```

Transform formula:
```
x' = a00*x + a01*y + a02
y' = a10*x + a11*y + a12
```

### RANSAC

The system uses RANSAC (Random Sample Consensus) to reject outliers:
- Tracking errors
- Brief occlusions
- LED reflections
- Jumping/jittering

This makes the calibration robust even if tracking isn't perfect.

### Validation

After computing calibration, the system validates it on a held-out test set (20% of data):
- Mean error should be < 20 pixels (good)
- Mean error 20-50 pixels (acceptable)
- Mean error > 50 pixels (poor - consider re-running)

## Best Practices

1. **Test tracking first**: Run a normal experiment to ensure tracking works well before calibrating
2. **Good lighting**: Ensure consistent lighting for LED tracking
3. **Steady hand**: Trace smoothly - sudden jumps will be filtered out but reduce data quality
4. **Complete traces**: Follow the dot for the entire duration
5. **Re-calibrate if needed**: If you move the camera or change the setup, run calibration again
6. **Keep existing FOV scaling**: The `finger_scale_x` and `finger_scale_y` settings still apply AFTER calibration

## Files Created

- `finger_tracing_refactor/src/calibration_utils.py` - Calibration math functions
- `finger_tracing_refactor/src/calibrate.py` - Calibration routine
- `finger_tracing_refactor/calibrate_main.py` - Entry point script
- `runs/calibration.json` - Saved calibration matrix (after running calibration)

## Example Workflow

```bash
# 1. Test tracking
py -m src.main --config ../config.yaml

# 2. If tracking looks good, run calibration
py -m calibrate_main --config ../config.yaml --method hsv

# 3. Enable calibration in config.yaml
# (edit file: set calibration.enabled = true)

# 4. Run experiments with calibration
py -m src.main --config ../config.yaml

# 5. Check if accuracy improved!
```

## Keyboard Controls

**Preview Phase:**
- **SPACE**: Start countdown and begin calibration
- **ESC**: Cancel and exit
- **F**: Toggle fullscreen mode

**Countdown/Calibration Phase:**
- **ESC**: Cancel calibration
- **F**: Toggle fullscreen mode

**Test Phase:**
- **K**: Keep this calibration and save it
- **R**: Redo calibration from the beginning
- **ESC**: Cancel and exit without saving

## Need Help?

If calibration isn't working:
1. Check that basic tracking works first (run normal experiment)
2. Review camera exposure settings
3. Try slower dot speed
4. Ensure LED/finger is clearly visible
5. Check for reflections or bright background lights
