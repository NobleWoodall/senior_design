# 3D Stereoscopic Finger Tracking for XReal Glasses

Main finger tracking system with integrated 3D stereoscopic visualization.

## Quick Start

```bash
# From the real_sense directory
python -m finger_tracing_refactor.src.main

# Or directly
python finger_tracing_refactor/src/main.py

# With custom config
python -m finger_tracing_refactor.src.main --config my_config.yaml
```

## What It Does

- **Tracks your finger in 3D** using RealSense camera + MediaPipe/LED tracking
- **Renders stereoscopic 3840x1080** side-by-side display for XReal glasses
- **Displays spiral at 0.5m depth** (configurable in config.yaml)
- **Shows finger with correct depth parallax** based on actual depth sensor data

## Setup

1. Mount RealSense D435/D455 camera on XReal glasses
2. Connect XReal as extended display (not mirrored)
3. Run the main script
4. Drag window to XReal display
5. Press 'f' for fullscreen
6. Start tracking!

## Controls

- **'f'** - Toggle fullscreen
- **ESC** - Exit trial

## Configuration

Edit `config.yaml` to customize:

```yaml
stereo_3d:
  target_depth_m: 0.5    # Spiral depth in meters (0.3-1.0 recommended)

spiral:
  a: 12.0                # Inner radius
  b: 8.0                 # Growth rate
  turns: 2               # Number of turns
  color_bgr: [40, 220, 40]  # Green color
  line_thickness: 5

# ... other settings
```

## How It Works

1. **Camera tracking**: 640x480 resolution
2. **Coordinate scaling**: Automatic scaling to 1920x1080 per eye
3. **Stereo rendering**: Side-by-side 3840x1080 output
4. **Depth mapping**: Finger dot appears at actual depth in 3D

### Technical Details

- **Display**: 3840x1080 (1920x1080 per eye)
- **Target depth**: 0.5m (configurable)
- **Disparity**: ~287px at 0.5m
- **Scaling**: 3.0x horizontal, 2.25x vertical
- **FPS**: ~30 Hz (camera-limited)

## Data Output

Results saved to `runs/` directory:

```
runs/
├── <timestamp>_method-mp_spiral-stereo_3d/
│   ├── frames.csv       # Position, depth, error data
│   ├── summary.json     # Metrics (RMSE, median error, etc.)
│   └── config.yaml      # Config snapshot
└── results.json         # Combined results + stereo params
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Window not on XReal | Manually drag window, then press 'f' |
| No depth | Check RealSense USB 3.0 connection |
| Low FPS | Close other apps, verify camera settings |
| Wrong depth | Adjust `target_depth_m` in config.yaml |

---

**Ready to run!**

```bash
python -m finger_tracing_refactor.src.main
```
