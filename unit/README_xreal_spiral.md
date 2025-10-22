# XReal Spiral Tracing with Hand Tracking

Real-time hand tracking application optimized for XReal 1 Pro AR glasses. Displays a bright spiral on black background with a tracking dot that follows your index fingertip.

## Quick Start

### 1. Navigate to project directory
```bash
cd "c:\Users\14845\MyStuff\ClassFolders\Senior Design\real_sense"
```

### 2. Install dependencies (if not already installed)
```bash
py -m pip install -r finger_tracing_refactor/requirements.txt
```

### 3. Run the application
```bash
py unit/xreal_spiral_tracing.py
```

## Features

- **Real-time hand tracking** using MediaPipe
- **Intel RealSense camera** for robust finger detection
- **Bright spiral visualization** on black background (transparent on XReal glasses)
- **Tracking dot** that follows your index fingertip
- **Fullscreen display** optimized for XReal 1 Pro (1920x1080)
- **Customizable colors and sizes** via keyboard controls

## Controls

| Key | Action |
|-----|--------|
| `q` or `ESC` | Quit application |
| `r` | Reset |
| `+` / `-` | Increase/decrease spiral thickness |
| `[` / `]` | Decrease/increase dot size |
| `c` | Cycle through color presets (Cyan → Green → Yellow → Orange → Magenta → White) |
| `d` | Toggle dot visibility |
| `s` | Toggle spiral visibility |

## Configuration

Edit `config.yaml` to customize:

- **Camera settings**: Resolution, FPS, exposure
- **Spiral parameters**: Size (a, b), turns, thickness, color
- **Hand tracking**: Model complexity, confidence thresholds
- **Depth filtering**: Min/max distance for finger detection

## How to Use with XReal 1 Pro Glasses

1. **Connect XReal glasses** to your laptop
2. **Set XReal to screen mirroring mode** (not 3D SBS mode for this app)
3. **Run the script** - it will open in fullscreen
4. **Hold your hand** at arm's length in front of the RealSense camera
5. **Extend your index finger** and move it around
6. **See the bright dot** following your fingertip through the glasses
7. **Trace the spiral** with your finger, focusing on the dot

## Troubleshooting

### "No RealSense device found"
- Make sure your Intel RealSense camera is connected
- Try unplugging and reconnecting the camera
- Check Device Manager to ensure drivers are installed

### Finger not detected
- Ensure good lighting conditions
- Keep your hand within 200-800mm from the camera (arm's length)
- Make sure your index finger is clearly visible
- Try adjusting camera exposure in `config.yaml`

### Low FPS / Laggy
- Close other applications using the camera
- Reduce camera resolution in `config.yaml` (try 640x480)
- Lower MediaPipe model_complexity to 0 in `config.yaml`

### Dot position seems off
- The dot position is scaled from camera resolution to display resolution
- Make sure camera width/height in config matches actual camera output
- Check that RealSense is in correct mode (640x480 or 1280x720)

## Technical Details

- **Input**: Intel RealSense D400 series camera (640x480@30fps)
- **Tracking**: MediaPipe Hands (landmark 8 = index fingertip)
- **Output**: 1920x1080 fullscreen display
- **Smoothing**: EMA + temporal median filter for stable tracking
- **Latency**: ~33ms (30fps) end-to-end

## Files

- `xreal_spiral_tracing.py` - Main application
- `config.yaml` - Configuration file (in project root)
- Dependencies:
  - `finger_tracing_refactor/src/io_rs.py` - RealSense camera interface
  - `finger_tracing_refactor/src/track_mp.py` - MediaPipe hand tracker
  - `finger_tracing_refactor/src/spiral.py` - Spiral curve generator

## Future Enhancements

- [ ] 3D SBS mode support for true depth perception
- [ ] Trail effect showing finger path history
- [ ] Multiple spiral patterns (square, circle, figure-8)
- [ ] Audio feedback when on/off path
- [ ] Performance metrics and tremor analysis
- [ ] Recording and playback of tracing sessions
