# 2D Finger-Tracing (MediaPipe vs HSV) — One-Each Refactor

Runs **exactly 1 MediaPipe trial** and **1 HSV trial**, then saves:
- `runs/paths_overlay.png` — both paths over the spiral
- `runs/signal_summary.json` — RMS (px) and resampling meta per method
- `runs/psd_mp_spatial.csv`, `runs/psd_hsv_spatial.csv` — Welch PSD of error vs arc-length (cycles/pixel)

## Quick Start (Windows-friendly)
```powershell
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
py -m src.main --config config.yaml
```

Make sure **RealSense D435** is on a **USB 3.x** port. Close other camera apps.

## Notes
- Only **index fingertip (landmark 8)** for MediaPipe.
- Hands-free start/stop via dwell at spiral endpoints.
- Depth sampled as **median in NxN window** (zeros ignored).
- **Jump gate** rejects sudden landmark teleports (`experiment.max_jump_px`).
- Live preview window can be disabled via `show_live_preview: false`.
