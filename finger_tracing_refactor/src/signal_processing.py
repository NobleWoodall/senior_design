import json
import cv2
import numpy as np
from typing import List, Tuple, Dict
from scipy.signal import welch, butter, filtfilt
from pathlib import Path

def draw_paths_overlay(width:int, height:int, spiral_pts:np.ndarray,
                       path_mp:List[Tuple[float,float]],
                       path_hsv:List[Tuple[float,float]], out_path:str,
                       dot_path:List[Tuple[float,float]]=None):
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw ground truth spiral path (static reference)
    if spiral_pts is not None and len(spiral_pts) > 1:
        pts = spiral_pts.astype(np.int32).reshape(-1,1,2)
        # Draw medium gray spiral as reference (visible but not too dominant)
        cv2.polylines(canvas, [pts], False, (160,160,160), 3, cv2.LINE_AA)
        # Add start and end markers
        start_pt = tuple(pts[0, 0])
        end_pt = tuple(pts[-1, 0])
        cv2.circle(canvas, start_pt, 8, (0, 180, 0), -1)  # Green start
        cv2.circle(canvas, start_pt, 10, (0, 180, 0), 2)
        cv2.circle(canvas, end_pt, 8, (180, 0, 0), -1)  # Dark blue end
        cv2.circle(canvas, end_pt, 10, (180, 0, 0), 2)
        # Add text labels
        cv2.putText(canvas, "START", (start_pt[0] + 15, start_pt[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "END", (end_pt[0] + 15, end_pt[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 0), 2, cv2.LINE_AA)

    # Draw the actual dot path (where the yellow dot moved)
    if dot_path and len(dot_path) >= 2:
        dot_pts = np.array(dot_path, dtype=np.int32).reshape(-1, 1, 2)
        # Draw thick yellow/orange path for the target dot
        cv2.polylines(canvas, [dot_pts], False, (0, 200, 255), 4, cv2.LINE_AA)  # Orange in BGR

    def _poly(points, color):
        if len(points) >= 2:
            p = np.array(points, dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(canvas, [p], False, color, 3, cv2.LINE_AA)
            cv2.circle(canvas, tuple(p[0,0]), 5, color, -1)
            cv2.circle(canvas, tuple(p[-1,0]), 5, color, -1)

    # Determine which methods have data and create label
    methods_drawn = []
    if path_mp and len(path_mp) >= 2:
        _poly(path_mp, (255, 0, 0))  # Blue in BGR
        methods_drawn.append("MediaPipe (blue)")
    if path_hsv and len(path_hsv) >= 2:
        _poly(path_hsv, (0, 0, 225))  # Red in BGR
        methods_drawn.append("HSV (red)")

    # Create legend with multiple lines
    y_offset = 28

    # Spiral reference (medium gray)
    cv2.putText(canvas, "Spiral Shape (gray)", (12, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,80), 2, cv2.LINE_AA)
    y_offset += 30

    # Dot path (orange)
    if dot_path and len(dot_path) >= 2:
        cv2.putText(canvas, "Target Dot Path (orange)", (12, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,150,255), 2, cv2.LINE_AA)
        y_offset += 30

    # Tracking methods
    if len(methods_drawn) > 0:
        for method in methods_drawn:
            cv2.putText(canvas, method, (12, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
            y_offset += 30
    else:
        cv2.putText(canvas, "No tracking data", (12, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas)

def _resample_vs_time(t_list, e_list, target_fs=30.0):
    """Resample error signal uniformly in time for temporal PSD analysis."""
    t = np.asarray(t_list, float)
    e = np.asarray(e_list, float)
    m = np.isfinite(t) & np.isfinite(e)
    t = t[m]
    e = e[m]
    if t.size < 8:
        return np.array([]), np.array([])

    # Sort by time
    o = np.argsort(t)
    t = t[o]
    e = e[o]

    # Remove duplicate times
    ut, idx = np.unique(t, return_inverse=True)
    e_avg = np.zeros_like(ut)
    cnt = np.zeros_like(ut)
    for i, u in enumerate(idx):
        e_avg[u] += e[i]
        cnt[u] += 1
    e_avg = e_avg / np.maximum(1, cnt)
    t = ut
    e = e_avg

    t0, t1 = float(t.min()), float(t.max())
    if not np.isfinite([t0, t1]).all() or t1 <= t0:
        return np.array([]), np.array([])

    # Create uniform time grid
    dt = 1.0 / target_fs
    n = max(8, int((t1 - t0) / dt))
    grid = np.linspace(t0, t1, n)

    return grid, np.interp(grid, t, e)

def _rms(x): return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")

def _bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Apply Butterworth band-pass filter."""
    if signal.size < 18:  # Need minimum samples for filtfilt
        return signal
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    # Ensure frequencies are in valid range (0, 1)
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def _welch_psd_temporal(e_resamp, fs=30.0):
    """Compute PSD in Hz (temporal frequency)."""
    if e_resamp.size < 32:
        return np.array([]), np.array([])
    nperseg = min(256, max(64, e_resamp.size//4))
    noverlap = nperseg//2
    return welch(e_resamp, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant", scaling="density")

def _compute_band_power(f, Pxx, band_low, band_high):
    """Compute power in a specific frequency band."""
    idx = np.where((f >= band_low) & (f <= band_high))[0]
    if len(idx) == 0:
        return 0.0
    # Integrate using trapezoidal rule
    return float(np.trapz(Pxx[idx], f[idx]))

def analyze_after_runs(paths:Dict[str,Dict[str,list]], spiral_pts, w, h, out_dir, target_fs=30.0,
                      tremor_band_low=4.0, tremor_band_high=10.0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Extract dot path from any available method
    dot_path = []
    for method in ['mp', 'hsv', 'led']:
        if method in paths and 'dot_xy' in paths[method]:
            dot_path = paths[method].get('dot_xy', [])
            if dot_path:
                break

    draw_paths_overlay(w,h,spiral_pts, paths.get('mp',{}).get('xy',[]),
                      paths.get('hsv',{}).get('xy',[]), str(out/"paths_overlay.png"),
                      dot_path=dot_path)
    summary = {}
    for m in paths.keys():
        t_list = paths.get(m,{}).get("times",[])
        e_list = paths.get(m,{}).get("err",[])
        t_grid, e_resamp = _resample_vs_time(t_list, e_list, target_fs=target_fs)

        # Raw signal analysis
        f, Pxx = _welch_psd_temporal(e_resamp, fs=target_fs)
        psd_data = [{"freq_hz": float(fi), "psd": float(pi)} for fi, pi in zip(f, Pxx)]

        # Band-pass filtered signal for tremor analysis
        e_tremor = _bandpass_filter(e_resamp, tremor_band_low, tremor_band_high, target_fs) if e_resamp.size >= 18 else np.array([])
        f_tremor, Pxx_tremor = _welch_psd_temporal(e_tremor, fs=target_fs) if e_tremor.size >= 32 else (np.array([]), np.array([]))
        psd_tremor_data = [{"freq_hz": float(fi), "psd": float(pi)} for fi, pi in zip(f_tremor, Pxx_tremor)]

        # Compute band power in tremor range
        tremor_power = _compute_band_power(f, Pxx, tremor_band_low, tremor_band_high)

        summary[m] = {
            "rms_px": _rms(e_resamp),
            "rms_tremor_band_px": _rms(e_tremor) if e_tremor.size else float("nan"),
            "tremor_band_power": tremor_power,
            "tremor_band_hz": [tremor_band_low, tremor_band_high],
            "resampled_len": int(e_resamp.size),
            "sampling_rate_hz": float(target_fs),
            "psd_temporal": psd_data,
            "psd_tremor_filtered": psd_tremor_data
        }
    return summary
