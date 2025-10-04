import json
import cv2
import numpy as np
from typing import List, Tuple, Dict
from scipy.signal import welch
from pathlib import Path

def draw_paths_overlay(width:int, height:int, spiral_pts:np.ndarray,
                       path_mp:List[Tuple[float,float]],
                       path_hsv:List[Tuple[float,float]], out_path:str):
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    if spiral_pts is not None and len(spiral_pts) > 1:
        pts = spiral_pts.astype(np.int32).reshape(-1,1,2)
        cv2.polylines(canvas, [pts], False, (200,200,200), 2)
    def _poly(points, color):
        if len(points) >= 2:
            p = np.array(points, dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(canvas, [p], False, color, 3, cv2.LINE_AA)
            cv2.circle(canvas, tuple(p[0,0]), 5, color, -1)
            cv2.circle(canvas, tuple(p[-1,0]), 5, color, -1)
    _poly(path_mp,  (255, 0, 0))   # bgr order
    _poly(path_hsv, (0, 0, 225))   # red-ish
    cv2.putText(canvas, "MediaPipe (blue) / HSV (red)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas)

def _resample_vs_s(s_list, e_list, ds=2.0):
    s = np.asarray(s_list, float)
    e = np.asarray(e_list, float)
    m = np.isfinite(s) & np.isfinite(e)
    s = s[m]
    e = e[m]
    if s.size < 8:
        return np.array([]), np.array([])
    o = np.argsort(s)
    s=s[o]
    e=e[o]
    us, idx = np.unique(s, return_inverse=True)
    e_avg = np.zeros_like(us)
    cnt = np.zeros_like(us)
    for i,u in enumerate(idx):
        e_avg[u]+=e[i]
        cnt[u]+=1
    e_avg = e_avg/np.maximum(1,cnt)
    s=us
    e=e_avg
    s0, s1 = float(s.min()), float(s.max())
    if not np.isfinite([s0,s1]).all() or s1<=s0:
        return np.array([]), np.array([])
    n = max(8, int((s1-s0)/max(ds,1e-6)))
    grid = np.linspace(s0,s1,n)
    return grid, np.interp(grid, s, e)

def _rms(x): return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")

def _welch_psd_spatial(e_resamp, ds=2.0):
    if e_resamp.size < 32:
        return np.array([]), np.array([])
    fs = 1.0/ds
    nperseg = min(256, max(64, e_resamp.size//4))
    noverlap = nperseg//2
    return welch(e_resamp, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant", scaling="density")

def _write_psd_csv(path, f, Pxx):
    import csv
    with open(path, "w", newline="") as fobj:
        w = csv.writer(fobj)
        w.writerow(["freq_cpp","psd"])
        for fi, pi in zip(f,Pxx): 
            w.writerow([f"{fi:.6f}", f"{pi:.12e}"])

def analyze_after_runs(paths:Dict[str,Dict[str,list]], spiral_pts, w, h, out_dir, ds_px=2.0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    draw_paths_overlay(w,h,spiral_pts, paths.get('mp',{}).get('xy',[]), paths.get('hsv',{}).get('xy',[]), str(out/"paths_overlay.png"))
    summary = {}
    for m in ["mp","hsv"]:
        s_list = paths.get(m,{}).get("s",[])
        e_list = paths.get(m,{}).get("err",[])
        s_grid, e_resamp = _resample_vs_s(s_list, e_list, ds=ds_px)
        f, Pxx = _welch_psd_spatial(e_resamp, ds=ds_px)
        summary[m] = {"rms_px": _rms(e_resamp), "resampled_len": int(e_resamp.size), "ds_px": float(ds_px)}
        _write_psd_csv(str(out/f"psd_{m}_spatial.csv"), f, Pxx)
    with open(out/"signal_summary.json","w") as fobj:
        json.dump(summary, fobj, indent=2)
    return summary
