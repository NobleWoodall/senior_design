"""
Signal processing utilities.
- welch_psd: Compute the PSD with Welch's method from a 1D time series.
- detrend_if: Optional detrending before PSD.
- estimate_fs: From time vector.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import signal

def estimate_fs(t: np.ndarray, default: float = 30.0) -> float:
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return default
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(1.0 / np.median(dt)) if dt.size else default

def detrend_if(y: np.ndarray, do: bool) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return signal.detrend(y) if do else y

def welch_psd(y: np.ndarray, fs: float, nperseg: int = 256, detrend: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    y = detrend_if(y, detrend)
    nperseg = min(nperseg, len(y)) if len(y) > 0 else nperseg
    f, Pxx = signal.welch(y, fs=fs, nperseg=nperseg)
    return f, Pxx
