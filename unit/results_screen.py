#!/usr/bin/env python3
"""
Results Screen: load a CSV of (t, x, y[, z]) and show:
- Position vs time charts (x(t), y(t)), each in its own window.
- Optional depth vs time.
- PSD of position signal(s) computed from x(t), y(t) after detrending.

Note:
- Each chart is its own plot window (no subplots) to keep UI simple.
- We compute PSD on position (pixels) by default. For tremor analysis,
  velocity or displacement after pixel->mm calibration may be more meaningful.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=str, help="CSV with columns like t,x,y[,z_m]")
    ap.add_argument("--fs", type=float, default=None, help="Override sampling rate Hz; else estimate from t.")
    ap.add_argument("--col-x", type=str, default="x")
    ap.add_argument("--col-y", type=str, default="y")
    ap.add_argument("--col-t", type=str, default="t")
    ap.add_argument("--detrend", action="store_true", help="Detrend before PSD.")
    ap.add_argument("--nperseg", type=int, default=256, help="nperseg for Welch PSD.")
    return ap.parse_args()

def maybe_plot_time_series(t, y, title):
    plt.figure()
    plt.plot(t, y)
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title(title)
    plt.show(block=False)

def plot_psd(y, fs, title, nperseg=256, detrend=False):
    y = np.asarray(y, dtype=float)
    if detrend:
        y = signal.detrend(y)
    f, Pxx = signal.welch(y, fs=fs, nperseg=min(nperseg, len(y)))
    plt.figure()
    plt.semilogy(f, Pxx)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title(title)
    plt.show(block=False)

def main():
    args = parse_args()
    df = pd.read_csv(args.csv_path)

    # Extract series
    t = df[args.col_t] if args.col_t in df.columns else np.arange(len(df)) * 0.033
    x = df[args.col_x] if args.col_x in df.columns else None
    y = df[args.col_y] if args.col_y in df.columns else None
    z = df["z_m"] if "z_m" in df.columns else None

    # Fix columns if the above had a small typo (dash)
    if isinstance(t, pd.Series) and t.name == "t":
        pass

    # Estimate fs if not provided
    if args.fs is None and len(t) > 1:
        dt = np.diff(t.astype(float))
        dt = dt[np.isfinite(dt) & (dt > 0)]
        fs = 1.0 / np.median(dt) if len(dt) else 30.0
    else:
        fs = args.fs if args.fs is not None else 30.0

    # Plot time series
    if x is not None:
        maybe_plot_time_series(t, x, "x(t)")
        plot_psd(x, fs, "PSD of x(t)", nperseg=args.nperseg, detrend=args.detrend)
    if y is not None:
        maybe_plot_time_series(t, y, "y(t)")
        plot_psd(y, fs, "PSD of y(t)", nperseg=args.nperseg, detrend=args.detrend)
    if z is not None:
        maybe_plot_time_series(t, z, "z(t) (meters)")

    plt.show()

if __name__ == "__main__":
    main()
