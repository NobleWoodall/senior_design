"""
Directional tremor analysis using 360-degree acceleration-based approach.
Designed for MRgFUS intraoperative tremor assessment.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import signal
import pandas as pd


def compute_derivatives(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute velocity and acceleration from position data.

    Args:
        t: Time array (seconds)
        x: X position array (pixels or mm)
        y: Y position array (pixels or mm)

    Returns:
        vx, vy: Velocity components
        ax, ay: Acceleration components
    """
    # Compute velocity using central differences
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)

    # Compute acceleration
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)

    return vx, vy, ax, ay


def bandpass_filter_acceleration(
    ax: np.ndarray,
    ay: np.ndarray,
    fs: float,
    lowcut: float = 4.0,
    highcut: float = 10.0,
    order: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply bandpass filter to acceleration to isolate tremor frequencies.

    Args:
        ax, ay: Acceleration components
        fs: Sampling frequency (Hz)
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        order: Filter order

    Returns:
        ax_filt, ay_filt: Filtered acceleration components
    """
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design Butterworth filter
    b, a = signal.butter(order, [low, high], btype='band')

    # Apply zero-phase filtering
    ax_filt = signal.filtfilt(b, a, ax)
    ay_filt = signal.filtfilt(b, a, ay)

    return ax_filt, ay_filt


def compute_directional_tremor(
    ax_tremor: np.ndarray,
    ay_tremor: np.ndarray,
    n_bins: int = 36
) -> Dict[str, Any]:
    """
    Compute tremor power distribution across 360 degrees.

    Args:
        ax_tremor: Tremor-filtered acceleration X component
        ay_tremor: Tremor-filtered acceleration Y component
        n_bins: Number of angular bins (36 = 10° per bin)

    Returns:
        Dictionary containing:
            - bin_edges: Angular bin edges (degrees)
            - bin_centers: Angular bin centers (degrees)
            - power_per_bin: Tremor power in each direction
            - worst_angle: Direction with maximum tremor
            - best_angle: Direction with minimum tremor
            - anisotropy_ratio: Worst/best ratio
            - mean_power: Average tremor power across all directions
    """
    # Compute instantaneous angle and magnitude
    angles = np.arctan2(ay_tremor, ax_tremor) * 180.0 / np.pi  # Convert to degrees
    angles = (angles + 360.0) % 360.0  # Ensure 0-360 range

    magnitudes = np.sqrt(ax_tremor**2 + ay_tremor**2)

    # Create angular bins
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Accumulate power in each bin
    power_per_bin = np.zeros(n_bins)

    for i in range(n_bins):
        # Find samples in this angular bin
        mask = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])

        # Accumulate RMS power (not just mean)
        if np.sum(mask) > 0:
            power_per_bin[i] = np.sqrt(np.mean(magnitudes[mask]**2))

    # Find worst and best directions
    worst_idx = np.argmax(power_per_bin)
    best_idx = np.argmin(power_per_bin)

    worst_angle = bin_centers[worst_idx]
    best_angle = bin_centers[best_idx]

    # Compute anisotropy ratio (how directional is the tremor?)
    anisotropy_ratio = power_per_bin[worst_idx] / (power_per_bin[best_idx] + 1e-10)

    # Overall mean power
    mean_power = np.mean(power_per_bin)

    return {
        'bin_edges': bin_edges.tolist(),
        'bin_centers': bin_centers.tolist(),
        'power_per_bin': power_per_bin.tolist(),
        'worst_angle': float(worst_angle),
        'best_angle': float(best_angle),
        'anisotropy_ratio': float(anisotropy_ratio),
        'mean_power': float(mean_power),
        'worst_power': float(power_per_bin[worst_idx]),
        'best_power': float(power_per_bin[best_idx])
    }


def analyze_directional_tremor(
    frames_csv_path: str,
    method: str = 'mp',
    tremor_band: Tuple[float, float] = (4.0, 10.0),
    sampling_rate: float = 30.0,
    n_angular_bins: int = 36
) -> Dict[str, Any]:
    """
    Complete directional tremor analysis from frames CSV file.

    Args:
        frames_csv_path: Path to frames.csv
        method: Tracking method to analyze ('mp', 'hsv', 'led')
        tremor_band: (low, high) frequency range in Hz
        sampling_rate: Target sampling rate for resampling (Hz)
        n_angular_bins: Number of angular bins for directional analysis

    Returns:
        Dictionary with directional tremor analysis results
    """
    # Load data
    df = pd.read_csv(frames_csv_path)

    # Filter by method
    df_method = df[df['method'] == method].copy()

    if len(df_method) == 0:
        raise ValueError(f"No data found for method '{method}'")

    # Extract time and position
    t = df_method['t_sec'].values
    x = df_method['x_px'].values
    y = df_method['y_px'].values

    # Remove NaN values
    valid_mask = ~(np.isnan(t) | np.isnan(x) | np.isnan(y))
    t = t[valid_mask]
    x = x[valid_mask]
    y = y[valid_mask]

    if len(t) < 10:
        raise ValueError(f"Insufficient valid data points: {len(t)}")

    # Resample to uniform time grid (needed for proper derivatives)
    t_uniform = np.arange(t[0], t[-1], 1.0 / sampling_rate)
    x_uniform = np.interp(t_uniform, t, x)
    y_uniform = np.interp(t_uniform, t, y)

    # Compute derivatives
    vx, vy, ax, ay = compute_derivatives(t_uniform, x_uniform, y_uniform)

    # Filter acceleration for tremor band
    ax_tremor, ay_tremor = bandpass_filter_acceleration(
        ax, ay,
        fs=sampling_rate,
        lowcut=tremor_band[0],
        highcut=tremor_band[1]
    )

    # Compute directional distribution
    directional_results = compute_directional_tremor(
        ax_tremor, ay_tremor,
        n_bins=n_angular_bins
    )

    # Add metadata
    directional_results['tremor_band_hz'] = list(tremor_band)
    directional_results['sampling_rate_hz'] = sampling_rate
    directional_results['n_samples'] = len(t_uniform)
    directional_results['duration_sec'] = float(t_uniform[-1] - t_uniform[0])

    # Store raw tremor acceleration for potential further analysis
    directional_results['acceleration_tremor_rms'] = float(
        np.sqrt(np.mean(ax_tremor**2 + ay_tremor**2))
    )

    return directional_results


def angle_to_compass(angle_deg: float) -> str:
    """
    Convert angle in degrees to compass direction for readability.

    Args:
        angle_deg: Angle in degrees (0=East, 90=North, etc.)

    Returns:
        Compass direction string (e.g., "NE", "SW")
    """
    # Convert to standard compass orientation (0=North)
    # Math: 0°=East, 90°=North -> Compass: 0°=North, 90°=East
    compass_angle = (90 - angle_deg) % 360

    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

    idx = int((compass_angle + 11.25) / 22.5) % 16
    return directions[idx]


def format_directional_summary(directional_results: Dict[str, Any]) -> str:
    """
    Format directional analysis results as human-readable summary.

    Args:
        directional_results: Output from analyze_directional_tremor()

    Returns:
        Formatted string summary
    """
    worst_angle = directional_results['worst_angle']
    best_angle = directional_results['best_angle']

    worst_compass = angle_to_compass(worst_angle)
    best_compass = angle_to_compass(best_angle)

    summary = f"""Directional Tremor Analysis:
  Worst direction: {worst_angle:.1f}° ({worst_compass}) - Power: {directional_results['worst_power']:.2f}
  Best direction:  {best_angle:.1f}° ({best_compass}) - Power: {directional_results['best_power']:.2f}
  Anisotropy ratio: {directional_results['anisotropy_ratio']:.2f}x
  Mean tremor power: {directional_results['mean_power']:.2f}
  Overall RMS acceleration (tremor band): {directional_results['acceleration_tremor_rms']:.2f}
  Duration: {directional_results['duration_sec']:.1f} sec, Samples: {directional_results['n_samples']}
"""
    return summary


if __name__ == '__main__':
    # Test with most recent session
    import os
    import glob
    import pandas as pd

    runs_dir = r"c:\Users\14845\MyStuff\ClassFolders\Senior Design\real_sense\runs"

    # Find most recent session directory
    session_dirs = glob.glob(os.path.join(runs_dir, "202*"))
    if session_dirs:
        latest_session = max(session_dirs, key=os.path.getmtime)
        frames_csv = os.path.join(latest_session, 'frames.csv')

        if os.path.exists(frames_csv):
            print(f"Analyzing: {latest_session}")

            # Detect which method is available
            df = pd.read_csv(frames_csv)
            available_methods = df['method'].unique()
            print(f"Available methods: {available_methods}")

            for method in available_methods:
                try:
                    print(f"\n=== Method: {method} ===")
                    results = analyze_directional_tremor(frames_csv, method=method)
                    print(format_directional_summary(results))
                except Exception as e:
                    print(f"Error analyzing {method}: {e}")
        else:
            print(f"No frames.csv found in {latest_session}")
    else:
        print("No session directories found")
