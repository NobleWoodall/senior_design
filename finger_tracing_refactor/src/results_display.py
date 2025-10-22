import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

def display_results(results_dict, overlay_path, output_dir):
    """
    Display consolidated results in a clean matplotlib figure.

    Args:
        results_dict: Dictionary with trial_summaries and signal_analysis
        overlay_path: Path to paths_overlay.png image
        output_dir: Directory where results are saved
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Finger Tracing Results', fontsize=16, fontweight='bold')

    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Top: Path overlay image
    ax_img = fig.add_subplot(gs[0, :])
    if os.path.exists(overlay_path):
        img = plt.imread(overlay_path)
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_title('Traced Paths: MediaPipe (blue) vs HSV (red)', fontsize=12)
    else:
        ax_img.text(0.5, 0.5, 'Overlay image not found', ha='center', va='center')
        ax_img.axis('off')

    # Middle-left: Error metrics comparison
    ax_metrics = fig.add_subplot(gs[1, 0])
    trial_summaries = results_dict.get('trial_summaries', {})

    methods = []
    rmse_vals = []
    median_vals = []
    p95_vals = []

    for method in ['mp', 'hsv']:
        if method in trial_summaries:
            s = trial_summaries[method]
            methods.append(method.upper())
            rmse_vals.append(s.get('rmse_time_weighted', 0))
            median_vals.append(s.get('err_px', {}).get('median', 0))
            p95_vals.append(s.get('err_px', {}).get('p95', 0))

    if methods:
        x = np.arange(len(methods))
        width = 0.25

        ax_metrics.bar(x - width, rmse_vals, width, label='RMSE', color='#3498db')
        ax_metrics.bar(x, median_vals, width, label='Median', color='#2ecc71')
        ax_metrics.bar(x + width, p95_vals, width, label='P95', color='#e74c3c')

        ax_metrics.set_ylabel('Error (pixels)', fontsize=10)
        ax_metrics.set_title('Tracking Error Metrics', fontsize=11, fontweight='bold')
        ax_metrics.set_xticks(x)
        ax_metrics.set_xticklabels(methods)
        ax_metrics.legend(fontsize=9)
        ax_metrics.grid(axis='y', alpha=0.3)

    # Middle-right: Tracking loss and stats
    ax_stats = fig.add_subplot(gs[1, 1])
    stats_text = "Performance Summary\n" + "="*30 + "\n\n"

    for method in ['mp', 'hsv']:
        if method in trial_summaries:
            s = trial_summaries[method]
            stats_text += f"{method.upper()} Tracking:\n"
            stats_text += f"  Duration: {s.get('duration_sec', 0):.2f}s\n"
            stats_text += f"  Samples: {s.get('sample_count', 0)}\n"
            stats_text += f"  FPS: {s.get('fps_avg', 0):.1f}\n"
            stats_text += f"  Loss Rate: {s.get('tracking_loss_rate', 0)*100:.1f}%\n"
            stats_text += f"  Mean Depth: {s.get('mean_depth_mm', 0):.1f}mm\n\n"

    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=9, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax_stats.axis('off')

    # Row 3 Left: Signal RMS comparison (raw + tremor band)
    ax_rms = fig.add_subplot(gs[2, 0])
    signal_analysis = results_dict.get('signal_analysis', {})

    if signal_analysis:
        methods_sig = []
        rms_vals_sig = []
        rms_tremor_vals = []

        for method in ['mp', 'hsv']:
            if method in signal_analysis:
                methods_sig.append(method.upper())
                rms_vals_sig.append(signal_analysis[method].get('rms_px', 0))
                rms_tremor_vals.append(signal_analysis[method].get('rms_tremor_band_px', 0))

        if methods_sig:
            x = np.arange(len(methods_sig))
            width = 0.35
            colors = ['#3498db', '#e74c3c']

            bars1 = ax_rms.bar(x - width/2, rms_vals_sig, width, label='Raw RMS', color=colors, alpha=0.7)
            bars2 = ax_rms.bar(x + width/2, rms_tremor_vals, width, label='Tremor Band RMS',
                             color=colors, alpha=0.4, hatch='//')

            ax_rms.set_ylabel('RMS Error (pixels)', fontsize=10)
            ax_rms.set_title('Signal RMS: Raw vs Tremor Band (4-10 Hz)', fontsize=11, fontweight='bold')
            ax_rms.set_xticks(x)
            ax_rms.set_xticklabels(methods_sig)
            ax_rms.legend(fontsize=9)
            ax_rms.grid(axis='y', alpha=0.3)

    # Row 3 Right: Band-Pass Filtered PSD (tremor band only) - LINEAR SCALE
    ax_psd = fig.add_subplot(gs[2, 1])

    psd_plotted = False
    tremor_band = [4.0, 10.0]  # Default
    for method, color in zip(['mp', 'hsv'], ['#3498db', '#e74c3c']):
        if method in signal_analysis:
            # Use the band-pass filtered PSD to remove low-frequency drift
            psd_filt_data = signal_analysis[method].get('psd_tremor_filtered', [])
            tremor_band = signal_analysis[method].get('tremor_band_hz', [4.0, 10.0])
            if psd_filt_data:
                freqs_f = [p['freq_hz'] for p in psd_filt_data]
                psds_f = [p['psd'] for p in psd_filt_data]
                # LINEAR scale to see tremor peaks clearly
                ax_psd.plot(freqs_f, psds_f, label=method.upper(), color=color, alpha=0.8, linewidth=2)
                psd_plotted = True

    if psd_plotted:
        # Shade tremor band region
        ax_psd.axvspan(tremor_band[0], tremor_band[1], alpha=0.25, color='yellow', label='Tremor Band', zorder=0)
        ax_psd.set_xlabel('Frequency (Hz)', fontsize=11)
        ax_psd.set_ylabel('PSD (px²/Hz)', fontsize=11)
        ax_psd.set_title('Band-Pass Filtered PSD (4-10 Hz Tremor)', fontsize=12, fontweight='bold')
        ax_psd.legend(fontsize=10, loc='upper right')
        ax_psd.grid(True, alpha=0.3, linestyle='--')
        ax_psd.set_xlim(left=0, right=15)
        ax_psd.set_ylim(bottom=0)  # Start at 0 for linear scale

    plt.tight_layout()

    # Save figure
    save_path = Path(output_dir) / "results_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Results figure saved: {save_path}")

    # Show the window
    plt.show()

def show_results_from_file(results_path):
    """Load results.json and display it."""
    import json

    results_dir = Path(results_path).parent

    with open(results_path, 'r') as f:
        results = json.load(f)

    overlay_path = results_dir / "paths_overlay.png"
    display_results(results, str(overlay_path), str(results_dir))
