import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from pathlib import Path


def display_results(results_dict, overlay_path, output_dir):
    """
    Display consolidated results in a professional clinical dashboard for MRgFUS tremor assessment.

    Args:
        results_dict: Dictionary with trial_summaries, signal_analysis, directional_analysis, and metadata
        overlay_path: Path to paths_overlay.png image
        output_dir: Directory where results are saved
    """
    # Extract data
    trial_summaries = results_dict.get('trial_summaries', {})
    signal_analysis = results_dict.get('signal_analysis', {})
    directional_analysis = results_dict.get('directional_analysis', {})
    improvements = results_dict.get('improvements', {})
    session_metadata = results_dict.get('session_metadata', {})

    # Detect available methods
    available_methods = [m for m in trial_summaries.keys() if trial_summaries.get(m)]

    if not available_methods:
        print("No data to display")
        return

    # Use primary method for main displays
    primary_method = available_methods[0]

    # Professional medical color palette
    PRIMARY_TEXT = '#212121'      # Near-black for main text
    SECONDARY_TEXT = '#757575'    # Medium gray for labels
    ACCENT_BLUE = '#1565C0'       # Clinical blue for emphasis
    ACCENT_GREEN = '#2E7D32'      # Success green
    ACCENT_RED = '#C62828'        # Alert red
    ACCENT_YELLOW = '#F9A825'     # Warning yellow
    BG_LIGHT = '#E3F2FD'          # Light blue background
    BG_GRAY = '#F5F5F5'           # Light gray background
    GRID_COLOR = '#E0E0E0'        # Subtle grid

    # Create figure with refined layout
    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor('white')

    # 4 rows: header, main analysis, path, metrics/assessment
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 2.5, 2.2, 1.8], hspace=0.45, wspace=0.4)

    # ==================== ROW 0: PROFESSIONAL HEADER ====================
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')

    # Extract session info
    patient_id = session_metadata.get('patient_id', 'ANON')
    session_type = session_metadata.get('session_type', 'unknown')
    sonication_num = session_metadata.get('sonication_number', 0)
    date = session_metadata.get('date', '')
    clinician = session_metadata.get('clinician', '')

    # Get real tremor metrics
    sig = signal_analysis.get(primary_method, {})
    tremor_rms = sig.get('rms_tremor_band_px', 0)
    overall_rms = sig.get('rms_px', 0)
    tremor_power = sig.get('tremor_band_power', 0)

    # Find peak frequency
    peak_freq = None
    psd_filt_data = sig.get('psd_tremor_filtered', [])
    tremor_band = sig.get('tremor_band_hz', [4.0, 10.0])
    if psd_filt_data:
        freqs_f = [p['freq_hz'] for p in psd_filt_data]
        psds_f = [p['psd'] for p in psd_filt_data]
        tremor_mask = [(f >= tremor_band[0] and f <= tremor_band[1]) for f in freqs_f]
        if any(tremor_mask):
            tremor_psds = [p for p, m in zip(psds_f, tremor_mask) if m]
            tremor_freqs = [f for f, m in zip(freqs_f, tremor_mask) if m]
            if tremor_psds:
                peak_idx = np.argmax(tremor_psds)
                peak_freq = tremor_freqs[peak_idx]

    # Get improvement info
    improvement_pct = 0
    if improvements and primary_method in improvements:
        improvement_pct = improvements[primary_method].get('summary', {}).get('primary_metric_reduction_pct', 0)

    # Left side: Patient info
    if session_type == 'baseline':
        session_label = "BASELINE SESSION"
    else:
        session_label = f"POST-SONICATION #{sonication_num}"

    ax_header.text(0.02, 0.75, f"TREMOR ASSESSMENT", fontsize=14, fontweight='bold',
                   transform=ax_header.transAxes, color=SECONDARY_TEXT)
    ax_header.text(0.02, 0.45, f"Patient: {patient_id}", fontsize=16, fontweight='bold',
                   transform=ax_header.transAxes, color=PRIMARY_TEXT)
    ax_header.text(0.02, 0.15, f"{session_label} | {date}", fontsize=11,
                   transform=ax_header.transAxes, color=SECONDARY_TEXT)

    # Right side: Key Metrics Card
    card_x = 0.55
    card_width = 0.43
    card_y = 0.05

    # Background card
    card_rect = Rectangle((card_x, card_y), card_width, 0.9,
                          transform=ax_header.transAxes,
                          facecolor=BG_LIGHT, edgecolor=ACCENT_BLUE, linewidth=2,
                          zorder=0)
    ax_header.add_patch(card_rect)

    # Metrics in card
    ax_header.text(card_x + card_width/2, 0.80, "KEY MEASUREMENTS", fontsize=11, fontweight='bold',
                   ha='center', transform=ax_header.transAxes, color=ACCENT_BLUE)

    # Row 1: Tremor RMS
    ax_header.text(card_x + 0.02, 0.60, "Tremor Band RMS", fontsize=10,
                   transform=ax_header.transAxes, color=SECONDARY_TEXT)
    ax_header.text(card_x + card_width - 0.02, 0.60, f"{tremor_rms:.2f} px", fontsize=11, fontweight='bold',
                   ha='right', transform=ax_header.transAxes, color=PRIMARY_TEXT)

    # Row 2: Peak Frequency
    ax_header.text(card_x + 0.02, 0.42, "Peak Frequency", fontsize=10,
                   transform=ax_header.transAxes, color=SECONDARY_TEXT)
    freq_text = f"{peak_freq:.1f} Hz" if peak_freq else "N/A"
    ax_header.text(card_x + card_width - 0.02, 0.42, freq_text, fontsize=11, fontweight='bold',
                   ha='right', transform=ax_header.transAxes, color=PRIMARY_TEXT)

    # Row 3: Overall RMS
    ax_header.text(card_x + 0.02, 0.24, "Overall RMS", fontsize=10,
                   transform=ax_header.transAxes, color=SECONDARY_TEXT)
    ax_header.text(card_x + card_width - 0.02, 0.24, f"{overall_rms:.1f} px", fontsize=11, fontweight='bold',
                   ha='right', transform=ax_header.transAxes, color=PRIMARY_TEXT)

    # Row 4: Improvement (if available)
    if improvement_pct != 0:
        ax_header.text(card_x + 0.02, 0.06, "Change from Baseline", fontsize=10,
                       transform=ax_header.transAxes, color=SECONDARY_TEXT)

        if improvement_pct > 0:
            change_text = f"↓ {improvement_pct:.1f}%"
            change_color = ACCENT_GREEN if improvement_pct >= 30 else ACCENT_YELLOW if improvement_pct >= 10 else SECONDARY_TEXT
        else:
            change_text = f"↑ {abs(improvement_pct):.1f}%"
            change_color = ACCENT_RED

        ax_header.text(card_x + card_width - 0.02, 0.06, change_text, fontsize=12, fontweight='bold',
                       ha='right', transform=ax_header.transAxes, color=change_color)

    # ==================== ROW 1 LEFT: DIRECTIONAL TREMOR ====================
    ax_polar = fig.add_subplot(gs[1, 0], projection='polar')

    if directional_analysis and primary_method in directional_analysis:
        dir_data = directional_analysis[primary_method]
        bin_centers = np.array(dir_data['bin_centers'])
        power_per_bin = np.array(dir_data['power_per_bin'])
        worst_angle = dir_data['worst_angle']
        best_angle = dir_data['best_angle']

        # Convert to radians
        theta = np.deg2rad(bin_centers)
        theta = np.append(theta, theta[0])
        power_per_bin = np.append(power_per_bin, power_per_bin[0])

        # Plot with professional styling
        ax_polar.plot(theta, power_per_bin, color=ACCENT_RED, linewidth=2.5)
        ax_polar.fill(theta, power_per_bin, alpha=0.25, color=ACCENT_RED)

        # Mark worst direction
        worst_rad = np.deg2rad(worst_angle)
        worst_power = dir_data['worst_power']
        ax_polar.plot([worst_rad, worst_rad], [0, worst_power], color=ACCENT_RED, linewidth=3.5, alpha=0.8)
        ax_polar.scatter([worst_rad], [worst_power], color=ACCENT_RED, s=120, zorder=5, edgecolors='white', linewidth=2)

        ax_polar.set_theta_zero_location('E')
        ax_polar.set_theta_direction(1)
        ax_polar.set_title(f'Directional Tremor Analysis\nWorst: {worst_angle:.0f}°  |  Best: {best_angle:.0f}°',
                          fontsize=13, fontweight='bold', pad=20, color=ACCENT_BLUE)
        ax_polar.set_ylim(0, max(power_per_bin) * 1.1)
        ax_polar.grid(True, alpha=0.2, color=GRID_COLOR)
        ax_polar.tick_params(labelsize=9, colors=SECONDARY_TEXT)
    else:
        ax_polar.text(0, 0, 'No directional data', ha='center', va='center', fontsize=11, color=SECONDARY_TEXT)
        ax_polar.set_title('Directional Tremor Analysis', fontsize=13, fontweight='bold', color=ACCENT_BLUE)

    # ==================== ROW 1 RIGHT: FREQUENCY ANALYSIS ====================
    ax_psd = fig.add_subplot(gs[1, 1])

    psd_plotted = False
    color_map = {'mp': ACCENT_BLUE, 'hsv': ACCENT_GREEN, 'led': ACCENT_YELLOW}

    for method in signal_analysis.keys():
        if signal_analysis.get(method):
            psd_filt_data = signal_analysis[method].get('psd_tremor_filtered', [])
            tremor_band = signal_analysis[method].get('tremor_band_hz', [4.0, 10.0])
            color = color_map.get(method, SECONDARY_TEXT)

            if psd_filt_data:
                freqs_f = [p['freq_hz'] for p in psd_filt_data]
                psds_f = [p['psd'] for p in psd_filt_data]

                ax_psd.plot(freqs_f, psds_f, label=method.upper(), color=color, alpha=0.85, linewidth=2.5)
                psd_plotted = True

    if psd_plotted:
        # Shade tremor band
        ax_psd.axvspan(tremor_band[0], tremor_band[1], alpha=0.15, color=ACCENT_YELLOW, zorder=0)

        # Mark peak if found
        if peak_freq is not None:
            ax_psd.axvline(peak_freq, color=ACCENT_RED, linestyle='--', linewidth=2, alpha=0.7, label=f'Peak: {peak_freq:.1f} Hz')

        ax_psd.set_xlabel('Frequency (Hz)', fontsize=11, color=PRIMARY_TEXT)
        ax_psd.set_ylabel('Power Spectral Density', fontsize=11, color=PRIMARY_TEXT)
        title = f'Tremor Frequency Analysis\nPeak: {peak_freq:.1f} Hz' if peak_freq else 'Tremor Frequency Analysis'
        ax_psd.set_title(title, fontsize=13, fontweight='bold', color=ACCENT_BLUE, pad=12)
        ax_psd.legend(fontsize=9, framealpha=0.9)
        ax_psd.grid(True, alpha=0.2, color=GRID_COLOR, linestyle='-')
        ax_psd.set_xlim(left=0, right=15)
        ax_psd.set_ylim(bottom=0)
        ax_psd.tick_params(labelsize=9, colors=SECONDARY_TEXT)

    # ==================== ROW 2: PATH TRACE ====================
    ax_path = fig.add_subplot(gs[2, :])

    if os.path.exists(overlay_path):
        img = plt.imread(overlay_path)
        ax_path.imshow(img)
        ax_path.axis('off')
        ax_path.set_title('Finger Path Trace', fontsize=13, fontweight='bold', color=ACCENT_BLUE, pad=10)
    else:
        ax_path.text(0.5, 0.5, 'Path overlay not available', ha='center', va='center',
                    fontsize=11, color=SECONDARY_TEXT)
        ax_path.axis('off')
        ax_path.set_title('Finger Path Trace', fontsize=13, fontweight='bold', color=ACCENT_BLUE)

    # ==================== ROW 3 LEFT: TREMOR METRICS ====================
    ax_metrics = fig.add_subplot(gs[3, 0])
    ax_metrics.axis('off')

    # Professional table-style layout
    y_pos = 0.92
    line_height = 0.13

    ax_metrics.text(0.5, y_pos, "TREMOR ANALYSIS", ha='center', fontsize=12, fontweight='bold',
                   transform=ax_metrics.transAxes, color=ACCENT_BLUE)
    y_pos -= line_height * 1.2

    if signal_analysis and primary_method in signal_analysis:
        sig = signal_analysis[primary_method]

        # Tremor Band Power
        ax_metrics.text(0.05, y_pos, "Tremor Band Power", fontsize=10,
                       transform=ax_metrics.transAxes, color=SECONDARY_TEXT)
        ax_metrics.text(0.95, y_pos, f"{sig.get('tremor_band_power', 0):.3f}", fontsize=10, fontweight='bold',
                       ha='right', transform=ax_metrics.transAxes, color=PRIMARY_TEXT)
        y_pos -= line_height

        # RMS Tremor Band
        ax_metrics.text(0.05, y_pos, "RMS (Tremor Band)", fontsize=10,
                       transform=ax_metrics.transAxes, color=SECONDARY_TEXT)
        ax_metrics.text(0.95, y_pos, f"{sig.get('rms_tremor_band_px', 0):.2f} px", fontsize=10, fontweight='bold',
                       ha='right', transform=ax_metrics.transAxes, color=PRIMARY_TEXT)
        y_pos -= line_height

        # RMS Overall
        ax_metrics.text(0.05, y_pos, "RMS (Overall)", fontsize=10,
                       transform=ax_metrics.transAxes, color=SECONDARY_TEXT)
        ax_metrics.text(0.95, y_pos, f"{sig.get('rms_px', 0):.1f} px", fontsize=10, fontweight='bold',
                       ha='right', transform=ax_metrics.transAxes, color=PRIMARY_TEXT)
        y_pos -= line_height

    if directional_analysis and primary_method in directional_analysis:
        dir_data = directional_analysis[primary_method]

        # Worst Direction
        ax_metrics.text(0.05, y_pos, "Worst Direction", fontsize=10,
                       transform=ax_metrics.transAxes, color=SECONDARY_TEXT)
        ax_metrics.text(0.95, y_pos, f"{dir_data.get('worst_angle', 0):.0f}°", fontsize=10, fontweight='bold',
                       ha='right', transform=ax_metrics.transAxes, color=PRIMARY_TEXT)
        y_pos -= line_height

        # Anisotropy Ratio
        ax_metrics.text(0.05, y_pos, "Anisotropy Ratio", fontsize=10,
                       transform=ax_metrics.transAxes, color=SECONDARY_TEXT)
        ax_metrics.text(0.95, y_pos, f"{dir_data.get('anisotropy_ratio', 1):.2f}×", fontsize=10, fontweight='bold',
                       ha='right', transform=ax_metrics.transAxes, color=PRIMARY_TEXT)

    # Light background
    bg_rect = Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax_metrics.transAxes,
                        facecolor=BG_GRAY, edgecolor=GRID_COLOR, linewidth=1, zorder=-1)
    ax_metrics.add_patch(bg_rect)

    # ==================== ROW 3 RIGHT: CLINICAL ASSESSMENT ====================
    ax_assess = fig.add_subplot(gs[3, 1])
    ax_assess.axis('off')

    y_pos = 0.92
    line_height = 0.13

    if improvements and primary_method in improvements:
        ax_assess.text(0.5, y_pos, "CLINICAL ASSESSMENT", ha='center', fontsize=12, fontweight='bold',
                      transform=ax_assess.transAxes, color=ACCENT_BLUE)
        y_pos -= line_height * 1.2

        imp = improvements[primary_method]
        summary = imp.get('summary', {})
        recommendation = summary.get('recommendation', 'N/A')

        # Status/Recommendation
        ax_assess.text(0.05, y_pos, "Status:", fontsize=10,
                      transform=ax_assess.transAxes, color=SECONDARY_TEXT)
        y_pos -= line_height * 0.8
        ax_assess.text(0.05, y_pos, recommendation, fontsize=9, style='italic',
                      transform=ax_assess.transAxes, color=PRIMARY_TEXT, wrap=True)
        y_pos -= line_height * 1.2

        # Improvements
        if 'tremor_power' in imp:
            tp = imp['tremor_power']
            ax_assess.text(0.05, y_pos, "Tremor Power Change", fontsize=10,
                          transform=ax_assess.transAxes, color=SECONDARY_TEXT)
            change_val = tp.get('percent_reduction', 0)
            change_color = ACCENT_GREEN if change_val >= 30 else ACCENT_YELLOW if change_val >= 10 else ACCENT_RED if change_val < 0 else SECONDARY_TEXT
            ax_assess.text(0.95, y_pos, f"{change_val:+.1f}%", fontsize=10, fontweight='bold',
                          ha='right', transform=ax_assess.transAxes, color=change_color)
            y_pos -= line_height

            ax_assess.text(0.15, y_pos, f"Baseline: {tp.get('baseline', 0):.3f}", fontsize=9,
                          transform=ax_assess.transAxes, color=SECONDARY_TEXT)
            y_pos -= line_height * 0.7
            ax_assess.text(0.15, y_pos, f"Current: {tp.get('current', 0):.3f}", fontsize=9,
                          transform=ax_assess.transAxes, color=SECONDARY_TEXT)

        # Color-coded background
        if improvement_pct >= 30:
            bg_color = ACCENT_GREEN
            bg_alpha = 0.12
        elif improvement_pct >= 10:
            bg_color = ACCENT_YELLOW
            bg_alpha = 0.12
        elif improvement_pct < 0:
            bg_color = ACCENT_RED
            bg_alpha = 0.12
        else:
            bg_color = BG_GRAY
            bg_alpha = 1.0

    else:
        # Baseline session
        ax_assess.text(0.5, y_pos, "BASELINE SESSION", ha='center', fontsize=12, fontweight='bold',
                      transform=ax_assess.transAxes, color=ACCENT_BLUE)
        y_pos -= line_height * 1.5

        ax_assess.text(0.5, y_pos, "No comparison data available", ha='center', fontsize=10,
                      transform=ax_assess.transAxes, color=SECONDARY_TEXT, style='italic')
        y_pos -= line_height * 1.3

        ax_assess.text(0.05, y_pos, "Tremor Band Power", fontsize=10,
                      transform=ax_assess.transAxes, color=SECONDARY_TEXT)
        ax_assess.text(0.95, y_pos, f"{tremor_power:.3f}", fontsize=10, fontweight='bold',
                      ha='right', transform=ax_assess.transAxes, color=PRIMARY_TEXT)

        bg_color = BG_GRAY
        bg_alpha = 1.0

    bg_rect = Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax_assess.transAxes,
                        facecolor=bg_color, edgecolor=GRID_COLOR, linewidth=1, alpha=bg_alpha, zorder=-1)
    ax_assess.add_patch(bg_rect)

    # ==================== FOOTER: TECHNICAL DETAILS ====================
    # Small footer text at very bottom
    footer_text = ""
    if trial_summaries and primary_method in trial_summaries:
        trial = trial_summaries[primary_method]
        footer_text = (f"Technical: Duration {trial.get('duration_sec', 0):.1f}s  |  "
                      f"Samples {trial.get('sample_count', 0)}  |  "
                      f"FPS {trial.get('fps_avg', 0):.1f}  |  "
                      f"Loss {trial.get('tracking_loss_rate', 0)*100:.1f}%  |  "
                      f"Depth {trial.get('mean_depth_mm', 0):.0f}mm")

    fig.text(0.5, 0.01, footer_text, ha='center', fontsize=7, color=SECONDARY_TEXT, style='italic')

    plt.tight_layout(rect=[0, 0.02, 1, 1])  # Leave space for footer

    # Save figure
    save_path = Path(output_dir) / "results_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
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
