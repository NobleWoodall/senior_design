"""
Multi-trial comparison HTML generator for MRgFUS tremor assessment.

Generates side-by-side comparison visualizations showing progress across multiple trials.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_comparison_html(patient_id: str, trials_data: List[Tuple[str, Dict]], output_path: str = None):
    """
    Generate HTML comparison report for multiple trials.

    Args:
        patient_id: Patient identifier
        trials_data: List of (trial_name, results_dict) tuples
        output_path: Where to save HTML (defaults to runs/comparison_<patient_id>_<timestamp>.html)

    Returns:
        Path to generated HTML file
    """
    if not trials_data:
        raise ValueError("No trial data provided for comparison")

    if output_path is None:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"runs/comparison_{patient_id}_{timestamp}.html"

    # Generate HTML content
    html = generate_comparison_content(patient_id, trials_data)

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Comparison HTML saved to: {output_path}")
    return output_path


def generate_comparison_content(patient_id: str, trials_data: List[Tuple[str, Dict]]):
    """Generate the complete HTML content for comparison view."""

    # Create comparison charts
    frequency_chart = create_frequency_comparison_chart(trials_data)
    directional_chart = create_directional_comparison_chart(trials_data)
    metrics_table = create_metrics_comparison_table(trials_data)
    trend_chart = create_trend_chart(trials_data)

    # Get patient metadata from first trial
    first_trial_metadata = trials_data[0][1].get('session_metadata', {})
    clinician = first_trial_metadata.get('clinician', 'Unknown')
    trial_names = [name for name, _ in trials_data]

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Trial Comparison - {patient_id}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            line-height: 1.6;
            padding: 20px;
        }}

        .container {{
            max-width: 1800px;
            margin: 0 auto;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}

        .header h1 {{
            font-size: 2.2em;
            margin-bottom: 10px;
            color: white;
        }}

        .header-info {{
            display: flex;
            gap: 30px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}

        .info-item {{
            display: flex;
            flex-direction: column;
        }}

        .info-label {{
            font-size: 0.85em;
            opacity: 0.9;
            margin-bottom: 3px;
        }}

        .info-value {{
            font-size: 1.1em;
            font-weight: 600;
        }}

        .trial-badges {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 15px;
        }}

        .trial-badge {{
            background: rgba(255, 255, 255, 0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 500;
        }}

        .section {{
            background: #2d2d2d;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }}

        .section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}

        .chart-container {{
            background: #1f1f1f;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}

        .metric-card {{
            background: #1f1f1f;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}

        .metric-label {{
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 8px;
        }}

        .metric-value {{
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 5px;
        }}

        .metric-change {{
            font-size: 0.95em;
            font-weight: 500;
        }}

        .improvement {{
            color: #4ade80;
        }}

        .degradation {{
            color: #f87171;
        }}

        .neutral {{
            color: #fbbf24;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #404040;
        }}

        th {{
            background: #1f1f1f;
            font-weight: 600;
            color: #667eea;
        }}

        tr:hover {{
            background: #3a3a3a;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Multi-Trial Progress Report</h1>
            <div class="header-info">
                <div class="info-item">
                    <span class="info-label">Patient ID</span>
                    <span class="info-value">{patient_id}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Clinician</span>
                    <span class="info-value">{clinician}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Trials Compared</span>
                    <span class="info-value">{len(trials_data)}</span>
                </div>
            </div>
            <div class="trial-badges">
                {' '.join(f'<span class="trial-badge">{name}</span>' for name in trial_names)}
            </div>
        </div>

        <div class="section">
            <h2>Tremor Power Trend</h2>
            <div class="chart-container">
                <div id="trend-chart"></div>
            </div>
        </div>

        <div class="section">
            <h2>Metrics Comparison</h2>
            {metrics_table}
        </div>

        <div class="section">
            <h2>Frequency Analysis Comparison</h2>
            <div class="chart-container">
                <div id="frequency-chart"></div>
            </div>
            <p style="margin-top: 10px; opacity: 0.7; font-size: 0.9em;">
                Overlaid power spectral density (PSD) for each trial. Lower values in the tremor band (4-10 Hz) indicate improvement.
            </p>
        </div>

        <div class="section">
            <h2>Directional Tremor Analysis</h2>
            <div class="chart-container">
                <div id="directional-chart"></div>
            </div>
            <p style="margin-top: 10px; opacity: 0.7; font-size: 0.9em;">
                Polar plot showing tremor power distribution across directions (0-360°). Helps identify directional treatment effects.
            </p>
        </div>

        <div class="footer">
            <p>MRgFUS Tremor Assessment System | Multi-Trial Comparison Report</p>
        </div>
    </div>

    <script>
        // Render charts
        {frequency_chart}
        {directional_chart}
        {trend_chart}

        // Make plots responsive
        window.addEventListener('resize', function() {{
            Plotly.Plots.resize('frequency-chart');
            Plotly.Plots.resize('directional-chart');
            Plotly.Plots.resize('trend-chart');
        }});
    </script>
</body>
</html>
"""

    return html


def create_frequency_comparison_chart(trials_data: List[Tuple[str, Dict]]) -> str:
    """Create overlaid frequency analysis chart for multiple trials."""

    # Color palette for trials
    colors = ['#667eea', '#4ade80', '#fbbf24', '#f87171', '#60a5fa', '#a78bfa', '#fb923c']

    fig = go.Figure()

    for idx, (trial_name, results) in enumerate(trials_data):
        signal_analysis = results.get('signal_analysis', {})

        # Try to find a valid method (hsv, mp, led)
        method_data = None
        for method in ['hsv', 'mp', 'led']:
            if method in signal_analysis:
                method_data = signal_analysis[method]
                break

        if not method_data:
            continue

        # Get PSD data
        psd_data = method_data.get('psd_tremor_filtered', [])
        if not psd_data:
            continue

        frequencies = [item["freq_hz"] for item in psd_data]
        powers = [item["psd"] for item in psd_data]

        # Add trace for this trial
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=frequencies,
            y=powers,
            mode='lines',
            name=trial_name,
            line=dict(width=3, color=color),
            hovertemplate='<b>%{fullData.name}</b><br>Freq: %{x:.2f} Hz<br>Power: %{y:.2f}<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1f1f1f',
        plot_bgcolor='#1f1f1f',
        font=dict(color='#e0e0e0'),
        xaxis=dict(
            title='Frequency (Hz)',
            gridcolor='#404040',
            range=[4, 10]
        ),
        yaxis=dict(
            title='Power Spectral Density',
            gridcolor='#404040'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        hovermode='x unified',
        height=450
    )

    return f"Plotly.newPlot('frequency-chart', {fig.to_json()}.data, {fig.to_json()}.layout);"


def create_directional_comparison_chart(trials_data: List[Tuple[str, Dict]]) -> str:
    """Create overlaid directional tremor polar chart for multiple trials."""

    colors = ['#667eea', '#4ade80', '#fbbf24', '#f87171', '#60a5fa', '#a78bfa', '#fb923c']

    fig = go.Figure()

    for idx, (trial_name, results) in enumerate(trials_data):
        directional_analysis = results.get('directional_analysis', {})

        # Try to find a valid method
        method_data = None
        for method in ['hsv', 'mp', 'led']:
            if method in directional_analysis:
                method_data = directional_analysis[method]
                break

        if not method_data:
            continue

        bin_centers = method_data.get('bin_centers', [])
        power_per_bin = method_data.get('power_per_bin', [])

        if not bin_centers or not power_per_bin:
            continue

        # Add trace for this trial
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatterpolar(
            r=power_per_bin,
            theta=bin_centers,
            mode='lines',
            name=trial_name,
            line=dict(width=3, color=color),
            fill='toself',
            opacity=0.5
        ))

    # Update layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1f1f1f',
        plot_bgcolor='#1f1f1f',
        font=dict(color='#e0e0e0'),
        polar=dict(
            radialaxis=dict(
                visible=True,
                gridcolor='#404040'
            ),
            angularaxis=dict(
                gridcolor='#404040',
                direction='counterclockwise',
                rotation=0  # Counterclockwise makes 90° point up
            )
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        height=500
    )

    return f"Plotly.newPlot('directional-chart', {fig.to_json()}.data, {fig.to_json()}.layout);"


def create_trend_chart(trials_data: List[Tuple[str, Dict]]) -> str:
    """Create line chart showing tremor power trend across trials."""

    trial_names = []
    tremor_powers = []
    tremor_rms = []

    for trial_name, results in trials_data:
        signal_analysis = results.get('signal_analysis', {})

        # Try to find a valid method
        method_data = None
        for method in ['hsv', 'mp', 'led']:
            if method in signal_analysis:
                method_data = signal_analysis[method]
                break

        if not method_data:
            continue

        trial_names.append(trial_name)
        tremor_powers.append(method_data.get('tremor_band_power', 0))
        tremor_rms.append(method_data.get('rms_tremor_band_px', 0))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Tremor power trace
    fig.add_trace(
        go.Scatter(
            x=trial_names,
            y=tremor_powers,
            mode='lines+markers',
            name='Tremor Power',
            line=dict(width=3, color='#667eea'),
            marker=dict(size=10)
        ),
        secondary_y=False
    )

    # Tremor RMS trace
    fig.add_trace(
        go.Scatter(
            x=trial_names,
            y=tremor_rms,
            mode='lines+markers',
            name='Tremor RMS (px)',
            line=dict(width=3, color='#4ade80', dash='dash'),
            marker=dict(size=10)
        ),
        secondary_y=True
    )

    # Update layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1f1f1f',
        plot_bgcolor='#1f1f1f',
        font=dict(color='#e0e0e0'),
        xaxis=dict(
            title='Trial',
            gridcolor='#404040'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        hovermode='x unified',
        height=400
    )

    fig.update_yaxes(title_text="Tremor Power", gridcolor='#404040', secondary_y=False)
    fig.update_yaxes(title_text="Tremor RMS (px)", gridcolor='#404040', secondary_y=True)

    return f"Plotly.newPlot('trend-chart', {fig.to_json()}.data, {fig.to_json()}.layout);"


def create_metrics_comparison_table(trials_data: List[Tuple[str, Dict]]) -> str:
    """Create HTML table comparing key metrics across trials."""

    # Extract metrics for each trial
    metrics_by_trial = []

    for trial_name, results in trials_data:
        signal_analysis = results.get('signal_analysis', {})

        # Try to find a valid method
        method_data = None
        for method in ['hsv', 'mp', 'led']:
            if method in signal_analysis:
                method_data = signal_analysis[method]
                break

        if not method_data:
            metrics_by_trial.append({
                'trial_name': trial_name,
                'tremor_power': 'N/A',
                'tremor_rms': 'N/A',
                'overall_rms': 'N/A',
                'peak_freq': 'N/A'
            })
            continue

        # Calculate peak frequency
        psd_data = method_data.get('psd_tremor_filtered', [])
        peak_freq = 0
        if psd_data:
            max_power = max(item["psd"] for item in psd_data)
            for item in psd_data:
                if item["psd"] == max_power:
                    peak_freq = item["freq_hz"]
                    break

        metrics_by_trial.append({
            'trial_name': trial_name,
            'tremor_power': method_data.get('tremor_band_power', 0),
            'tremor_rms': method_data.get('rms_tremor_band_px', 0),
            'overall_rms': method_data.get('rms_px', 0),
            'peak_freq': peak_freq
        })

    # Generate table HTML
    table_html = '<table><thead><tr>'
    table_html += '<th>Trial</th>'
    table_html += '<th>Tremor Power</th>'
    table_html += '<th>Change</th>'
    table_html += '<th>Tremor RMS (px)</th>'
    table_html += '<th>Change</th>'
    table_html += '<th>Overall RMS (px)</th>'
    table_html += '<th>Peak Frequency (Hz)</th>'
    table_html += '</tr></thead><tbody>'

    baseline_tremor_power = metrics_by_trial[0]['tremor_power'] if metrics_by_trial else 0
    baseline_tremor_rms = metrics_by_trial[0]['tremor_rms'] if metrics_by_trial else 0

    for idx, metrics in enumerate(metrics_by_trial):
        table_html += '<tr>'
        table_html += f'<td><strong>{metrics["trial_name"]}</strong></td>'

        # Tremor Power
        tremor_power = metrics['tremor_power']
        if tremor_power == 'N/A':
            table_html += '<td>N/A</td><td>-</td>'
        else:
            table_html += f'<td>{tremor_power:.2f}</td>'

            # Calculate change
            if idx == 0 or baseline_tremor_power == 0:
                table_html += '<td class="neutral">Baseline</td>'
            else:
                change_pct = ((tremor_power - baseline_tremor_power) / baseline_tremor_power) * 100
                change_class = 'improvement' if change_pct < 0 else 'degradation'
                table_html += f'<td class="{change_class}">{change_pct:+.1f}%</td>'

        # Tremor RMS
        tremor_rms = metrics['tremor_rms']
        if tremor_rms == 'N/A':
            table_html += '<td>N/A</td><td>-</td>'
        else:
            table_html += f'<td>{tremor_rms:.2f}</td>'

            # Calculate change
            if idx == 0 or baseline_tremor_rms == 0:
                table_html += '<td class="neutral">Baseline</td>'
            else:
                change_pct = ((tremor_rms - baseline_tremor_rms) / baseline_tremor_rms) * 100
                change_class = 'improvement' if change_pct < 0 else 'degradation'
                table_html += f'<td class="{change_class}">{change_pct:+.1f}%</td>'

        # Overall RMS
        overall_rms = metrics['overall_rms']
        if overall_rms == 'N/A':
            table_html += '<td>N/A</td>'
        else:
            table_html += f'<td>{overall_rms:.2f}</td>'

        # Peak Frequency
        peak_freq = metrics['peak_freq']
        if peak_freq == 'N/A':
            table_html += '<td>N/A</td>'
        else:
            table_html += f'<td>{peak_freq:.2f}</td>'

        table_html += '</tr>'

    table_html += '</tbody></table>'

    return table_html


if __name__ == "__main__":
    # Test the comparison generator
    from case_manager import CaseManager

    manager = CaseManager()
    cases = manager.list_cases()

    if cases:
        patient_id = cases[0]['patient_id']
        trials = manager.get_trials(patient_id)

        if len(trials) >= 2:
            # Get first two trials for comparison
            trial_names = [trials[0]['trial_name'], trials[1]['trial_name']]
            comparison_data = manager.get_comparison_data(patient_id, trial_names)

            if comparison_data:
                output_path = generate_comparison_html(patient_id, comparison_data)
                print(f"\nTest comparison generated: {output_path}")
                print(f"Open in browser to view")
        else:
            print("Need at least 2 trials for comparison")
    else:
        print("No cases found")
