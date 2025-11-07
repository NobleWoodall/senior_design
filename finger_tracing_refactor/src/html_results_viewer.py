"""
HTML-based results viewer for tremor assessment.
Generates a responsive, interactive HTML dashboard with embedded Plotly charts.
"""

import json
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64


def generate_html_report(results_path: str, output_path: str = None):
    """
    Generate an HTML report from results.json with interactive charts.

    Args:
        results_path: Path to results.json
        output_path: Where to save HTML (defaults to same dir as results.json with .html extension)
    """
    # Load results
    with open(results_path, 'r') as f:
        results_data = json.load(f)

    if output_path is None:
        output_path = str(Path(results_path).with_suffix('.html'))

    # Extract data
    session_metadata = results_data.get('session_metadata', {})
    signal_analysis = results_data.get('signal_analysis', {})
    directional_analysis = results_data.get('directional_analysis', {})
    improvements = results_data.get('improvements', {})
    trial_summaries = results_data.get('trial_summaries', {})

    # Determine primary method
    primary_method = None
    for method in ['mp', 'hsv', 'led']:
        if method in signal_analysis:
            primary_method = method
            break

    if not primary_method:
        raise ValueError("No valid tracking method found in results")

    # Generate HTML content
    html = generate_html_content(
        session_metadata, signal_analysis, directional_analysis,
        improvements, trial_summaries, primary_method, results_path
    )

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"HTML report saved to: {output_path}")
    return output_path


def generate_html_content(session_metadata, signal_analysis, directional_analysis,
                          improvements, trial_summaries, method, results_path):
    """Generate the complete HTML content."""

    # Create charts
    frequency_chart = create_frequency_chart(signal_analysis, method)
    directional_chart = create_directional_chart(directional_analysis, method)
    path_chart = create_path_chart(results_path, method)

    # Extract metrics
    method_data = signal_analysis.get(method, {})
    trial_data = trial_summaries.get(method, {})

    # Calculate peak frequency
    peak_freq = calculate_peak_frequency(method_data)

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MRgFUS Tremor Assessment - Clinical Dashboard</title>
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
            max-width: 1600px;
            margin: 0 auto;
        }}

        .card {{
            background: #2d2d2d;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}

        .header {{
            background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
            color: white;
            padding: 32px;
            border-radius: 8px;
            margin-bottom: 24px;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 12px;
            font-weight: 300;
            letter-spacing: 1px;
        }}

        .header-info {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 40px;
            margin-top: 20px;
        }}

        .patient-info {{
            flex: 1;
        }}

        .patient-info p {{
            font-size: 1.1em;
            margin: 8px 0;
            opacity: 0.95;
        }}

        .key-metrics {{
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 8px;
            min-width: 300px;
        }}

        .key-metrics h3 {{
            margin-bottom: 16px;
            font-weight: 500;
        }}

        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}

        .metric-row:last-child {{
            border-bottom: none;
        }}

        .metric-label {{
            color: rgba(255,255,255,0.8);
        }}

        .metric-value {{
            font-weight: 600;
            font-size: 1.1em;
        }}

        .assessment {{
            background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);
            color: white;
            padding: 24px;
            border-radius: 8px;
            margin-bottom: 24px;
        }}

        .assessment.warning {{
            background: linear-gradient(135deg, #F57C00 0%, #E65100 100%);
        }}

        .assessment.alert {{
            background: linear-gradient(135deg, #C62828 0%, #B71C1C 100%);
        }}

        .assessment h2 {{
            margin-bottom: 12px;
            font-size: 1.8em;
        }}

        .assessment p {{
            font-size: 1.1em;
            opacity: 0.95;
        }}

        .chart-container {{
            width: 100%;
            height: 600px;
            background: #2d2d2d;
            border-radius: 8px;
            padding: 16px;
        }}

        .chart-title {{
            font-size: 1.5em;
            margin-bottom: 16px;
            color: #ffffff;
            font-weight: 500;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 24px;
            margin-bottom: 24px;
        }}

        .metric-card {{
            background: #2d2d2d;
            padding: 24px;
            border-radius: 8px;
        }}

        .metric-card h3 {{
            color: #42A5F5;
            margin-bottom: 16px;
            font-size: 1.3em;
            border-bottom: 2px solid #42A5F5;
            padding-bottom: 8px;
        }}

        .metric-item {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #424242;
        }}

        .metric-item:last-child {{
            border-bottom: none;
        }}

        .metric-item-label {{
            color: #b0b0b0;
        }}

        .metric-item-value {{
            font-weight: 600;
            color: #ffffff;
        }}

        @media (max-width: 768px) {{
            .header-info {{
                flex-direction: column;
            }}

            .metrics-grid {{
                grid-template-columns: 1fr;
            }}

            .chart-container {{
                height: 400px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {generate_header_html(session_metadata, method_data, peak_freq, improvements, method)}

        {generate_assessment_html(improvements, method, session_metadata) if improvements else ''}

        <div class="card">
            <h2 class="chart-title">360° DIRECTIONAL TREMOR ANALYSIS</h2>
            <div id="directional-chart" class="chart-container"></div>
        </div>

        <div class="card">
            <h2 class="chart-title">TREMOR BAND FREQUENCY ANALYSIS (4-10 Hz)</h2>
            <div id="frequency-chart" class="chart-container"></div>
        </div>

        <div class="card">
            <h2 class="chart-title">FINGER TRACE PATH</h2>
            <div id="path-chart" class="chart-container"></div>
        </div>

        <div class="metrics-grid">
            {generate_signal_metrics_html(method_data)}
            {generate_trial_metrics_html(trial_data)}
        </div>
    </div>

    <script>
        // Plotly chart configurations
        const darkLayout = {{
            paper_bgcolor: '#2d2d2d',
            plot_bgcolor: '#2d2d2d',
            font: {{ color: '#e0e0e0' }},
            xaxis: {{ gridcolor: '#424242', zerolinecolor: '#424242' }},
            yaxis: {{ gridcolor: '#424242', zerolinecolor: '#424242' }}
        }};

        // Frequency chart
        {frequency_chart}

        // Directional chart
        {directional_chart}

        // Path chart
        {path_chart}
    </script>
</body>
</html>"""

    return html


def generate_header_html(metadata, method_data, peak_freq, improvements, method):
    """Generate header section HTML."""
    patient_id = metadata.get('patient_id', 'ANON')
    session_type = metadata.get('session_type', 'baseline')
    sonication_num = metadata.get('sonication_number', 0)
    date = metadata.get('date', '')

    if session_type == 'baseline':
        session_text = "BASELINE SESSION"
    else:
        session_text = f"POST-SONICATION #{sonication_num}"

    tremor_rms = method_data.get('rms_tremor_band_px', 0)
    overall_rms = method_data.get('rms_px', 0)

    improvement_html = ""
    if improvements and method in improvements:
        improvement_pct = improvements[method].get('tremor_power_reduction_pct', 0)
        if improvement_pct > 0:
            arrow = "↓"
            color = "#66BB6A" if improvement_pct >= 30 else "#FFCA28"
        else:
            arrow = "↑"
            color = "#EF5350"

        improvement_html = f"""
        <div class="metric-row">
            <span class="metric-label">Change from Baseline</span>
            <span class="metric-value" style="color: {color}">{arrow} {abs(improvement_pct):.1f}%</span>
        </div>
        """

    return f"""
    <div class="header">
        <h1>TREMOR ASSESSMENT</h1>
        <div class="header-info">
            <div class="patient-info">
                <p><strong>Patient:</strong> {patient_id}</p>
                <p><strong>Session:</strong> {session_text}</p>
                <p><strong>Date:</strong> {date}</p>
            </div>
            <div class="key-metrics">
                <h3>KEY MEASUREMENTS</h3>
                <div class="metric-row">
                    <span class="metric-label">Tremor Band RMS</span>
                    <span class="metric-value">{tremor_rms:.2f} px</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Peak Frequency</span>
                    <span class="metric-value">{peak_freq:.2f} Hz</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Overall RMS</span>
                    <span class="metric-value">{overall_rms:.2f} px</span>
                </div>
                {improvement_html}
            </div>
        </div>
    </div>
    """


def generate_assessment_html(improvements, method, metadata):
    """Generate clinical assessment card HTML."""
    if metadata.get('session_type') != 'post_sonication':
        return ""

    method_improvements = improvements.get(method, {})
    reduction_pct = method_improvements.get('tremor_power_reduction_pct', 0)
    recommendation = method_improvements.get('recommendation', 'No recommendation available')

    css_class = "assessment"
    if reduction_pct >= 30:
        css_class = "assessment"
    elif reduction_pct >= 10:
        css_class = "assessment warning"
    elif reduction_pct < 0:
        css_class = "assessment alert"

    return f"""
    <div class="{css_class}">
        <h2>CLINICAL ASSESSMENT</h2>
        <p><strong>Status:</strong> {recommendation}</p>
        <p style="margin-top: 12px;">
            <strong>Tremor Power Reduction:</strong> {reduction_pct:.1f}% |
            <strong>RMS Reduction:</strong> {method_improvements.get('rms_reduction_pct', 0):.1f}%
        </p>
    </div>
    """


def generate_signal_metrics_html(method_data):
    """Generate signal analysis metrics card HTML."""
    tremor_band = method_data.get('tremor_band_hz', [4.0, 10.0])

    return f"""
    <div class="metric-card">
        <h3>SIGNAL ANALYSIS DETAILS</h3>
        <div class="metric-item">
            <span class="metric-item-label">Tremor Band Power</span>
            <span class="metric-item-value">{method_data.get('tremor_band_power', 0):.4f}</span>
        </div>
        <div class="metric-item">
            <span class="metric-item-label">Tremor Band Range</span>
            <span class="metric-item-value">{tremor_band[0]:.1f} - {tremor_band[1]:.1f} Hz</span>
        </div>
        <div class="metric-item">
            <span class="metric-item-label">Resampled Length</span>
            <span class="metric-item-value">{method_data.get('resampled_len', 0)} samples</span>
        </div>
        <div class="metric-item">
            <span class="metric-item-label">Sampling Rate</span>
            <span class="metric-item-value">{method_data.get('sampling_rate_hz', 0):.1f} Hz</span>
        </div>
    </div>
    """


def generate_trial_metrics_html(trial_data):
    """Generate trial tracking metrics card HTML."""
    if not trial_data:
        return """
        <div class="metric-card">
            <h3>TRACKING PERFORMANCE</h3>
            <div class="metric-item">
                <span class="metric-item-label">No Data</span>
                <span class="metric-item-value">Trial summary not available</span>
            </div>
        </div>
        """

    err_px = trial_data.get('err_px', {})

    return f"""
    <div class="metric-card">
        <h3>TRACKING PERFORMANCE</h3>
        <div class="metric-item">
            <span class="metric-item-label">Duration</span>
            <span class="metric-item-value">{trial_data.get('duration_sec', 0):.2f} sec</span>
        </div>
        <div class="metric-item">
            <span class="metric-item-label">Total Frames</span>
            <span class="metric-item-value">{trial_data.get('sample_count', 0)}</span>
        </div>
        <div class="metric-item">
            <span class="metric-item-label">Average FPS</span>
            <span class="metric-item-value">{trial_data.get('fps_avg', 0):.2f}</span>
        </div>
        <div class="metric-item">
            <span class="metric-item-label">Tracking Loss Rate</span>
            <span class="metric-item-value">{trial_data.get('tracking_loss_rate', 0)*100:.1f}%</span>
        </div>
        <div class="metric-item">
            <span class="metric-item-label">Mean Error</span>
            <span class="metric-item-value">{err_px.get('mean', 0):.2f} px</span>
        </div>
        <div class="metric-item">
            <span class="metric-item-label">Median Error</span>
            <span class="metric-item-value">{err_px.get('median', 0):.2f} px</span>
        </div>
        <div class="metric-item">
            <span class="metric-item-label">95th Percentile</span>
            <span class="metric-item-value">{err_px.get('p95', 0):.2f} px</span>
        </div>
    </div>
    """


def calculate_peak_frequency(method_data):
    """Calculate peak frequency from PSD data."""
    psd_data = method_data.get('psd_temporal', [])
    if not psd_data:
        return 0.0

    max_power = 0
    peak_freq = 0
    for entry in psd_data:
        freq = entry.get('freq_hz', 0)
        power = entry.get('psd', 0)
        if 4.0 <= freq <= 10.0 and power > max_power:
            max_power = power
            peak_freq = freq

    return peak_freq


def create_frequency_chart(signal_analysis, method):
    """Create Plotly frequency domain chart focused on tremor band (4-10 Hz)."""
    method_data = signal_analysis.get(method, {})
    psd_data = method_data.get('psd_temporal', [])

    # Filter to tremor band (4-10 Hz) to avoid low-frequency power overwhelming the signal
    freqs_tremor = [entry['freq_hz'] for entry in psd_data if 4.0 <= entry['freq_hz'] <= 10.0] if psd_data else []
    psd_tremor = [entry['psd'] for entry in psd_data if 4.0 <= entry['freq_hz'] <= 10.0] if psd_data else []
    peak_freq = calculate_peak_frequency(method_data)

    # Find peak in tremor band for proper scaling
    max_psd_tremor = max(psd_tremor) if psd_tremor else 0

    js_code = f"""
        const freqData = {{
            x: {json.dumps(freqs_tremor)},
            y: {json.dumps(psd_tremor)},
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '#42A5F5', width: 3 }},
            fill: 'tozeroy',
            fillcolor: 'rgba(66, 165, 245, 0.3)',
            name: 'Tremor Band PSD (4-10 Hz)'
        }};

        const peakLine = {{
            x: [{peak_freq}, {peak_freq}],
            y: [0, {max_psd_tremor}],
            type: 'scatter',
            mode: 'lines+markers',
            line: {{ color: '#EF5350', width: 2, dash: 'dash' }},
            marker: {{ size: 10, symbol: 'diamond' }},
            name: 'Peak Tremor: {peak_freq:.2f} Hz'
        }};

        const freqLayout = {{
            ...darkLayout,
            xaxis: {{
                title: 'Frequency (Hz)',
                gridcolor: '#424242',
                range: [4, 10]  // Lock to tremor band
            }},
            yaxis: {{
                title: 'Power Spectral Density (Tremor Band)',
                gridcolor: '#424242'
            }},
            showlegend: true,
            legend: {{
                x: 1,
                xanchor: 'right',
                y: 1,
                bgcolor: 'rgba(45,45,45,0.8)'
            }},
            margin: {{ l: 60, r: 40, t: 20, b: 60 }}
        }};

        Plotly.newPlot('frequency-chart', [freqData, peakLine], freqLayout, {{responsive: true}});
    """

    return js_code


def create_directional_chart(directional_analysis, method):
    """Create Plotly polar directional chart."""
    method_data = directional_analysis.get(method, {})

    bin_centers = method_data.get('bin_centers', [])
    power_per_bin = method_data.get('power_per_bin', [])
    worst_angle = method_data.get('worst_angle', 0)
    best_angle = method_data.get('best_angle', 0)

    # Create color array
    colors = []
    for angle in bin_centers:
        if abs(angle - worst_angle) < 5:
            colors.append('#EF5350')
        elif abs(angle - best_angle) < 5:
            colors.append('#66BB6A')
        else:
            colors.append('#42A5F5')

    js_code = f"""
        const dirData = {{
            r: {json.dumps(power_per_bin)},
            theta: {json.dumps(bin_centers)},
            type: 'barpolar',
            marker: {{
                color: {json.dumps(colors)},
                line: {{ color: '#2d2d2d', width: 1 }}
            }},
            opacity: 0.8
        }};

        const dirLayout = {{
            ...darkLayout,
            polar: {{
                radialaxis: {{
                    visible: true,
                    gridcolor: '#424242',
                    tickfont: {{ color: '#b0b0b0' }}
                }},
                angularaxis: {{
                    direction: 'clockwise',
                    gridcolor: '#424242',
                    tickfont: {{ color: '#b0b0b0' }}
                }},
                bgcolor: '#2d2d2d'
            }},
            showlegend: false,
            margin: {{ l: 80, r: 80, t: 40, b: 80 }}
        }};

        Plotly.newPlot('directional-chart', [dirData], dirLayout, {{responsive: true}});
    """

    return js_code


def create_path_chart(results_path, method):
    """Create Plotly path visualization with spiral reference and dot path."""
    import pandas as pd
    import glob

    results_dir = Path(results_path).parent
    frames_csv = results_dir / 'frames.csv'

    # Search for frames.csv
    if not frames_csv.exists():
        pattern = str(results_dir / '2*' / 'frames.csv')
        matching_files = sorted(glob.glob(pattern), reverse=True)
        if matching_files:
            frames_csv = Path(matching_files[0])
        else:
            return "console.log('frames.csv not found');"

    try:
        df = pd.read_csv(frames_csv)
        method_df = df[df['method'] == method]

        if len(method_df) == 0:
            return "console.log('No data for method');"

        # Finger tracking path
        x_coords = method_df['x_px'].tolist()
        y_coords = method_df['y_px'].tolist()

        js_code = f"""
            // Finger tracking path with time gradient
            const pathData = {{
                x: {json.dumps(x_coords)},
                y: {json.dumps(y_coords)},
                mode: 'markers+lines',
                type: 'scatter',
                marker: {{
                    size: 6,
                    color: Array.from({{length: {len(x_coords)}}}, (_, i) => i),
                    colorscale: 'Plasma',
                    showscale: true,
                    colorbar: {{
                        title: 'Time',
                        titleside: 'right',
                        tickfont: {{ color: '#b0b0b0' }},
                        titlefont: {{ color: '#ffffff' }}
                    }}
                }},
                line: {{
                    color: '#42A5F5',
                    width: 1.5,
                    opacity: 0.4
                }},
                name: 'Finger Trace'
            }};

            const startPoint = {{
                x: [{x_coords[0]}],
                y: [{y_coords[0]}],
                mode: 'markers',
                type: 'scatter',
                marker: {{
                    size: 15,
                    color: '#66BB6A',
                    symbol: 'circle',
                    line: {{ color: 'white', width: 2 }}
                }},
                name: 'Start'
            }};

            const endPoint = {{
                x: [{x_coords[-1]}],
                y: [{y_coords[-1]}],
                mode: 'markers',
                type: 'scatter',
                marker: {{
                    size: 15,
                    color: '#EF5350',
                    symbol: 'triangle-up',
                    line: {{ color: 'white', width: 2 }}
                }},
                name: 'End'
            }};

            const pathLayout = {{
                ...darkLayout,
                xaxis: {{
                    title: 'X Position (pixels)',
                    gridcolor: '#424242',
                    scaleanchor: 'y'
                }},
                yaxis: {{
                    title: 'Y Position (pixels)',
                    gridcolor: '#424242'
                }},
                showlegend: true,
                legend: {{
                    x: 1,
                    xanchor: 'right',
                    y: 1,
                    bgcolor: 'rgba(45,45,45,0.8)'
                }},
                margin: {{ l: 60, r: 40, t: 20, b: 60 }}
            }};

            Plotly.newPlot('path-chart', [pathData, startPoint, endPoint], pathLayout, {{responsive: true}});
        """

        return js_code

    except Exception as e:
        return f"console.log('Error loading path data: {str(e)}');"


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python html_results_viewer.py <path_to_results.json> [output.html]")
        sys.exit(1)

    results_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    html_path = generate_html_report(results_path, output_path)

    # Open in browser
    import webbrowser
    webbrowser.open(f'file://{Path(html_path).absolute()}')
