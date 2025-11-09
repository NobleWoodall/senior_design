"""
Session management for longitudinal tremor tracking across MRgFUS sonications.
"""

import os
import json
from typing import Dict, Any, Tuple
from pathlib import Path


def calculate_improvements(
    baseline_results: Dict[str, Any],
    current_results: Dict[str, Any],
    method: str = 'mp'
) -> Dict[str, Any]:
    """
    Calculate improvement metrics from baseline to current session.

    Args:
        baseline_results: Results dict from baseline session
        current_results: Results dict from current session
        method: Tracking method to compare ('mp', 'hsv', 'led')

    Returns:
        Dictionary with improvement metrics
    """
    improvements = {}

    # Extract signal analysis data
    baseline_signal = baseline_results.get('signal_analysis', {}).get(method, {})
    current_signal = current_results.get('signal_analysis', {}).get(method, {})

    if not baseline_signal or not current_signal:
        return {'error': f'No signal data for method {method}'}

    # Tremor power improvement
    baseline_power = baseline_signal.get('tremor_band_power', 0)
    current_power = current_signal.get('tremor_band_power', 0)

    if baseline_power > 0:
        power_reduction_pct = ((baseline_power - current_power) / baseline_power) * 100
    else:
        power_reduction_pct = 0

    improvements['tremor_power'] = {
        'baseline': baseline_power,
        'current': current_power,
        'change': current_power - baseline_power,
        'percent_reduction': power_reduction_pct
    }

    # RMS tremor improvement
    baseline_rms = baseline_signal.get('rms_tremor_band_px', 0)
    current_rms = current_signal.get('rms_tremor_band_px', 0)

    if baseline_rms > 0:
        rms_reduction_pct = ((baseline_rms - current_rms) / baseline_rms) * 100
    else:
        rms_reduction_pct = 0

    improvements['rms_tremor'] = {
        'baseline': baseline_rms,
        'current': current_rms,
        'change': current_rms - baseline_rms,
        'percent_reduction': rms_reduction_pct
    }

    # Overall RMS improvement
    baseline_rms_overall = baseline_signal.get('rms_px', 0)
    current_rms_overall = current_signal.get('rms_px', 0)

    if baseline_rms_overall > 0:
        rms_overall_reduction_pct = ((baseline_rms_overall - current_rms_overall) / baseline_rms_overall) * 100
    else:
        rms_overall_reduction_pct = 0

    improvements['rms_overall'] = {
        'baseline': baseline_rms_overall,
        'current': current_rms_overall,
        'change': current_rms_overall - baseline_rms_overall,
        'percent_reduction': rms_overall_reduction_pct
    }

    # Directional improvements (if available)
    baseline_directional = baseline_results.get('directional_analysis', {}).get(method, {})
    current_directional = current_results.get('directional_analysis', {}).get(method, {})

    if baseline_directional and current_directional:
        baseline_worst_power = baseline_directional.get('worst_power', 0)
        current_worst_power = current_directional.get('worst_power', 0)

        if baseline_worst_power > 0:
            worst_angle_reduction_pct = ((baseline_worst_power - current_worst_power) / baseline_worst_power) * 100
        else:
            worst_angle_reduction_pct = 0

        improvements['directional'] = {
            'baseline_worst_angle': baseline_directional.get('worst_angle', 0),
            'current_worst_angle': current_directional.get('worst_angle', 0),
            'baseline_worst_power': baseline_worst_power,
            'current_worst_power': current_worst_power,
            'worst_angle_reduction_pct': worst_angle_reduction_pct,
            'anisotropy_baseline': baseline_directional.get('anisotropy_ratio', 1),
            'anisotropy_current': current_directional.get('anisotropy_ratio', 1)
        }

    # Tracking accuracy improvements
    baseline_trial = baseline_results.get('trial_summaries', {}).get(method, {})
    current_trial = current_results.get('trial_summaries', {}).get(method, {})

    if baseline_trial and current_trial:
        baseline_rmse = baseline_trial.get('rmse_time_weighted', 0)
        current_rmse = current_trial.get('rmse_time_weighted', 0)

        if baseline_rmse > 0:
            rmse_reduction_pct = ((baseline_rmse - current_rmse) / baseline_rmse) * 100
        else:
            rmse_reduction_pct = 0

        improvements['tracking_accuracy'] = {
            'baseline_rmse': baseline_rmse,
            'current_rmse': current_rmse,
            'percent_improvement': rmse_reduction_pct
        }

    # Overall summary
    improvements['summary'] = {
        'primary_metric_reduction_pct': power_reduction_pct,
        'recommendation': _generate_recommendation(power_reduction_pct, current_power)
    }

    return improvements


def _generate_recommendation(reduction_pct: float, current_power: float) -> str:
    """Generate clinical recommendation based on improvement."""
    if reduction_pct >= 50:
        return "Excellent improvement - consider concluding"
    elif reduction_pct >= 30:
        return "Good improvement - may consider additional sonication"
    elif reduction_pct >= 10:
        return "Moderate improvement - consider additional sonication"
    elif reduction_pct >= 0:
        return "Minimal improvement - evaluate approach"
    else:
        return "No improvement detected - evaluate approach"
