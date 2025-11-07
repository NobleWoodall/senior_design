"""
Session management for longitudinal tremor tracking across MRgFUS sonications.
"""

import os
import json
import glob
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import yaml


def find_sessions(
    runs_dir: str,
    patient_id: Optional[str] = None,
    session_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Find all sessions in runs directory, optionally filtered by patient and type.

    Args:
        runs_dir: Path to runs directory
        patient_id: Filter by patient ID (None = all patients)
        session_type: Filter by session type (None = all types)

    Returns:
        List of session info dicts, sorted by timestamp (oldest first)
    """
    sessions = []

    # Find all session directories (timestamp-based naming)
    session_dirs = glob.glob(os.path.join(runs_dir, "202*"))

    for session_dir in session_dirs:
        # Check if valid session (has results.json)
        results_path = os.path.join(session_dir, 'results.json')
        config_path = os.path.join(session_dir, 'config.yaml')

        if not os.path.exists(results_path):
            continue

        # Load session metadata
        metadata = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                metadata = config_data.get('session_metadata', {})

        # Also check results.json for metadata (newer format)
        try:
            with open(results_path, 'r') as f:
                results_data = json.load(f)
                if 'session_metadata' in results_data:
                    metadata.update(results_data['session_metadata'])
        except:
            pass

        # Apply filters
        if patient_id and metadata.get('patient_id') != patient_id:
            continue

        if session_type and metadata.get('session_type') != session_type:
            continue

        # Extract session info
        session_info = {
            'path': session_dir,
            'name': os.path.basename(session_dir),
            'timestamp': os.path.basename(session_dir).split('_')[0] + '_' + os.path.basename(session_dir).split('_')[1],
            'patient_id': metadata.get('patient_id', 'ANON'),
            'session_type': metadata.get('session_type', 'unknown'),
            'sonication_number': metadata.get('sonication_number', 0),
            'date': metadata.get('date', ''),
            'clinician': metadata.get('clinician', ''),
            'notes': metadata.get('notes', '')
        }

        sessions.append(session_info)

    # Sort by timestamp
    sessions.sort(key=lambda x: x['timestamp'])

    return sessions


def load_session_results(session_path: str) -> Dict[str, Any]:
    """
    Load complete results for a session.

    Args:
        session_path: Path to session directory

    Returns:
        Dictionary with all session data
    """
    results_path = os.path.join(session_path, 'results.json')

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No results.json in {session_path}")

    with open(results_path, 'r') as f:
        results = json.load(f)

    return results


def get_baseline_for_patient(
    runs_dir: str,
    patient_id: str
) -> Optional[Dict[str, Any]]:
    """
    Find the baseline session for a given patient.

    Args:
        runs_dir: Path to runs directory
        patient_id: Patient ID

    Returns:
        Session info dict for baseline, or None if not found
    """
    baselines = find_sessions(runs_dir, patient_id=patient_id, session_type='baseline')

    if not baselines:
        return None

    # Return most recent baseline
    return baselines[-1]


def get_sonication_sequence(
    runs_dir: str,
    patient_id: str,
    include_baseline: bool = True
) -> List[Dict[str, Any]]:
    """
    Get all sessions for a patient in chronological order.

    Args:
        runs_dir: Path to runs directory
        patient_id: Patient ID
        include_baseline: Whether to include baseline session

    Returns:
        List of session info dicts in order (baseline, sonica 1, sonica 2, ...)
    """
    all_sessions = find_sessions(runs_dir, patient_id=patient_id)

    if not include_baseline:
        all_sessions = [s for s in all_sessions if s['session_type'] != 'baseline']

    # Sort by sonication number (baseline = 0)
    all_sessions.sort(key=lambda x: x['sonication_number'])

    return all_sessions


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


def format_session_summary(session_info: Dict[str, Any]) -> str:
    """
    Format session info as readable string.

    Args:
        session_info: Session info dict from find_sessions()

    Returns:
        Formatted string
    """
    summary = f"""Session: {session_info['name']}
  Patient: {session_info['patient_id']}
  Type: {session_info['session_type']}
  Sonication: #{session_info['sonication_number']}
  Date: {session_info['date']}
  Clinician: {session_info['clinician']}
  Notes: {session_info['notes']}
"""
    return summary


if __name__ == '__main__':
    # Test session management
    runs_dir = r"c:\Users\14845\MyStuff\ClassFolders\Senior Design\real_sense\runs"

    print("Finding all sessions...")
    all_sessions = find_sessions(runs_dir)

    print(f"\nFound {len(all_sessions)} sessions:")
    for session in all_sessions[:5]:  # Show first 5
        print(f"  - {session['name']}: {session['patient_id']} ({session['session_type']})")

    if all_sessions:
        print("\nFirst session details:")
        print(format_session_summary(all_sessions[0]))
