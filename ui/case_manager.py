"""
Case and trial management for MRgFUS tremor assessment system.

Manages patient cases, trial organization, and metadata indexing.
"""

import os
import json
import time
import shutil
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class CaseManager:
    """Manages patient cases and trials in the runs directory."""

    # Standard trial types and their default ordering
    TRIAL_TYPES = {
        "Baseline": 0,
        "Preop": 1,
        "Intraop 1": 2,
        "Intraop 2": 3,
        "Intraop 3": 4,
        "Postop": 5,
        "Custom": 99
    }

    def __init__(self, runs_dir: str = "runs"):
        """
        Initialize case manager.

        Args:
            runs_dir: Path to runs directory containing trial data
        """
        self.runs_dir = Path(runs_dir)
        self.case_index_path = self.runs_dir / "case_index.json"
        self._ensure_runs_dir()

    def _ensure_runs_dir(self):
        """Ensure runs directory exists."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _load_case_index(self) -> Dict:
        """Load case index from disk, creating if it doesn't exist."""
        if self.case_index_path.exists():
            try:
                with open(self.case_index_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load case index: {e}")
                return {"patients": {}}
        return {"patients": {}}

    def _save_case_index(self, index: Dict):
        """Save case index to disk."""
        try:
            with open(self.case_index_path, 'w') as f:
                json.dump(index, f, indent=2)
        except IOError as e:
            print(f"Error: Could not save case index: {e}")

    def _scan_runs_directory(self) -> Dict[str, List[Dict]]:
        """
        Scan runs directory for session folders and extract metadata.

        Returns:
            Dictionary mapping patient_id to list of trial metadata
        """
        patients = {}

        # Look for session directories (format: PatientName_TrialName)
        for item in self.runs_dir.iterdir():
            if not item.is_dir():
                continue
            
            # Skip special directories
            if item.name in ['__pycache__', '.git']:
                continue

            # Try to load results.json
            results_path = item / "results.json"
            if not results_path.exists():
                continue

            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)

                metadata = results.get("session_metadata", {})
                patient_id = metadata.get("patient_id", "ANON")
                trial_name = metadata.get("trial_name", "Unknown")
                trial_order = metadata.get("trial_order", 99)

                # Use directory creation time as timestamp
                import os
                timestamp = str(int(os.path.getctime(item)))

                trial_info = {
                    "trial_name": trial_name,
                    "trial_order": trial_order,
                    "session_path": str(item),
                    "timestamp": timestamp,
                    "date": metadata.get("date", ""),
                    "session_type": metadata.get("session_type", ""),
                    "clinician": metadata.get("clinician", "")
                }

                if patient_id not in patients:
                    patients[patient_id] = []
                patients[patient_id].append(trial_info)

            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load results from {item.name}: {e}")
                continue

        # Sort trials for each patient by trial_order, then timestamp
        for patient_id in patients:
            patients[patient_id].sort(key=lambda x: (x["trial_order"], x["timestamp"]))

        return patients

    def refresh_index(self):
        """Scan runs directory and rebuild case index."""
        patients = self._scan_runs_directory()

        index = {
            "patients": {},
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        for patient_id, trials in patients.items():
            if trials:
                # Get earliest trial date as case creation date
                created_date = trials[0].get("date", "")
                clinician = trials[0].get("clinician", "")

                index["patients"][patient_id] = {
                    "patient_id": patient_id,
                    "clinician": clinician,
                    "created_date": created_date,
                    "trials": trials,
                    "num_trials": len(trials)
                }

        self._save_case_index(index)
        return index

    def list_cases(self) -> List[Dict]:
        """
        Get list of all patient cases.

        Returns:
            List of patient case dictionaries with metadata
        """
        # Refresh index to get latest data
        index = self.refresh_index()

        cases = list(index["patients"].values())
        # Sort by created_date (most recent first)
        cases.sort(key=lambda x: x["created_date"], reverse=True)

        return cases

    def get_case(self, patient_id: str) -> Optional[Dict]:
        """
        Get case information for a specific patient.

        Args:
            patient_id: Patient identifier

        Returns:
            Case dictionary or None if not found
        """
        index = self.refresh_index()
        return index["patients"].get(patient_id)

    def get_trials(self, patient_id: str) -> List[Dict]:
        """
        Get all trials for a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            List of trial metadata dictionaries, sorted by trial_order
        """
        case = self.get_case(patient_id)
        if case:
            return case.get("trials", [])
        return []

    def get_trial_by_name(self, patient_id: str, trial_name: str) -> Optional[Dict]:
        """
        Get specific trial by name.

        Args:
            patient_id: Patient identifier
            trial_name: Name of trial to find

        Returns:
            Trial metadata dictionary or None if not found
        """
        trials = self.get_trials(patient_id)
        for trial in trials:
            if trial["trial_name"] == trial_name:
                return trial
        return None

    def get_next_trial_order(self, patient_id: str) -> int:
        """
        Get the next available trial order number for a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            Next trial order number
        """
        trials = self.get_trials(patient_id)
        if not trials:
            return 0
        return max(trial["trial_order"] for trial in trials) + 1

    def create_case_metadata(self, patient_id: str, clinician: str = "", notes: str = "") -> Dict:
        """
        Create metadata dictionary for a new case.

        This doesn't create the actual trial - it just prepares the metadata
        that can be used in config.yaml when launching the main program.

        Args:
            patient_id: Patient identifier
            clinician: Clinician name
            notes: Clinical notes

        Returns:
            Session metadata dictionary ready for config.yaml
        """
        return {
            "patient_id": patient_id,
            "date": time.strftime("%Y-%m-%d"),
            "clinician": clinician,
            "notes": notes,
            "trial_name": "Baseline",
            "trial_order": 0
        }

    def create_trial_metadata(self, patient_id: str, trial_type: str, custom_name: str = "") -> Dict:
        """
        Create metadata dictionary for a new trial.

        Args:
            patient_id: Patient identifier
            trial_type: Type of trial (from TRIAL_TYPES keys)
            custom_name: Custom name if trial_type is "Custom"

        Returns:
            Session metadata dictionary ready for config.yaml
        """
        # Get existing case info
        case = self.get_case(patient_id)
        clinician = case.get("clinician", "") if case else ""

        # Determine trial name and order
        if trial_type == "Custom" and custom_name:
            trial_name = custom_name
            trial_order = self.get_next_trial_order(patient_id)
        else:
            trial_name = trial_type
            trial_order = self.TRIAL_TYPES.get(trial_type, 99)

        return {
            "patient_id": patient_id,
            "date": time.strftime("%Y-%m-%d"),
            "clinician": clinician,
            "notes": "",
            "trial_name": trial_name,
            "trial_order": trial_order
        }

    def load_trial_results(self, trial_info: Dict) -> Optional[Dict]:
        """
        Load results.json for a specific trial.

        Args:
            trial_info: Trial metadata dictionary (from get_trials)

        Returns:
            Results dictionary or None if not found
        """
        session_path = Path(trial_info["session_path"])
        results_path = session_path / "results.json"

        if not results_path.exists():
            return None

        try:
            with open(results_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading results from {results_path}: {e}")
            return None

    def get_comparison_data(self, patient_id: str, trial_names: List[str]) -> List[Tuple[str, Dict]]:
        """
        Get results data for multiple trials for comparison.

        Args:
            patient_id: Patient identifier
            trial_names: List of trial names to compare

        Returns:
            List of (trial_name, results_dict) tuples
        """
        comparison_data = []

        for trial_name in trial_names:
            trial_info = self.get_trial_by_name(patient_id, trial_name)
            if trial_info:
                results = self.load_trial_results(trial_info)
                if results:
                    comparison_data.append((trial_name, results))

        return comparison_data

    def delete_trial(self, patient_id: str, trial_name: str) -> bool:
        """
        Delete a specific trial for a patient.

        Args:
            patient_id: Patient identifier
            trial_name: Name of trial to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        # Get trial info
        trial_info = self.get_trial_by_name(patient_id, trial_name)
        if not trial_info:
            print(f"Error: Trial '{trial_name}' not found for patient '{patient_id}'")
            return False

        session_path = Path(trial_info["session_path"])

        # Delete the trial directory
        try:
            if session_path.exists():
                shutil.rmtree(session_path)
                print(f"Deleted trial directory: {session_path}")

            # Refresh the index to reflect the deletion
            self.refresh_index()
            return True

        except Exception as e:
            print(f"Error deleting trial: {e}")
            return False

    def delete_case(self, patient_id: str) -> bool:
        """
        Delete entire patient case (all trials).

        Args:
            patient_id: Patient identifier

        Returns:
            True if deletion was successful, False otherwise
        """
        # Get all trials for the patient
        trials = self.get_trials(patient_id)
        if not trials:
            print(f"Error: No trials found for patient '{patient_id}'")
            return False

        try:
            # Delete all trial directories
            for trial in trials:
                session_path = Path(trial["session_path"])
                if session_path.exists():
                    shutil.rmtree(session_path)
                    print(f"Deleted trial directory: {session_path}")

            # Delete any comparison HTML files for this patient
            comparison_pattern = f"comparison_{patient_id}_*.html"
            for comparison_file in self.runs_dir.glob(comparison_pattern):
                comparison_file.unlink()
                print(f"Deleted comparison file: {comparison_file}")

            # Refresh the index to reflect the deletions
            self.refresh_index()
            return True

        except Exception as e:
            print(f"Error deleting case: {e}")
            return False


if __name__ == "__main__":
    # Test the case manager
    manager = CaseManager()

    print("=== Refreshing case index ===")
    manager.refresh_index()

    print("\n=== All cases ===")
    cases = manager.list_cases()
    for case in cases:
        print(f"\nPatient: {case['patient_id']}")
        print(f"  Clinician: {case['clinician']}")
        print(f"  Created: {case['created_date']}")
        print(f"  Trials: {case['num_trials']}")

        for trial in case['trials']:
            print(f"    - {trial['trial_name']} (Order: {trial['trial_order']}, Date: {trial['date']})")
