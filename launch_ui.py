"""
Simple launcher for the Doctor UI application.

Usage:
    python launch_ui.py
"""

import sys
from pathlib import Path

# Add ui directory to path
ui_dir = Path(__file__).parent / "ui"
sys.path.insert(0, str(ui_dir))

from doctor_ui import DoctorUI

if __name__ == "__main__":
    print("=" * 60)
    print("  MRgFUS Tremor Assessment System - Doctor UI")
    print("=" * 60)
    print()

    app = DoctorUI()
    app.run()
