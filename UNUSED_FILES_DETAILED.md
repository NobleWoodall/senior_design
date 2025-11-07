# Unused Files Analysis - Real Sense Tremor Tracking Codebase

## Overview
- Total Python files in `finger_tracing_refactor/src/`: 23
- Completely unused files: 3
- Qt/GUI only files: 3
- Imported but not called: 1

---

## COMPLETELY UNUSED FILES (3)

### 1. viz.py
**Path:** `c:\Users\14845\MyStuff\ClassFolders\Senior Design\real_sense\finger_tracing_refactor\src\viz.py`

**Status:** UNUSED - Never imported by any other module

**Contents:** Drawing utility functions
```python
- draw_status() - Draw text on frame
- draw_cross() - Draw crosshair marker
- draw_circle() - Draw circle on frame
```

**Why unused:** These are simple drawing utilities that aren't needed by the core workflow. The drawing functionality is incorporated directly in runner.py (e.g., `_draw_stereo_text()`, `_draw_stereo_circle()`)

---

### 2. comparison_display.py
**Path:** `c:\Users\14845\MyStuff\ClassFolders\Senior Design\real_sense\finger_tracing_refactor\src\comparison_display.py`

**Status:** UNUSED - Never imported by any other module

**Contents:** Multi-session comparison visualization
```python
def display_session_comparison(session_paths, output_path)
  - Display side-by-side comparison of baseline vs post-sonication
  - Create comparison_summary.png
```

**Why unused:** Appears to be experimental functionality for comparing multiple sessions. The HTML results viewer (html_results_viewer.py) handles result visualization instead.

---

### 3. track_hsv.py
**Path:** `c:\Users\14845\MyStuff\ClassFolders\Senior Design\real_sense\finger_tracing_refactor\src\track_hsv.py`

**Status:** UNUSED - Never imported by any other module

**Contains:** `HSVTracker` class - HSV-based color tracking

**Why unused:** The LEDTracker class in track_led.py replaces this functionality with a combined red HSV + brightness tracking approach that's specifically tuned for red LEDs. The track_led.py version is the one actually used in runner.py:
```python
# In runner.py (line 453):
hsv_tracker = LEDTracker(...)  # <- This is used
```

**Note:** track_hsv.py contains HSVTracker class, but track_led.py contains LEDTracker class (which is what's imported)

---

## QT/GUI APPLICATION FILES (3)

### These form a SEPARATE GUI application not used by main workflow

### 1. qt_main.py
**Path:** `c:\Users\14845\MyStuff\ClassFolders\Senior Design\real_sense\finger_tracing_refactor\src\qt_main.py`

**Status:** UNUSED in main workflow - Standalone GUI entry point

**Purpose:** Entry point for PyQt6-based results viewer application

**Usage:**
```bash
python -m finger_tracing_refactor.src.qt_main
python -m finger_tracing_refactor.src.qt_main --results path/to/results.json
```

**Imports:** qt_results_viewer

**Current workflow:** NOT called from main.py or runner.py. This is an optional alternative GUI for viewing results post-experiment.

---

### 2. qt_results_viewer.py
**Path:** `c:\Users\14845\MyStuff\ClassFolders\Senior Design\real_sense\finger_tracing_refactor\src\qt_results_viewer.py`

**Status:** UNUSED in main workflow - Only used by qt_main.py

**Purpose:** Main window class for PyQt6 results viewer

**Contains:**
```python
class ResultsViewer(QMainWindow)
  - Professional dark mode clinical dashboard
  - Loads and displays tremor assessment results
  - Interactive plots and metrics
```

**Imports:** qt_components

---

### 3. qt_components.py
**Path:** `c:\Users\14845\MyStuff\ClassFolders\Senior Design\real_sense\finger_tracing_refactor\src\qt_components.py`

**Status:** UNUSED in main workflow - Library for Qt GUI

**Purpose:** Reusable PyQt6 UI components

**Contains:**
```python
class MetricCard - Professional card for displaying metrics
class MatplotlibWidget - Matplotlib canvas embedded in Qt
class ChartCard - Card containing matplotlib chart
class HeaderCard - Header with patient info and key metrics
class AssessmentCard - Clinical assessment card with color coding
def create_separator() - Horizontal separator line
```

**Notes:** 
- Designed with 8px grid system and medical color palette
- Only imported by qt_results_viewer.py

---

## IMPORTED BUT NOT CALLED (1)

### results_display.py
**Path:** `c:\Users\14845\MyStuff\ClassFolders\Senior Design\real_sense\finger_tracing_refactor\src\results_display.py`

**Status:** IMPORTED but NOT CALLED

**Location:** Imported in runner.py (line 19)
```python
from .results_display import display_results
```

**But never executed:** No call to `display_results()` function found in runner.py

**Purpose:** Create matplotlib-based results visualization

**Contains:**
```python
def display_results(results_dict, overlay_path, output_dir)
  - Display consolidated results in professional clinical dashboard
  - Creates matplotlib visualization (not Qt)
  - Uses matplotlib instead of interactive HTML/Qt
```

**Note:** The HTML results viewer (html_results_viewer.py) is used instead for result visualization in the main workflow.

---

## MAIN WORKFLOW - ENTRY POINTS & DEPENDENCIES

### Primary Entry Point
```
main.py
  └─> ExperimentRunner(config) in runner.py
       ├─> ExperimentRunner.run_all()
       └─> Manages entire experiment flow
```

### Actual Used Imports in runner.py
```
config.py          - Configuration dataclasses
io_rs.py           - RealSense camera I/O
spiral_3d.py       - 3D spiral generation
track_mp.py        - MediaPipe hand tracking
track_led.py       - LED/HSV brightness tracking
depth_utils.py     - Depth processing utilities
metrics.py         - Error metric calculations
save.py            - CSV/JSON result saving
signal_processing.py - Tremor/FFT analysis
html_results_viewer.py - HTML report generation
calibration_utils.py - Calibration matrix operations
directional_analysis.py - Directional tremor analysis
session_manager.py - Session/baseline management
results_display.py - IMPORTED BUT NOT CALLED
```

### Secondary Entry Point
```
calibrate.py (Standalone camera calibration tool)
  ├─> config.py
  ├─> io_rs.py
  ├─> spiral_3d.py
  ├─> track_mp.py
  ├─> track_led.py
  └─> calibration_utils.py
```

### Optional GUI Entry Point (Separate)
```
qt_main.py (Alternative GUI viewer for results)
  └─> qt_results_viewer.py
       └─> qt_components.py
```

---

## RECOMMENDATION SUMMARY

### Safe to Delete (3 files)
1. **viz.py** - Drawing utilities duplicated in runner.py
2. **comparison_display.py** - Unused session comparison tool
3. **track_hsv.py** - Replaced by LEDTracker in track_led.py

### Keep (Qt Application)
Keep qt_main.py, qt_results_viewer.py, qt_components.py if you want to maintain the optional GUI viewer for post-experiment result analysis. They're completely self-contained and don't interfere with the main workflow.

### Consider Removing Dead Import (1)
- **results_display.py** - Either call `display_results()` if needed for matplotlib visualization, or remove the import from runner.py

---

## FILE SIZES & COMPLEXITY

Here's what each unused file contains:

| File | Size | Complexity | Type |
|------|------|-----------|------|
| viz.py | ~100 lines | Simple | Drawing utilities |
| comparison_display.py | ~150 lines | Moderate | Visualization |
| track_hsv.py | ~200 lines | Moderate | Tracking algorithm |
| qt_main.py | ~60 lines | Simple | App entry point |
| qt_results_viewer.py | ~500+ lines | Complex | Qt UI class |
| qt_components.py | ~320 lines | Complex | Qt component library |
| results_display.py | ~200+ lines | Moderate | Matplotlib visualization |

