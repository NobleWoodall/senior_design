"""
Doctor UI for MRgFUS Tremor Assessment System.

Desktop application for managing patient cases, trials, and viewing comparison results.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import webbrowser
import subprocess
import os
import sys
import yaml
import tempfile
from pathlib import Path

# Add parent directory to path to import case_manager
sys.path.insert(0, str(Path(__file__).parent))

from case_manager import CaseManager
from comparison_generator import generate_comparison_html


class DoctorUI:
    """Main UI application for doctors to manage tremor assessment cases."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MRgFUS Tremor Assessment System")
        self.root.geometry("900x700")
        self.root.configure(bg='#2d2d2d')

        # Initialize case manager
        self.case_manager = CaseManager()

        # State
        self.current_patient_id = None
        self.selected_trials = []

        # Set style
        self._setup_style()

        # Show case selection screen
        self.show_case_selection()

    def _setup_style(self):
        """Setup ttk styles for a modern dark theme."""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure colors
        bg_color = '#2d2d2d'
        fg_color = '#e0e0e0'
        select_bg = '#667eea'
        select_fg = '#ffffff'

        style.configure('TFrame', background=bg_color)
        style.configure('TLabel', background=bg_color, foreground=fg_color, font=('Segoe UI', 10))
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'), foreground='#667eea')
        style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'), foreground='#667eea')

        style.configure('TButton',
                       background='#667eea',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 10))
        style.map('TButton',
                 background=[('active', '#5568d3')])

        style.configure('Accent.TButton',
                       background='#4ade80',
                       foreground='white')
        style.map('Accent.TButton',
                 background=[('active', '#3bc46e')])

        style.configure('Danger.TButton',
                       background='#f87171',
                       foreground='white')
        style.map('Danger.TButton',
                 background=[('active', '#e65c5c')])

        # Treeview style
        style.configure('Treeview',
                       background='#1f1f1f',
                       foreground=fg_color,
                       fieldbackground='#1f1f1f',
                       borderwidth=0)
        style.configure('Treeview.Heading',
                       background='#667eea',
                       foreground='white',
                       borderwidth=0,
                       font=('Segoe UI', 10, 'bold'))
        style.map('Treeview',
                 background=[('selected', select_bg)],
                 foreground=[('selected', select_fg)])

    def _clear_window(self):
        """Clear all widgets from the window."""
        for widget in self.root.winfo_children():
            widget.destroy()

    def show_case_selection(self):
        """Show case selection screen."""
        self._clear_window()

        # Main container
        container = ttk.Frame(self.root, padding=20)
        container.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(container, text="MRgFUS Tremor Assessment System", style='Title.TLabel')
        title_label.pack(pady=(0, 20))

        # Cases section
        cases_frame = ttk.Frame(container)
        cases_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(cases_frame, text="Patient Cases", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 10))

        # Cases tree
        tree_frame = ttk.Frame(cases_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Tree view
        self.cases_tree = ttk.Treeview(tree_frame,
                                       columns=('patient_id', 'clinician', 'created', 'num_trials'),
                                       show='headings',
                                       yscrollcommand=scrollbar.set,
                                       height=15)
        scrollbar.config(command=self.cases_tree.yview)

        self.cases_tree.heading('patient_id', text='Patient ID')
        self.cases_tree.heading('clinician', text='Clinician')
        self.cases_tree.heading('created', text='Created Date')
        self.cases_tree.heading('num_trials', text='Trials')

        self.cases_tree.column('patient_id', width=200)
        self.cases_tree.column('clinician', width=200)
        self.cases_tree.column('created', width=150)
        self.cases_tree.column('num_trials', width=100)

        self.cases_tree.pack(fill=tk.BOTH, expand=True)

        # Load cases
        self._load_cases()

        # Buttons
        buttons_frame = ttk.Frame(container)
        buttons_frame.pack(pady=(20, 0))

        ttk.Button(buttons_frame, text="New Case", command=self.new_case, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Load Case", command=self.load_case).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Delete Case", command=self.delete_selected_case, style='Danger.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Settings", command=self.show_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Refresh", command=self._load_cases).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Exit", command=self.root.quit, style='Danger.TButton').pack(side=tk.LEFT, padx=5)

    def _load_cases(self):
        """Load and display all cases."""
        # Clear existing items
        for item in self.cases_tree.get_children():
            self.cases_tree.delete(item)

        # Load cases from case manager
        cases = self.case_manager.list_cases()

        for case in cases:
            self.cases_tree.insert('', tk.END, values=(
                case['patient_id'],
                case['clinician'] or 'Unknown',
                case['created_date'],
                case['num_trials']
            ))

    def new_case(self):
        """Create a new patient case."""
        # Dialog for patient info
        dialog = tk.Toplevel(self.root)
        dialog.title("New Patient Case")
        dialog.geometry("400x250")
        dialog.configure(bg='#2d2d2d')
        dialog.transient(self.root)
        dialog.grab_set()

        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        # Form
        form_frame = ttk.Frame(dialog, padding=20)
        form_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(form_frame, text="Patient ID:").grid(row=0, column=0, sticky=tk.W, pady=5)
        patient_id_entry = ttk.Entry(form_frame, width=30)
        patient_id_entry.grid(row=0, column=1, pady=5)

        ttk.Label(form_frame, text="Clinician:").grid(row=1, column=0, sticky=tk.W, pady=5)
        clinician_entry = ttk.Entry(form_frame, width=30)
        clinician_entry.grid(row=1, column=1, pady=5)

        ttk.Label(form_frame, text="Notes:").grid(row=2, column=0, sticky=tk.W, pady=5)
        notes_text = tk.Text(form_frame, width=30, height=4, bg='#1f1f1f', fg='#e0e0e0', insertbackground='white')
        notes_text.grid(row=2, column=1, pady=5)

        def create():
            patient_id = patient_id_entry.get().strip()
            clinician = clinician_entry.get().strip()
            notes = notes_text.get('1.0', tk.END).strip()

            if not patient_id:
                messagebox.showerror("Error", "Patient ID is required", parent=dialog)
                return

            # Check if patient already exists
            existing_case = self.case_manager.get_case(patient_id)
            if existing_case:
                messagebox.showerror("Error", f"Patient {patient_id} already exists", parent=dialog)
                return

            dialog.destroy()

            # Launch baseline trial
            metadata = self.case_manager.create_case_metadata(patient_id, clinician, notes)
            self._launch_trial(metadata)

        buttons_frame = ttk.Frame(form_frame)
        buttons_frame.grid(row=3, column=0, columnspan=2, pady=(15, 0))

        ttk.Button(buttons_frame, text="Create & Run Baseline", command=create, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def load_case(self):
        """Load selected case and show trial management screen."""
        selection = self.cases_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a case to load")
            return

        item = self.cases_tree.item(selection[0])
        patient_id = item['values'][0]

        self.current_patient_id = patient_id
        self.show_trial_management()

    def show_trial_management(self):
        """Show trial management screen for current patient."""
        if not self.current_patient_id:
            return

        self._clear_window()

        # Main container
        container = ttk.Frame(self.root, padding=20)
        container.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = ttk.Frame(container)
        header_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(header_frame, text=f"Case: {self.current_patient_id}", style='Title.TLabel').pack(side=tk.LEFT)
        ttk.Button(header_frame, text="← Back to Cases", command=self.show_case_selection).pack(side=tk.RIGHT)

        # Trials section
        trials_frame = ttk.Frame(container)
        trials_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(trials_frame, text="Trials", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 10))

        # Trials tree with checkboxes
        tree_frame = ttk.Frame(trials_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.trials_tree = ttk.Treeview(tree_frame,
                                       columns=('select', 'trial_name', 'date', 'status'),
                                       show='headings',
                                       yscrollcommand=scrollbar.set,
                                       height=12)
        scrollbar.config(command=self.trials_tree.yview)

        self.trials_tree.heading('select', text='☐')
        self.trials_tree.heading('trial_name', text='Trial Name')
        self.trials_tree.heading('date', text='Date')
        self.trials_tree.heading('status', text='Status')

        self.trials_tree.column('select', width=50)
        self.trials_tree.column('trial_name', width=200)
        self.trials_tree.column('date', width=150)
        self.trials_tree.column('status', width=150)

        self.trials_tree.bind('<Double-Button-1>', self._on_trial_double_click)
        self.trials_tree.bind('<Button-1>', self._on_trial_click)

        self.trials_tree.pack(fill=tk.BOTH, expand=True)

        # Load trials
        self._load_trials()

        # Action buttons
        actions_frame = ttk.Frame(container)
        actions_frame.pack(pady=(20, 0))

        ttk.Button(actions_frame, text="Add Trial", command=self.add_trial, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="View Selected", command=self.view_selected_trial).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="Compare Selected", command=self.compare_trials).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="Delete Trial", command=self.delete_selected_trial, style='Danger.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="Refresh", command=self._load_trials).pack(side=tk.LEFT, padx=5)

    def _load_trials(self):
        """Load and display all trials for current patient."""
        # Clear existing items
        for item in self.trials_tree.get_children():
            self.trials_tree.delete(item)

        self.selected_trials = []

        # Load trials from case manager
        trials = self.case_manager.get_trials(self.current_patient_id)

        for trial in trials:
            status = "Completed" if trial.get('date') else "Pending"
            self.trials_tree.insert('', tk.END, values=(
                '☐',
                trial['trial_name'],
                trial.get('date', ''),
                status
            ), tags=(trial['trial_name'],))

    def _on_trial_click(self, event):
        """Handle trial selection checkbox toggle."""
        region = self.trials_tree.identify_region(event.x, event.y)
        if region != 'cell':
            return

        column = self.trials_tree.identify_column(event.x)
        if column != '#1':  # First column (checkbox)
            return

        item = self.trials_tree.identify_row(event.y)
        if not item:
            return

        # Toggle checkbox
        current_values = list(self.trials_tree.item(item, 'values'))
        trial_name = current_values[1]

        if current_values[0] == '☐':
            current_values[0] = '☑'
            self.selected_trials.append(trial_name)
        else:
            current_values[0] = '☐'
            if trial_name in self.selected_trials:
                self.selected_trials.remove(trial_name)

        self.trials_tree.item(item, values=current_values)

    def _on_trial_double_click(self, event):
        """Handle double-click to view trial results."""
        column = self.trials_tree.identify_column(event.x)
        if column == '#1':  # Checkbox column
            return

        self.view_selected_trial()

    def add_trial(self):
        """Add a new trial for current patient."""
        # Dialog for trial type
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New Trial")
        dialog.geometry("400x300")
        dialog.configure(bg='#2d2d2d')
        dialog.transient(self.root)
        dialog.grab_set()

        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        form_frame = ttk.Frame(dialog, padding=20)
        form_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(form_frame, text="Select Trial Type:", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 10))

        trial_type_var = tk.StringVar(value="Preop")

        trial_types = list(self.case_manager.TRIAL_TYPES.keys())
        for trial_type in trial_types:
            ttk.Radiobutton(form_frame, text=trial_type, variable=trial_type_var, value=trial_type).pack(anchor=tk.W, pady=2)

        # Custom name entry (shown when Custom is selected)
        custom_frame = ttk.Frame(form_frame)
        custom_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(custom_frame, text="Custom Name:").pack(side=tk.LEFT)
        custom_name_entry = ttk.Entry(custom_frame, width=20)
        custom_name_entry.pack(side=tk.LEFT, padx=(10, 0))

        def create():
            trial_type = trial_type_var.get()
            custom_name = custom_name_entry.get().strip()

            if trial_type == "Custom" and not custom_name:
                messagebox.showerror("Error", "Please enter a custom name", parent=dialog)
                return

            dialog.destroy()

            # Create trial metadata
            metadata = self.case_manager.create_trial_metadata(
                self.current_patient_id,
                trial_type,
                custom_name
            )

            # Launch trial
            self._launch_trial(metadata)

        buttons_frame = ttk.Frame(form_frame)
        buttons_frame.pack(pady=(20, 0))

        ttk.Button(buttons_frame, text="Run Trial", command=create, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def _launch_trial(self, metadata: dict):
        """
        Launch main program with specified metadata.

        Args:
            metadata: Session metadata dictionary
        """
        # Load base config
        config_path = Path("config.yaml")
        if not config_path.exists():
            messagebox.showerror("Error", "config.yaml not found in project root")
            return

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Update session metadata
        config['session_metadata'] = metadata

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_config_path = f.name

        try:
            # Launch main program
            print(f"\nLaunching trial: {metadata['trial_name']} for patient {metadata['patient_id']}")
            print(f"Config: {temp_config_path}\n")

            result = subprocess.run([
                sys.executable, '-m', 'finger_tracing_refactor.src.main',
                '--config', temp_config_path
            ], cwd=os.getcwd())

            if result.returncode == 0:
                messagebox.showinfo("Success", "Trial completed successfully!")
                # Refresh trials list
                if self.current_patient_id:
                    self._load_trials()
            else:
                messagebox.showerror("Error", f"Trial failed with exit code {result.returncode}")

        finally:
            # Clean up temp config
            try:
                os.unlink(temp_config_path)
            except:
                pass

    def view_selected_trial(self):
        """View results for selected trial."""
        selection = self.trials_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a trial to view")
            return

        item = self.trials_tree.item(selection[0])
        trial_name = item['values'][1]

        # Get trial info
        trial_info = self.case_manager.get_trial_by_name(self.current_patient_id, trial_name)
        if not trial_info:
            messagebox.showerror("Error", "Trial not found")
            return

        # Open results HTML
        session_path = Path(trial_info['session_path'])
        results_html = session_path / "results.html"

        if not results_html.exists():
            messagebox.showerror("Error", f"Results not found at {results_html}")
            return

        webbrowser.open(str(results_html.absolute()))

    def compare_trials(self):
        """Generate and view comparison for selected trials."""
        if len(self.selected_trials) < 2:
            messagebox.showwarning("Insufficient Selection", "Please select at least 2 trials to compare")
            return

        # Get comparison data
        comparison_data = self.case_manager.get_comparison_data(
            self.current_patient_id,
            self.selected_trials
        )

        if not comparison_data:
            messagebox.showerror("Error", "Could not load trial data for comparison")
            return

        try:
            # Generate comparison HTML
            output_path = generate_comparison_html(
                self.current_patient_id,
                comparison_data
            )

            # Open in browser
            webbrowser.open(str(Path(output_path).absolute()))

            messagebox.showinfo("Success", f"Comparison report generated!\n\nComparing:\n" + "\n".join(f"  • {name}" for name in self.selected_trials))

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate comparison:\n{str(e)}")

    def delete_selected_trial(self):
        """Delete the selected trial."""
        selection = self.trials_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a trial to delete")
            return

        item = self.trials_tree.item(selection[0])
        trial_name = item['values'][1]

        # Confirm deletion
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete the trial '{trial_name}'?\n\n"
            f"This will permanently delete all data for this trial.\n"
            f"This action cannot be undone.",
            icon='warning'
        )

        if not confirm:
            return

        # Delete the trial
        success = self.case_manager.delete_trial(self.current_patient_id, trial_name)

        if success:
            messagebox.showinfo("Success", f"Trial '{trial_name}' has been deleted")
            # Refresh trials list
            self._load_trials()
        else:
            messagebox.showerror("Error", f"Failed to delete trial '{trial_name}'")

    def delete_selected_case(self):
        """Delete the selected case (all trials for a patient)."""
        selection = self.cases_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a case to delete")
            return

        item = self.cases_tree.item(selection[0])
        patient_id = item['values'][0]
        num_trials = item['values'][3]

        # Confirm deletion
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete the entire case for patient '{patient_id}'?\n\n"
            f"This will permanently delete ALL {num_trials} trial(s) and all associated data.\n"
            f"This action cannot be undone.",
            icon='warning'
        )

        if not confirm:
            return

        # Delete the case
        success = self.case_manager.delete_case(patient_id)

        if success:
            messagebox.showinfo("Success", f"Case for patient '{patient_id}' has been deleted")
            # Refresh cases list
            self._load_cases()
        else:
            messagebox.showerror("Error", f"Failed to delete case for patient '{patient_id}'")

    def show_settings(self):
        """Show settings dialog for configuring system parameters."""
        # Load current config
        config_path = Path("config.yaml")
        if not config_path.exists():
            messagebox.showerror("Error", "config.yaml not found")
            return

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Create settings dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("System Settings")
        dialog.geometry("600x700")
        dialog.configure(bg='#2d2d2d')
        dialog.transient(self.root)
        dialog.grab_set()

        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        # Main container with scrollbar
        main_frame = ttk.Frame(dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas and scrollbar
        canvas = tk.Canvas(main_frame, bg='#2d2d2d', highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Title
        ttk.Label(scrollable_frame, text="System Settings", style='Title.TLabel').pack(pady=(0, 20))

        # Tracking Method Section
        tracking_frame = ttk.LabelFrame(scrollable_frame, text="Tracking Method", padding=15)
        tracking_frame.pack(fill=tk.X, pady=10)

        tracking_var = tk.StringVar(value=config.get('experiment', {}).get('methods_order', ['mp'])[0])
        ttk.Radiobutton(tracking_frame, text="MediaPipe (Hand Tracking)", variable=tracking_var, value='mp').pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(tracking_frame, text="HSV (LED Tracking)", variable=tracking_var, value='hsv').pack(anchor=tk.W, pady=2)

        # LED Color Settings (for HSV tracking)
        led_frame = ttk.LabelFrame(scrollable_frame, text="LED Tracking Settings", padding=15)
        led_frame.pack(fill=tk.X, pady=10)

        ttk.Label(led_frame, text="HSV Low (H, S, V):").grid(row=0, column=0, sticky=tk.W, pady=5)
        hsv_low = config.get('led', {}).get('hsv_low', [4, 0, 227])
        hsv_low_h = ttk.Entry(led_frame, width=10)
        hsv_low_h.insert(0, str(hsv_low[0]))
        hsv_low_h.grid(row=0, column=1, padx=2)
        hsv_low_s = ttk.Entry(led_frame, width=10)
        hsv_low_s.insert(0, str(hsv_low[1]))
        hsv_low_s.grid(row=0, column=2, padx=2)
        hsv_low_v = ttk.Entry(led_frame, width=10)
        hsv_low_v.insert(0, str(hsv_low[2]))
        hsv_low_v.grid(row=0, column=3, padx=2)

        ttk.Label(led_frame, text="HSV High (H, S, V):").grid(row=1, column=0, sticky=tk.W, pady=5)
        hsv_high = config.get('led', {}).get('hsv_high', [35, 255, 255])
        hsv_high_h = ttk.Entry(led_frame, width=10)
        hsv_high_h.insert(0, str(hsv_high[0]))
        hsv_high_h.grid(row=1, column=1, padx=2)
        hsv_high_s = ttk.Entry(led_frame, width=10)
        hsv_high_s.insert(0, str(hsv_high[1]))
        hsv_high_s.grid(row=1, column=2, padx=2)
        hsv_high_v = ttk.Entry(led_frame, width=10)
        hsv_high_v.insert(0, str(hsv_high[2]))
        hsv_high_v.grid(row=1, column=3, padx=2)

        ttk.Label(led_frame, text="Brightness Threshold:").grid(row=2, column=0, sticky=tk.W, pady=5)
        brightness_entry = ttk.Entry(led_frame, width=10)
        brightness_entry.insert(0, str(config.get('led', {}).get('brightness_threshold', 221)))
        brightness_entry.grid(row=2, column=1, padx=2)

        # Spiral Settings
        spiral_frame = ttk.LabelFrame(scrollable_frame, text="Spiral Settings", padding=15)
        spiral_frame.pack(fill=tk.X, pady=10)

        ttk.Label(spiral_frame, text="Parameter A:").grid(row=0, column=0, sticky=tk.W, pady=5)
        spiral_a_entry = ttk.Entry(spiral_frame, width=15)
        spiral_a_entry.insert(0, str(config.get('spiral', {}).get('a', 30.0)))
        spiral_a_entry.grid(row=0, column=1, padx=5)

        ttk.Label(spiral_frame, text="Parameter B:").grid(row=1, column=0, sticky=tk.W, pady=5)
        spiral_b_entry = ttk.Entry(spiral_frame, width=15)
        spiral_b_entry.insert(0, str(config.get('spiral', {}).get('b', 35.0)))
        spiral_b_entry.grid(row=1, column=1, padx=5)

        ttk.Label(spiral_frame, text="Number of Turns:").grid(row=2, column=0, sticky=tk.W, pady=5)
        spiral_turns_entry = ttk.Entry(spiral_frame, width=15)
        spiral_turns_entry.insert(0, str(config.get('spiral', {}).get('turns', 2)))
        spiral_turns_entry.grid(row=2, column=1, padx=5)

        ttk.Label(spiral_frame, text="Line Thickness:").grid(row=3, column=0, sticky=tk.W, pady=5)
        spiral_thickness_entry = ttk.Entry(spiral_frame, width=15)
        spiral_thickness_entry.insert(0, str(config.get('spiral', {}).get('line_thickness', 30)))
        spiral_thickness_entry.grid(row=3, column=1, padx=5)

        # Dot Follow Settings
        dot_frame = ttk.LabelFrame(scrollable_frame, text="Dot Follow Settings", padding=15)
        dot_frame.pack(fill=tk.X, pady=10)

        ttk.Label(dot_frame, text="Dot Speed (sec/spiral):").grid(row=0, column=0, sticky=tk.W, pady=5)
        dot_speed_entry = ttk.Entry(dot_frame, width=15)
        dot_speed_entry.insert(0, str(config.get('dot_follow', {}).get('dot_speed_sec_per_spiral', 20.0)))
        dot_speed_entry.grid(row=0, column=1, padx=5)

        ttk.Label(dot_frame, text="Countdown (sec):").grid(row=1, column=0, sticky=tk.W, pady=5)
        countdown_entry = ttk.Entry(dot_frame, width=15)
        countdown_entry.insert(0, str(config.get('dot_follow', {}).get('countdown_sec', 3)))
        countdown_entry.grid(row=1, column=1, padx=5)

        # Stereo 3D Settings
        stereo_frame = ttk.LabelFrame(scrollable_frame, text="Stereo 3D Settings", padding=15)
        stereo_frame.pack(fill=tk.X, pady=10)

        ttk.Label(stereo_frame, text="Disparity Offset (px):").grid(row=0, column=0, sticky=tk.W, pady=5)
        disparity_offset_entry = ttk.Entry(stereo_frame, width=15)
        disparity_offset_entry.insert(0, str(config.get('stereo_3d', {}).get('disparity_offset_px', -500)))
        disparity_offset_entry.grid(row=0, column=1, padx=5)

        ttk.Label(stereo_frame, text="(Negative = closer, Positive = farther)", font=('Segoe UI', 8)).grid(row=0, column=2, sticky=tk.W, padx=5)

        # Calibration Section
        calibration_frame = ttk.LabelFrame(scrollable_frame, text="Calibration", padding=15)
        calibration_frame.pack(fill=tk.X, pady=10)

        ttk.Label(calibration_frame, text="Run calibration to align XReal display with RealSense camera").pack(anchor=tk.W, pady=5)

        def run_calibration():
            confirm = messagebox.askyesno(
                "Run Calibration",
                "This will launch the calibration process.\n\nMake sure:\n• RealSense camera is connected\n• XReal glasses are connected\n• You're ready to trace the spiral\n\nContinue?",
                parent=dialog
            )
            if confirm:
                try:
                    # Determine tracking method
                    tracking_method = config.get('experiment', {}).get('methods_order', ['mp'])[0]

                    # Run calibration with proper arguments
                    result = subprocess.run([
                        sys.executable,
                        'finger_tracing_refactor/calibrate_main.py',
                        '--config', 'config.yaml',
                        '--method', tracking_method
                    ], cwd=os.getcwd())

                    if result.returncode == 0:
                        messagebox.showinfo("Success", "Calibration completed successfully!", parent=dialog)
                    else:
                        messagebox.showerror("Error", f"Calibration failed with exit code {result.returncode}", parent=dialog)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to run calibration:\n{str(e)}", parent=dialog)

        ttk.Button(calibration_frame, text="Run Calibration", command=run_calibration, style='Accent.TButton').pack(pady=5)

        # Save function
        def save_settings():
            try:
                # Update config with new values
                if 'experiment' not in config:
                    config['experiment'] = {}
                config['experiment']['methods_order'] = [tracking_var.get()]

                if 'led' not in config:
                    config['led'] = {}
                config['led']['hsv_low'] = [int(hsv_low_h.get()), int(hsv_low_s.get()), int(hsv_low_v.get())]
                config['led']['hsv_high'] = [int(hsv_high_h.get()), int(hsv_high_s.get()), int(hsv_high_v.get())]
                config['led']['brightness_threshold'] = int(brightness_entry.get())

                if 'spiral' not in config:
                    config['spiral'] = {}
                config['spiral']['a'] = float(spiral_a_entry.get())
                config['spiral']['b'] = float(spiral_b_entry.get())
                config['spiral']['turns'] = float(spiral_turns_entry.get())
                config['spiral']['line_thickness'] = int(spiral_thickness_entry.get())

                if 'dot_follow' not in config:
                    config['dot_follow'] = {}
                config['dot_follow']['dot_speed_sec_per_spiral'] = float(dot_speed_entry.get())
                config['dot_follow']['countdown_sec'] = int(countdown_entry.get())

                if 'stereo_3d' not in config:
                    config['stereo_3d'] = {}
                config['stereo_3d']['disparity_offset_px'] = float(disparity_offset_entry.get())

                # Save to file
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)

                messagebox.showinfo("Success", "Settings saved successfully!", parent=dialog)
                dialog.destroy()

            except ValueError as e:
                messagebox.showerror("Error", f"Invalid input: {str(e)}", parent=dialog)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings:\n{str(e)}", parent=dialog)

        # Pack canvas and scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Buttons at bottom
        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(side=tk.BOTTOM, pady=10)

        ttk.Button(buttons_frame, text="Save Settings", command=save_settings, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def run(self):
        """Start the UI application."""
        self.root.mainloop()


if __name__ == "__main__":
    app = DoctorUI()
    app.run()
