import os, csv, json, yaml, cv2
from pathlib import Path
from typing import Dict, Any

class RunSaver:
    def __init__(self, base_dir:str, method:str, spiral_id:str):
        ts = __import__('time').strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / f"{ts}_method-{method}_spiral-{spiral_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.frames_path = self.run_dir / "frames.csv"
        self.preview_path = self.run_dir / "preview.mp4"
        self.summary_path = self.run_dir / "summary.json"
        self.header = ["t_sec","method","frame_idx","x_px","y_px","z_mm",
                       "dot_x_px","dot_y_px","dot_dist_px",
                       "x_spiral_px","y_spiral_px","s_spiral_px",
                       "err_px","depth_window","valid_depth_fraction"]
        self.csv_file = open(self.frames_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(self.header)
        self.video_writer = None

    def save_config_snapshot(self, cfg:Dict[str,Any]):
        with open(self.run_dir / "config.yaml", 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    def save_intrinsics(self, intr:Dict[str,Any]):
        with open(self.run_dir / "intrinsics.json", 'w') as f:
            json.dump(intr, f, indent=2)

    def write_frame_row(self, row):
        self.writer.writerow(row)

    def open_video(self, w:int, h:int, fps:int):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(str(self.preview_path), fourcc, fps, (w,h))

    def write_video_frame(self, frame_bgr):
        if self.video_writer is not None: 
            self.video_writer.write(frame_bgr)

    def close(self):
        if self.video_writer is not None: 
            self.video_writer.release()
            self.video_writer=None
        if hasattr(self, "csv_file") and self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file=None

    def save_summary(self, summary:Dict[str,Any]):
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
