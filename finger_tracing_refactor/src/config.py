from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import time

@dataclass
class SessionMetadata:
    """Metadata for clinical session tracking (MRgFUS)"""
    patient_id: str = "ANON"
    date: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))
    clinician: str = ""
    notes: str = ""
    trial_name: str = "Baseline"  # Custom trial name (e.g., "Preop", "Intraop 1", "Postop")
    trial_order: int = 0  # Sorting order for trials (0=baseline, 1=preop, 2=intraop1, etc.)

@dataclass
class CameraCfg:
    width:int=640
    height:int=480
    fps:int=30
    use_auto_exposure:bool=True
    exposure:int=100
    depth_width:int=640
    depth_height:int=480
    depth_fps:int=30
    depth_window:int=5
    min_depth_mm:float=200.0
    max_depth_mm:float=600.0

@dataclass
class SpiralCfg:
    a:float=20.0
    b:float=14.0
    turns:float=3.5
    theta_step:float=0.01
    line_thickness:int=3
    color_bgr:Tuple[int,int,int]=(40,220,40)

@dataclass
class DotFollowCfg:
    countdown_sec:int=3
    dot_speed_sec_per_spiral:float=20.0
    end_wait_sec:float=3.0
    depth_close_mm:float=400.0
    depth_far_mm:float=600.0
    display_smooth_alpha:float=0.3  # Display smoothing (0.0=max smooth, 1.0=no smooth)

@dataclass
class MPcfg:
    model_complexity:int=1
    detection_confidence:float=0.9
    tracking_confidence:float=0.9

@dataclass
class LEDCfg:
    hsv_low:Tuple[int,int,int]=(0,60,160)
    hsv_high:Tuple[int,int,int]=(110,255,255)
    brightness_threshold:int=110
    morph_kernel:int=3
    min_area:int=5

@dataclass
class ExperimentCfg:
    methods_order:List[str]=field(default_factory=lambda:["mp","hsv"])
    save_preview:bool=False
    output_dir:str="runs"
    max_jump_px:int=60

@dataclass
class TremorAnalysisCfg:
    band_low_hz:float=4.0
    band_high_hz:float=10.0

@dataclass
class Stereo3DCfg:
    target_depth_m:float=0.5
    disparity_offset_px:float=0.0
    flip_x:bool=False
    flip_y:bool=False
    finger_scale_x:float=1.0
    finger_scale_y:float=1.0

@dataclass
class CalibrationCfg:
    enabled:bool=False
    calibration_file:str="calibration.json"
    dot_speed_revolutions_per_sec:float=0.15
    num_traces:int=2
    countdown_sec:int=3

@dataclass
class AppConfig:
    camera:CameraCfg=field(default_factory=CameraCfg)
    spiral:SpiralCfg=field(default_factory=SpiralCfg)
    dot_follow:DotFollowCfg=field(default_factory=DotFollowCfg)
    mediapipe:MPcfg=field(default_factory=MPcfg)
    led:LEDCfg=field(default_factory=LEDCfg)
    experiment:ExperimentCfg=field(default_factory=ExperimentCfg)
    tremor_analysis:TremorAnalysisCfg=field(default_factory=TremorAnalysisCfg)
    stereo_3d:Stereo3DCfg=field(default_factory=Stereo3DCfg)
    calibration:CalibrationCfg=field(default_factory=CalibrationCfg)
    session_metadata:SessionMetadata=field(default_factory=SessionMetadata)
    show_live_preview:bool=True

    @staticmethod
    def from_dict(d:Dict[str,Any])->"AppConfig":
        return AppConfig(
            camera=CameraCfg(**d.get("camera",{})),
            spiral=SpiralCfg(**d.get("spiral",{})),
            dot_follow=DotFollowCfg(**d.get("dot_follow",{})),
            mediapipe=MPcfg(**d.get("mediapipe",{})),
            led=LEDCfg(**d.get("led",{})),
            experiment=ExperimentCfg(**d.get("experiment",{})),
            tremor_analysis=TremorAnalysisCfg(**d.get("tremor_analysis",{})),
            stereo_3d=Stereo3DCfg(**d.get("stereo_3d",{})),
            calibration=CalibrationCfg(**d.get("calibration",{})),
            session_metadata=SessionMetadata(**d.get("session_metadata",{})),
            show_live_preview=bool(d.get("show_live_preview", True)),
        )
