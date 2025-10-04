from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

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

@dataclass
class SpiralCfg:
    a:float=20.0
    b:float=14.0
    turns:float=3.5
    theta_step:float=0.01
    line_thickness:int=3
    color_bgr:Tuple[int,int,int]=(40,220,40)
    start_radius_px:int=18
    end_radius_px:int=18

@dataclass
class DwellCfg:
    StartRadiusPx:int=18
    StartDwellSec:float=1.0
    EndRadiusPx:int=18
    StopDwellSec:float=1.0
    hysteresis_px:int=6

@dataclass
class MPcfg:
    model_complexity:int=1
    detection_confidence:float=0.9
    tracking_confidence:float=0.9
    ema_alpha:float=1.0

@dataclass
class ColorCfg:
    hsv_low:Tuple[int,int,int]=(35,60,60)
    hsv_high:Tuple[int,int,int]=(85,255,255)
    morph_kernel:int=3
    min_area:int=60
    use_flow_fallback:bool=True
    flow_win:int=15
    flow_max_level:int=2

@dataclass
class ExperimentCfg:
    methods_order:List[str]=field(default_factory=lambda:["mp","hsv"])
    trials_per_method:int=1
    save_preview:bool=False
    output_dir:str="runs"
    metronome_bpm:int=0
    max_jump_px:int=60

@dataclass
class AppConfig:
    camera:CameraCfg=field(default_factory=CameraCfg)
    spiral:SpiralCfg=field(default_factory=SpiralCfg)
    dwell:DwellCfg=field(default_factory=DwellCfg)
    mediapipe:MPcfg=field(default_factory=MPcfg)
    color:ColorCfg=field(default_factory=ColorCfg)
    experiment:ExperimentCfg=field(default_factory=ExperimentCfg)
    show_live_preview:bool=True

    @staticmethod
    def from_dict(d:Dict[str,Any])->"AppConfig":
        return AppConfig(
            camera=CameraCfg(**d.get("camera",{})),
            spiral=SpiralCfg(**d.get("spiral",{})),
            dwell=DwellCfg(**d.get("dwell",{})),
            mediapipe=MPcfg(**d.get("mediapipe",{})),
            color=ColorCfg(**d.get("color",{})),
            experiment=ExperimentCfg(**d.get("experiment",{})),
            show_live_preview=bool(d.get("show_live_preview", True)),
        )
