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
    use_depth_filter:bool=True

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
    trials_per_method:int=1
    save_preview:bool=False
    output_dir:str="runs"
    metronome_bpm:int=0
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
class AppConfig:
    camera:CameraCfg=field(default_factory=CameraCfg)
    spiral:SpiralCfg=field(default_factory=SpiralCfg)
    dwell:DwellCfg=field(default_factory=DwellCfg)
    mediapipe:MPcfg=field(default_factory=MPcfg)
    color:ColorCfg=field(default_factory=ColorCfg)
    led:LEDCfg=field(default_factory=LEDCfg)
    experiment:ExperimentCfg=field(default_factory=ExperimentCfg)
    tremor_analysis:TremorAnalysisCfg=field(default_factory=TremorAnalysisCfg)
    stereo_3d:Stereo3DCfg=field(default_factory=Stereo3DCfg)
    show_live_preview:bool=True

    @staticmethod
    def from_dict(d:Dict[str,Any])->"AppConfig":
        return AppConfig(
            camera=CameraCfg(**d.get("camera",{})),
            spiral=SpiralCfg(**d.get("spiral",{})),
            dwell=DwellCfg(**d.get("dwell",{})),
            mediapipe=MPcfg(**d.get("mediapipe",{})),
            color=ColorCfg(**d.get("color",{})),
            led=LEDCfg(**d.get("led",{})),
            experiment=ExperimentCfg(**d.get("experiment",{})),
            tremor_analysis=TremorAnalysisCfg(**d.get("tremor_analysis",{})),
            stereo_3d=Stereo3DCfg(**d.get("stereo_3d",{})),
            show_live_preview=bool(d.get("show_live_preview", True)),
        )
