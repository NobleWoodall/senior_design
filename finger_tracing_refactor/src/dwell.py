from dataclasses import dataclass
from typing import Optional

@dataclass
class DwellState:
    armed: bool=False
    recording: bool=False
    end_detected: bool=False
    start_enter_time: Optional[float]=None
    end_enter_time: Optional[float]=None

class DwellDetector:
    def __init__(self, start_radius_px:int, start_dwell_sec:float, end_radius_px:int, stop_dwell_sec:float, hysteresis_px:int):
        self.cfg = (start_radius_px, start_dwell_sec, end_radius_px, stop_dwell_sec, hysteresis_px)
        self.state = DwellState()

    def reset(self): self.state = DwellState()

    def update(self, t_now:float, dist_to_start:float, dist_to_end:float):
        start_r, start_sec, end_r, stop_sec, hyst = self.cfg
        st = self.state
        if not st.recording:
            if dist_to_start <= start_r + hyst:
                if st.start_enter_time is None: st.start_enter_time = t_now
                elif (t_now - st.start_enter_time) >= start_sec:
                    st.armed = True; st.recording = True
            else:
                st.start_enter_time = None; st.armed = False
        else:
            if dist_to_end <= end_r + hyst:
                if st.end_enter_time is None: st.end_enter_time = t_now
                elif (t_now - st.end_enter_time) >= stop_sec:
                    st.end_detected = True
            else:
                st.end_enter_time = None
        return st
