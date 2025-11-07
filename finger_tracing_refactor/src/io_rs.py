import time
from typing import Tuple, Dict, Any, List
import numpy as np
import pyrealsense2 as rs

def _list_profiles(device):
    out = {"color": [], "depth": []}
    for s in device.sensors:
        for p in s.profiles:
            vs = p.as_video_stream_profile()
            if not vs: 
                continue
            entry = (vs.stream_type(), vs.width(), vs.height(), vs.fps(), vs.format())
            if vs.stream_type() == rs.stream.color:
                out["color"].append(entry)
            elif vs.stream_type() == rs.stream.depth:
                out["depth"].append(entry)
    return out

def _has_profile(profiles, stream, w, h, fps, fmt):
    for st, pw, ph, pfps, pfmt in profiles:
        if st == stream and pw == w and ph == h and pfps == fps and pfmt == fmt:
            return True
    return False

class RealSenseIO:
    def __init__(self, width, height, fps, depth_width, depth_height, depth_fps,
                 use_auto_exposure=True, exposure=100):
        self.width=width
        self.height=height
        self.fps=fps
        self.depth_width=depth_width
        self.depth_height=depth_height
        self.depth_fps=depth_fps
        self.use_auto_exposure=use_auto_exposure
        self.exposure=exposure
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.profile = None

    def start(self):
        ctx = rs.context()
        if len(ctx.devices)==0:
            raise RuntimeError("No RealSense device found.")
        dev = ctx.devices[0]
        profiles = _list_profiles(dev)
        COLOR_FMT = rs.format.bgr8
        DEPTH_FMT = rs.format.z16
        want_color=(rs.stream.color, self.width, self.height, self.fps, COLOR_FMT)
        want_depth=(rs.stream.depth, self.depth_width, self.depth_height, self.depth_fps, DEPTH_FMT)
        fallback_color=(rs.stream.color, 640,480,30,COLOR_FMT)
        fallback_depth=(rs.stream.depth, 640,480,30,DEPTH_FMT)

        color_ok=_has_profile(profiles["color"], *want_color)
        depth_ok=_has_profile(profiles["depth"], *want_depth)
        use_color = want_color if color_ok else fallback_color
        use_depth = want_depth if depth_ok else fallback_depth

        if (not color_ok) or (not depth_ok):
            print("[RealSense] Falling back to: "
                  f"{use_color[1]}x{use_color[2]}@{use_color[3]} (color), "
                  f"{use_depth[1]}x{use_depth[2]}@{use_depth[3]} (depth)")
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, use_color[1], use_color[2], use_color[4], use_color[3])
        cfg.enable_stream(rs.stream.depth, use_depth[1], use_depth[2], use_depth[4], use_depth[3])
        self.profile = self.pipeline.start(cfg)

        try:
            sensor = self.profile.get_device().first_color_sensor()
            if not self.use_auto_exposure:
                sensor.set_option(rs.option.enable_auto_exposure, 0)
                sensor.set_option(rs.option.exposure, float(self.exposure))
            else:
                sensor.set_option(rs.option.enable_auto_exposure, 1)
            # Disable auto-white-balance for consistent LED brightness
            sensor.set_option(rs.option.enable_auto_white_balance, 1)

        except Exception:
            pass
        time.sleep(1.0)

    def stop(self):
        try: 
            self.pipeline.stop()
        except Exception: 
            pass

    def get_aligned(self)->Tuple[np.ndarray, np.ndarray, float]:
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        t = time.monotonic()
        if not color or not depth: 
            return None, None, t
        return np.asanyarray(color.get_data()), np.asanyarray(depth.get_data()), t

    def get_intrinsics(self)->Dict[str,Any]:
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        c = color_stream.get_intrinsics()
        d = depth_stream.get_intrinsics()
        ext = depth_stream.get_extrinsics_to(color_stream)
        return {
            "color_intrinsics": dict(width=c.width,height=c.height,ppx=c.ppx,ppy=c.ppy,fx=c.fx,fy=c.fy,model=str(c.model),coeffs=list(c.coeffs)),
            "depth_intrinsics": dict(width=d.width,height=d.height,ppx=d.ppx,ppy=d.ppy,fx=d.fx,fy=d.fy,model=str(d.model),coeffs=list(d.coeffs)),
            "depth_to_color_extrinsics": {"rotation": list(ext.rotation), "translation": list(ext.translation)},
            "aligned_mode": "depth->color"
        }
