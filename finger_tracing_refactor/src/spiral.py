import numpy as np
import cv2
from typing import Tuple, Dict

class Spiral:
    def __init__(self, width:int, height:int, a:float, b:float, turns:float, theta_step:float):
        self.width=width
        self.height=height
        self.cx=width//2
        self.cy=height//2
        self.a=a
        self.b=b
        self.turns=turns
        self.theta_step=theta_step
        self._precompute()

    def _precompute(self):
        theta_max = 2*np.pi*self.turns
        thetas = np.arange(0, theta_max+self.theta_step, self.theta_step, dtype=np.float32)
        r = self.a + self.b*thetas
        xs = self.cx + r*np.cos(thetas)
        ys = self.cy + r*np.sin(thetas)
        ds = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        self.curve = np.stack([xs, ys], axis=1)
        self.thetas = thetas
        self.s = s

    def draw(self, frame, color=(40,220,40), thickness=3):
        pts = self.curve.astype(np.int32).reshape(-1,1,2)
        cv2.polylines(frame, [pts], False, color, thickness)

    def nearest_point(self, x:float, y:float):
        dx = self.curve[:,0]-x
        dy = self.curve[:,1]-y
        i = int(np.argmin(dx*dx+dy*dy))
        xs, ys = float(self.curve[i,0]), float(self.curve[i,1])
        s = float(self.s[i])
        if i < len(self.curve)-1:
            tx = self.curve[i+1,0]-self.curve[i,0]
            ty = self.curve[i+1,1]-self.curve[i,1]
        else:
            tx = self.curve[i,0]-self.curve[i-1,0]
            ty = self.curve[i,1]-self.curve[i-1,1]
        theta_tan = float(np.arctan2(ty, tx))
        return xs, ys, s, theta_tan

    def endpoints(self)->Dict[str,Tuple[int,int]]:
        return {"start": tuple(self.curve[0].astype(int)), "end": tuple(self.curve[-1].astype(int))}
