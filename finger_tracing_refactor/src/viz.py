import cv2
def draw_status(frame, txt, color=(200,200,200)):
    cv2.putText(frame, txt, (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

def draw_cross(frame, x:int, y:int, color=(0,255,255)):
    cv2.drawMarker(frame, (int(x),int(y)), color, markerType=cv2.MARKER_CROSS, markerSize=24, thickness=3)

def draw_circle(frame, x:int, y:int, radius:int, color=(0,255,255), thickness=2):
    cv2.circle(frame, (int(x),int(y)), int(radius), color, thickness)

def draw_meter(frame, err_px:float):
    cv2.putText(frame, f"err: {err_px:.1f}px", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

def metronome_overlay(frame, t_now:float, bpm:int):
    if bpm<=0:
        return
    period = 60.0/bpm
    phase = (t_now % period)/period
    w = frame.shape[1]
    x = int(10 + phase*(w-20))
    cv2.line(frame, (x,0), (x, frame.shape[0]-1), (255,255,255), 2)
