import numpy as np
def median_depth_mm(depth_img, x:int, y:int, win:int)->float:
    h, w = depth_img.shape[:2]
    r = win // 2
    x0=max(0,x-r)
    x1=min(w, x+r+1)
    y0=max(0,y-r)
    y1=min(h, y+r+1)
    patch = depth_img[y0:y1, x0:x1].astype(np.float32)
    valid = patch[patch>0]
    return float(np.median(valid)) if valid.size else float('nan')
