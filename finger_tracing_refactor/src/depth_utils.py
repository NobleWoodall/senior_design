import numpy as np
def median_depth_mm(depth_img, x:int, y:int, win:int, min_depth:float=200.0, max_depth:float=600.0)->float:
    """
    Get median depth from a window around (x, y) with range filtering.

    Args:
        depth_img: Depth image in mm
        x, y: Center coordinates
        win: Window size (e.g., 5 for 5x5 patch)
        min_depth: Minimum valid depth in mm (default 200mm)
        max_depth: Maximum valid depth in mm (default 600mm)

    Returns:
        Median depth in mm, or nan if no valid pixels
    """
    h, w = depth_img.shape[:2]
    r = win // 2
    x0=max(0,x-r)
    x1=min(w, x+r+1)
    y0=max(0,y-r)
    y1=min(h, y+r+1)
    patch = depth_img[y0:y1, x0:x1].astype(np.float32)
    # Filter by range: only keep values in [min_depth, max_depth]
    valid = patch[(patch >= min_depth) & (patch <= max_depth)]
    return float(np.median(valid)) if valid.size else float('nan')
