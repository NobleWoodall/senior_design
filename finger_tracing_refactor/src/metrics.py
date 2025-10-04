import numpy as np
def summarize_errors(times, errs, depths, valid_mask):
    t = np.array(times, dtype=np.float64)
    e = np.array(errs, dtype=np.float64)
    d = np.array(depths, dtype=np.float64)
    vm = np.array(valid_mask, dtype=bool)
    if t.size==0:
        return {"sample_count":0,"duration_sec":0,"fps_avg":0,
                "err_px":{"mean":None,"median":None,"std":None,"p90":None,"p95":None,"max":None},
                "rmse_time_weighted":None,"tracking_loss_rate":1.0,
                "mean_depth_mm":None,"valid_depth_fraction_avg":None}
    duration = max(0.0, t[-1]-t[0]); fps = (t.size/duration) if duration>0 else 0.0
    v = np.isfinite(e); e_v = e[v]
    stats = {"mean": float(np.mean(e_v)) if e_v.size else None,
             "median": float(np.median(e_v)) if e_v.size else None,
             "std": float(np.std(e_v)) if e_v.size else None,
             "p90": float(np.percentile(e_v,90)) if e_v.size else None,
             "p95": float(np.percentile(e_v,95)) if e_v.size else None,
             "max": float(np.max(e_v)) if e_v.size else None}
    if t.size>=2:
        dt = np.diff(t, prepend=t[0])
        rmse = float(np.sqrt(np.nansum((e**2)*dt) / np.sum(dt)))
    else:
        rmse = float(np.sqrt(np.nanmean(e**2))) if e.size else None
    loss = float(1.0 - (np.count_nonzero(v)/e.size)) if e.size else 1.0
    mean_depth = float(np.nanmean(d)) if np.isfinite(d).any() else None
    valid_depth_frac = float(np.mean(valid_mask)) if len(valid_mask) else None
    return {"sample_count":int(t.size),"duration_sec":float(duration),"fps_avg":float(fps),
            "err_px":stats,"rmse_time_weighted":rmse,"tracking_loss_rate":loss,
            "mean_depth_mm":mean_depth,"valid_depth_fraction_avg":valid_depth_frac}
