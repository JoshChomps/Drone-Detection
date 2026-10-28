"""
Per-track features for telling drones from birds, aircraft and clutter by HOW THEY MOVE (milestone M2).

A single frame of a few-pixel target carries almost no class information (doc 07 §3); a track does:
drones hover, stop and turn sharply; birds flap (periodic shape change) and glide; aircraft fly straight.

Speeds are expressed in **body lengths per second** (pixel speed / median box width). For a target crossing
the view, pixel speed and pixel size both scale as 1/range, so their ratio is roughly range-independent.

`silhouette` is the cheap alternative to a segmentation network: inside a box against sky, Otsu thresholding
separates target from background in a fraction of a millisecond. Its area and brightness over time are the
flap signal. It works on uniform sky; against clutter it degrades and should be treated as missing.
"""
import numpy as np

FEATURE_NAMES = [
    "duration_s", "n_obs", "obs_rate_hz",
    "speed_bl_med", "speed_bl_p90", "hover_frac", "straightness", "turn_rate_med", "accel_bl_p90",
    "vert_frac", "size_cv", "aspect_med", "aspect_cv",
    "aspect_peak_hz", "aspect_peak_pow", "area_peak_hz", "area_peak_pow", "bright_peak_hz", "bright_peak_pow",
    "score_mean", "cls_drone_frac",
]


def silhouette(gray_crop: np.ndarray):
    """(area_px, mean_contrast) of the target blob in a crop around one box, or (nan, nan) if unusable.

    Polarity is decided from the crop's border (assumed background): the blob is whichever side of the Otsu
    threshold the border is NOT on. Only the connected component touching the crop centre is kept.
    """
    import cv2
    g = np.asarray(gray_crop)
    if g.ndim != 2 or min(g.shape) < 3:
        return float("nan"), float("nan")
    g = g.astype(np.uint8)
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]]).astype(float)
    thr, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(border > thr) > 0.5:            # border is bright, so the target is the dark side
        mask = 255 - mask
    n, lab = cv2.connectedComponents((mask > 0).astype(np.uint8))
    c = lab[g.shape[0] // 2, g.shape[1] // 2]
    if c == 0:                                 # centre pixel is background: take the largest component
        if n <= 1:
            return float("nan"), float("nan")
        c = 1 + int(np.argmax([(lab == k).sum() for k in range(1, n)]))
    blob = lab == c
    sd = border.std() or 1.0
    return float(blob.sum()), float(abs(g[blob].mean() - border.mean()) / sd)


def silhouette_at(gray: np.ndarray, box):
    """`silhouette` on a crop around box (x, y, w, h) with a margin of half the box (at least 4 px)."""
    x, y, w, h = box[:4]
    m = max(4.0, 0.5 * max(w, h))
    x0, y0 = int(max(0, x - m)), int(max(0, y - m))
    x1, y1 = int(min(gray.shape[1], x + w + m)), int(min(gray.shape[0], y + h + m))
    return silhouette(gray[y0:y1, x0:x1])


def _peak(t: np.ndarray, v: np.ndarray, fmin: float = 0.5):
    """Dominant frequency (Hz) and its normalised Lomb-Scargle power for an unevenly sampled series.

    Returns (nan, nan) when the series is too short (< 16 samples or < 1 s) to say anything.
    """
    from scipy.signal import lombscargle
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    if len(t) < 16 or t[-1] - t[0] < 1.0:
        return float("nan"), float("nan")
    v = v - np.polyval(np.polyfit(t, v, 1), t)          # remove slow trend (approach/recede)
    if v.std() == 0:
        return float("nan"), 0.0
    fmax = 0.5 / np.median(np.diff(t))                   # Nyquist of the typical sampling interval
    f = np.linspace(fmin, fmax, 200)
    p = lombscargle(t, v, 2 * np.pi * f, normalize=True)
    k = int(np.argmax(p))
    return float(f[k]), float(p[k])


def ttc_expansion(t, w, window_s: float = 2.0, min_obs: int = 5, max_ttc: float = 50.0, z: float = 4.0,
                  min_span_s: float = 0.5):
    """Time to collision with the camera, in seconds, from optical expansion (doc 09 §1.3), or None.

    A target of size S closing at constant speed v, due at time T, has width w = f S / (v (T - t)), so 1/w is a
    straight line in time that crosses zero at T, whatever its size or range. A least-squares line through 1/width
    over the last `window_s` seconds gives T directly: exact for constant closing speed, with no lag correction
    (the earlier fit of ln(width) estimated TTC at the window's middle and curved the wrong way near impact).
    None when the track is too short, not measurably closing, or more than `max_ttc` away.

    "Measurably" means the slope exceeds `z` standard errors of the fit. Detector box widths jitter by 3-6%
    (measured on Halmstad tracks), and five boxes spanning 0.13 s give a slope that is pure noise: with no test,
    75% of NON-closing tracks read "TTC under 10 s" at least once in their first 3 s. With z = 4, a 2-s window and
    `min_span_s`, under 0.5% do, while a real TTC of 10 s is always seen (90% of errors under 2.2 s; under 0.3 s at
    TTC 3 s). A 1-s window cannot resolve 10 s at that jitter: it sees it 29% of the time (16 #54).

    The boxes must span `min_span_s`: a flapping bird's box widens 2-3x within one wing beat, a clean trend that
    passes any noise test and reads as "TTC 0.2 s" (found by the simulated scene, 16 #54). Over a full beat (0.5 s
    covers flapping down to 2 Hz) the oscillation lands in the residuals and the test rejects it.

    Known bias: blur adds a roughly constant amount to a small box, which flattens its growth, so TTC reads LONG
    while the target is only a few pixels wide (simulated scene: 9.6 s for a true 6.5 s at 4 px; 3.0 for 2.7 at 14 px).
    """
    t, w = np.asarray(t, float), np.asarray(w, float)
    k = (t >= t[-1] - window_s) & (w > 0) if len(t) else np.zeros(0, bool)
    if k.sum() < min_obs or t[k][-1] - t[k][0] < min_span_s:
        return None
    tk, y = t[k], 1.0 / w[k]
    slope, c = np.polyfit(tk, y, 1)
    se = np.sqrt(np.sum((y - (slope * tk + c)) ** 2) / (len(tk) - 2) / np.sum((tk - tk.mean()) ** 2))
    if -slope <= z * se:                                # not shrinking 1/w, i.e. not growing, beyond the noise
        return None
    ttc = -(c + slope * t[-1]) / slope                  # time from the last box to the line's zero crossing
    return float(max(ttc, 0.0)) if ttc <= max_ttc else None


def track_features(rows: np.ndarray, area=None, bright=None, drone_cls: int = 0, warmup: int = 3) -> dict:
    """Features for one track.

    rows: Track.rows() -> columns t, cx, cy, w, h, score, cls, fx, fy, fvx, fvy (f* = filtered state).
    area, bright: optional per-row silhouette area (px) and blob brightness/contrast, same length as rows.
    warmup: rows whose filtered VELOCITY is ignored. A new track starts at zero velocity, so its first few
    estimates read as hovering and as a huge acceleration.
    """
    rows = np.asarray(rows, float)
    t, w, h = rows[:, 0], rows[:, 3], rows[:, 4]
    score, cls = rows[:, 5], rows[:, 6]
    fx, fy = rows[:, 7], rows[:, 8]
    k0 = warmup if len(rows) > warmup + 2 else 0
    tv, vx, vy = t[k0:], rows[k0:, 9], rows[k0:, 10]
    dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    size = float(np.median(w)) or 1.0
    speed = np.hypot(vx, vy) / size                      # body lengths per second
    path = float(np.sum(np.hypot(np.diff(fx), np.diff(fy)))) if len(t) > 1 else 0.0
    net = float(np.hypot(fx[-1] - fx[0], fy[-1] - fy[0]))
    moving = speed > 0.5
    heading = np.unwrap(np.arctan2(vy, vx))
    if moving.sum() > 2 and len(tv) > 2:
        turn = np.abs(np.diff(heading)) / np.maximum(np.diff(tv), 1e-6)
        turn_med = float(np.median(turn[moving[1:]])) if moving[1:].any() else float("nan")
    else:
        turn_med = float("nan")
    if len(tv) > 2:
        acc = np.hypot(np.diff(vx), np.diff(vy)) / np.maximum(np.diff(tv), 1e-6) / size
        acc_p90 = float(np.percentile(acc, 90))
    else:
        acc_p90 = float("nan")
    aspect = h / np.maximum(w, 1e-6)
    feats = {
        "duration_s": dur, "n_obs": float(len(t)), "obs_rate_hz": (len(t) - 1) / dur if dur > 0 else float("nan"),
        "speed_bl_med": float(np.median(speed)), "speed_bl_p90": float(np.percentile(speed, 90)),
        "hover_frac": float(np.mean(~moving)),
        "straightness": net / path if path > 0 else float("nan"),
        "turn_rate_med": turn_med, "accel_bl_p90": acc_p90,
        "vert_frac": float(np.sum(np.abs(np.diff(fy))) / path) if path > 0 else float("nan"),
        "size_cv": float(np.std(w) / size),
        "aspect_med": float(np.median(aspect)), "aspect_cv": float(np.std(aspect) / (np.median(aspect) or 1.0)),
        "score_mean": float(np.mean(score)), "cls_drone_frac": float(np.mean(cls == drone_cls)),
    }
    feats["aspect_peak_hz"], feats["aspect_peak_pow"] = _peak(t, aspect)
    for name, series in (("area", area), ("bright", bright)):
        if series is None:
            feats[f"{name}_peak_hz"] = feats[f"{name}_peak_pow"] = float("nan")
        else:
            feats[f"{name}_peak_hz"], feats[f"{name}_peak_pow"] = _peak(t, np.asarray(series, float))
    return {k: feats[k] for k in FEATURE_NAMES}
