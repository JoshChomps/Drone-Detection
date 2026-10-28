"""
E21: how much sensor noise, blur, compression and haze can the detector take? (REAL frames, degraded.)

The zoom-out mosaics are cleaner than a real distant drone (15 section 7.3): area-downscaling averages noise and
blur away. This puts numbers on that caveat without new footage or a renderer: every evaluation image is degraded
just before the detector sees it, and drone recall and background false alarms are re-measured at the run's own
operating threshold. One degradation at a time, so each row is one cause.

    C:\\cuas_venv\\Scripts\\python experiments/e21_degradation.py [run or run@last]      (~13 min on the laptop GPU)

The last four conditions are HELD OUT: E22 trains on Gaussian blur, white noise and JPEG (cuas/degrade.py), so a
gain measured only on those would repeat the E2 mistake (train and test from one generator). Motion blur,
signal-dependent noise and WebP are never used in training.

Sets: Halmstad val native (~34 px drones), Halmstad k4 mosaics (~8.5 px) and Anti-UAV TEST k8 mosaics (~14 px).
Output: C:/cuas_data/runs/e21_<run>.txt and .json.
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.detector import load as load_detector  # noqa: E402
from cuas.eval import evaluate, run_files, run_split  # noqa: E402

D = Path(os.environ.get("CUAS_DATA", "C:/cuas_data"))
SETS = {"halmstad native": D / "halmstad_yolo" / "val", "halmstad k4": D / "halmstad_zoom" / "k4",
        "antiuav test k8": D / "antiuav_zoom_test" / "k8"}


def noise(sigma):
    def f(img, rng):
        return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    return f


def blur(sigma):
    return lambda img, rng: cv2.GaussianBlur(img, (0, 0), sigma)


def jpeg(q):
    return lambda img, rng: cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def haze(k):
    """Contrast scaled by k towards a bright veil (airlight), as distance through haze does (10 section 3)."""
    return lambda img, rng: np.clip(k * img.astype(np.float32) + (1 - k) * 200, 0, 255).astype(np.uint8)


def field(img, rng):
    """A plausible distant target on a small sensor: defocus, then shot/read noise, then video compression."""
    return jpeg(40)(noise(6)(blur(1.0)(img, rng), rng), rng)


def motion(length):
    """Linear motion blur along a 30-degree line (target or mount motion during the exposure)."""
    k = np.zeros((length, length), np.float32)
    c = (length - 1) / 2
    for t in np.linspace(-c, c, 4 * length):
        k[int(round(c + t * 0.5)), int(round(c + t * 0.866))] = 1
    k /= k.sum()
    return lambda img, rng: cv2.filter2D(img, -1, k)


def shot(gain):
    """Signal-dependent (photon) noise: variance proportional to brightness, sigma = sqrt(gain * value)."""
    def f(img, rng):
        x = img.astype(np.float32)
        return np.clip(x + rng.standard_normal(x.shape, dtype=np.float32) * np.sqrt(gain * x), 0, 255).astype(np.uint8)
    return f


def webp(q):
    return lambda img, rng: cv2.imdecode(cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, q])[1], cv2.IMREAD_COLOR)


def field2(img, rng):
    """The held-out counterpart of `field`: motion blur, photon noise, WebP."""
    return webp(30)(shot(0.4)(motion(5)(img, rng), rng), rng)


CONDITIONS = [("clean", None), ("noise sigma 4", noise(4)), ("noise sigma 8", noise(8)), ("noise sigma 16", noise(16)),
              ("blur sigma 0.75", blur(0.75)), ("blur sigma 1.5", blur(1.5)),
              ("jpeg q50", jpeg(50)), ("jpeg q25", jpeg(25)), ("jpeg q10", jpeg(10)),
              ("haze contrast 0.5", haze(0.5)), ("haze contrast 0.25", haze(0.25)),
              ("blur 1.0 + noise 6 + jpeg 40", field),
              ("HELD OUT motion blur 5 px", motion(5)), ("HELD OUT shot noise gain 0.4", shot(0.4)),
              ("HELD OUT webp q30", webp(30)), ("HELD OUT motion 5 + shot 0.4 + webp 30", field2)]


class Degraded:
    def __init__(self, det, fn, seed=0):
        self.det, self.fn, self.rng, self.names = det, fn, np.random.default_rng(seed), det.names

    def __call__(self, images):
        return self.det([self.fn(i, self.rng) for i in images] if self.fn else images)


def main(run):
    name, weights, sfx = run_files(run)
    conf = json.loads((D / "runs" / name / f"e12_eval{sfx}.json").read_text())["conf"]
    det = load_detector(str(D / "runs" / name / "weights" / weights), conf=0.05, device=0)
    lines = [f"E21 degradation sweep: {run} at conf {conf} (its val-chosen threshold). Drone recall / background false "
             f"drones per 1000 images.", f"{'condition':40s}" + "".join(f"{n:>22s}" for n in SETS)]
    print("\n".join(lines), flush=True)
    out = {}
    for name, fn in CONDITIONS:
        cells = []
        for set_name, path in SETS.items():
            r = evaluate(run_split(Degraded(det, fn), path, conf=conf, tile=640))
            rec = sum(r["n_hit"]) / max(sum(r["n_gt"]), 1)
            fa = 1000 * r["false_drone_on_background"] / r["n_images"]
            out.setdefault(name, {})[set_name] = {"recall": rec, "bg_per_1000": fa, "drones": sum(r["n_gt"])}
            cells.append(f"{rec:.3f} / {fa:6.1f}")
        lines.append(f"{name:40s}" + "".join(f"{c:>22s}" for c in cells))
        print(lines[-1], flush=True)
    text = "\n".join(lines)
    key = run.replace("@", "_")
    (D / "runs" / f"e21_{key}.txt").write_text(text)
    (D / "runs" / f"e21_{key}.json").write_text(json.dumps({"run": run, "conf": conf, "results": out}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E12_multidata_yolo26s")
