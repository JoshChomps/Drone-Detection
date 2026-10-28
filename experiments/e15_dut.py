"""
E15: evaluate trained runs on DUT Anti-UAV, a dataset NEVER used for training or threshold choice (03 section 2.1 rule 3).

Each run is scored at its own operating threshold, the one experiments/e12_eval.py chose on Anti-UAV val, so nothing
here is tuned on DUT. Images larger than 640 px are tiled at true size; smaller ones are PADDED to 640 rather than
letting Ultralytics upscale them, so a 10-px drone stays 10 px (13 section 3.1).
DUT's detection images carry no video ids, so the 95% intervals resample images: narrower than a per-video interval
would be if the images come from a few videos. Check the file names on receipt.

    C:\\cuas_venv\\Scripts\\python -m cuas.datasets.dut --src <unzipped test folder> --dst C:/cuas_data/dut_test
    C:\\cuas_venv\\Scripts\\python experiments/e15_dut.py E12_multidata_yolo26s E14_hardneg_yolo26s
    ... e15_dut.py --out e15_e18.json E14_hardneg_yolo26s_L4 E18_full_s0   (default output: runs/e15_dut.json)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.detector import load as load_detector  # noqa: E402
from cuas.eval import (bootstrap_ci, evaluate, format_report, image_counts, iou, read_yolo_labels,  # noqa: E402
                       recall_precision, run_files, run_split, split_pairs, w_min)

D = Path(os.environ.get("CUAS_DATA", "C:/cuas_data"))  # CUAS_DATA overrides it (e.g. on Colab)
SPLIT = D / "dut_test"
TILE = 640


class Padded:
    """Pads images smaller than TILE to TILE x TILE (grey, bottom/right), so the detector never rescales them."""

    def __init__(self, det):
        self.det, self.names = det, det.names

    def __call__(self, images):
        out = []
        for img in images:
            h, w = img.shape[:2]
            if h < TILE or w < TILE:
                canvas = np.full((max(h, TILE), max(w, TILE), 3), 114, np.uint8)
                canvas[:h, :w] = img
                img = canvas
            out.append(img)
        return self.det(out)


SKY_T = 5.0   # ring texture below this = plain sky around the drone (threshold set by eye on a sorted crop sheet)


def ring_texture(gray, box, ring=1.0):
    """Median gradient magnitude in a ring one box-size wide around `box` (box excluded). Low = plain sky;
    high = buildings, trees, ground. DUT mixes both, while this project's camera looks at sky (13 §2.4)."""
    import cv2
    x, y, w, h = box[:4]
    H, W = gray.shape
    x0, y0 = max(0, int(x - ring * w)), max(0, int(y - ring * h))
    x1, y1 = min(W, int(x + w + ring * w)), min(H, int(y + h + ring * h))
    p = gray[y0:y1, x0:x1]
    m = np.ones(p.shape, bool)
    m[max(0, int(y) - y0):int(y + h) - y0, max(0, int(x) - x0):int(x + w) - x0] = False
    mag = np.hypot(cv2.Sobel(p, cv2.CV_32F, 1, 0), cv2.Sobel(p, cv2.CV_32F, 0, 1))
    return float(np.median(mag[m])) if m.any() else 0.0


def backgrounds():
    """'sky' / 'clutter' for every labelled drone, per image, in split order."""
    import cv2
    out = []
    for ip, lp in split_pairs(SPLIT):
        g = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = g.shape
        out.append(["sky" if ring_texture(g, b) < SKY_T else "clutter"
                    for b in read_yolo_labels(lp.read_text() if lp.exists() else "", w, h)])
    return out


def breakdown(images, bgs, names, iou_thr=0.3):
    """Each labelled drone as: found as drone / boxed as another aircraft class / missed, split by background.
    'Any class' recall separates "never localised" from "localised but named wrong" (the COCO-head problem, E11)."""
    t = {bg: {"drone": 0, "other": 0, "missed": 0} for bg in ("sky", "clutter")}
    for (gts, preds), bg_list in zip(images, bgs):
        drones = [g for g in gts if g[4] == 0]
        for g, bg in zip(drones, bg_list):
            hits = [p for p in preds if iou(p, g) >= iou_thr]
            t[bg]["drone" if any(p[4] == 0 for p in hits) else "other" if hits else "missed"] += 1
    lines = ["background  drones  as drone  as other class  missed   any-class recall"]
    for bg, c in t.items():
        n = sum(c.values()) or 1
        lines.append(f"{bg:10s} {sum(c.values()):7d}  {c['drone'] / n:8.2f}  {c['other'] / n:14.2f}  "
                     f"{c['missed'] / n:6.2f}   {(c['drone'] + c['other']) / n:.2f}")
    return t, "\n".join(lines)


def main(runs, out_name="e15_dut.json"):
    names = [p.name for p, _ in split_pairs(SPLIT)]
    bgs = backgrounds()
    summary = {}
    for run in runs:
        name, weights, sfx = run_files(run)              # "run" = best.pt, "run@last" = the final weights
        conf = json.loads((D / "runs" / name / f"e12_eval{sfx}.json").read_text())["conf"]
        det = Padded(load_detector(str(D / "runs" / name / "weights" / weights), conf=0.05, device=0))
        images = run_split(det, SPLIT, conf=conf, tile=TILE)
        r = evaluate(images)
        r["w_min_px"] = w_min(r)
        rows = [image_counts(g, p) + [1] for g, p in images]
        ci = {"recall": bootstrap_ci(names, rows, lambda s: recall_precision(s)[0]),
              "precision": bootstrap_ci(names, rows, lambda s: recall_precision(s)[1]),
              "background_per_1000_images": bootstrap_ci(names, rows, lambda s: 1000 * s[3] / s[4])}
        print(f"=== {run} on DUT Anti-UAV test ({len(images)} images), conf {conf} (chosen on Anti-UAV val) ===")
        print(format_report(r, det.names))
        print(f"w_min: {r['w_min_px']} px")
        print("95% CI (per image): " + ", ".join(f"{k} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for k, v in ci.items()))
        bd, table = breakdown(images, bgs, det.names)
        print(f"By background (sky = ring texture < {SKY_T}; this project's camera sees sky):\n{table}\n")
        summary[run] = {"conf": conf, "ci95": ci, "w_min_px": r["w_min_px"], "by_background": bd,
                        "recall_by_width": r["recall"],
                        "n_by_width": r["n_gt"], "background": r["false_drone_on_background"],
                        "on_drone": r["false_drone_on_drone"],
                        "boxed_as_other": {str(k): v for k, v in r["missed_drone_boxed_as_other"].items()}}
    out = D / "runs" / out_name
    out.write_text(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    out_name = "e15_dut.json"
    if "--out" in args:
        i = args.index("--out")
        out_name = args[i + 1]
        del args[i:i + 2]
    main(args or ["E12_multidata_yolo26s"], out_name)
