"""
E30: does calibrating detector scores per box size make one threshold work across sizes and datasets?

Detector confidence depends on box size (19 section 10.2), and here a threshold chosen on Anti-UAV validation
frames removes a quarter of Halmstad's drones (recall 0.95 at 0.40, 0.72 at 0.55). Two questions, both answered
from cached native detections (no inference):

  1. Where does the score gap sit: do correct "drone" boxes of the SAME width score lower on Halmstad than on
     Anti-UAV? If so the bias is the dataset, and size calibration cannot remove it.
  2. The test: fit score -> P(correct) per width bin on Anti-UAV VAL (isotonic), replace every score by it, choose
     one threshold on Anti-UAV val, and compare with the raw score at the same rule on Anti-UAV TEST and Halmstad.

    python experiments/e30_size_calibration.py E18_base_s1@last E18_base_s2@last E22a_s1@last E22a_s2@last   (CPU, ~1 min)

Output: C:/cuas_data/runs/e30_size_calibration.txt/.json.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.eval import evaluate, match_quality, run_files  # noqa: E402

D = Path(os.environ.get("CUAS_DATA", "C:/cuas_data"))
BINS = (0, 16, 32, 64, float("inf"))
GRID = [round(0.05 + 0.01 * k, 2) for k in range(91)]


def labelled(images):
    """(width, score, correct) for every 'drone' box: greedy by score, one box per labelled drone (as evaluate)."""
    rows = []
    for gts, preds in images:
        drones, used = [g for g in gts if g[4] == 0], set()
        for p in sorted((p for p in preds if p[4] == 0), key=lambda p: -p[5]):
            best, best_q = None, 0.0
            for i, g in enumerate(drones):
                q = 0.0 if i in used else match_quality(p, g)
                if q > best_q:
                    best, best_q = i, q
            used.add(best) if best is not None else None
            rows.append((p[2], p[5], best is not None))
    return np.array(rows, float).reshape(-1, 3)


def fit(rows):
    """One isotonic map per width bin (the global map where a bin has under 50 boxes or one outcome only)."""
    def iso(r):
        return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(r[:, 1], r[:, 2])
    whole, maps = iso(rows), []
    for lo, hi in zip(BINS, BINS[1:]):
        r = rows[(rows[:, 0] >= lo) & (rows[:, 0] < hi)]
        maps.append(iso(r) if len(r) >= 50 and 0 < r[:, 2].mean() < 1 else whole)
    return maps


def calibrate(images, maps):
    out = []
    for gts, preds in images:
        ps = []
        for p in preds:
            if p[4] == 0:
                b = int(np.searchsorted(BINS, p[2], side="right")) - 1
                p = list(p[:5]) + [float(maps[b].predict([p[5]])[0])]
            ps.append(p)
        out.append((gts, ps))
    return out


def at(images, t):
    r = evaluate([(g, [p for p in ps if p[5] >= t]) for g, ps in images])
    n, h = sum(r["n_gt"]), sum(r["n_hit"])
    fp = r["false_drone_on_drone"] + r["false_drone_on_background"] + sum(r["false_drone_on_distractor"].values())
    return h / max(n, 1), h / max(h + fp, 1), 1000 * r["false_drone_on_background"] / r["n_images"]


def choose(images, rule):
    """Threshold on validation: maximum F1, or the lowest threshold with <= 10 background false alarms per 1000."""
    best, best_v = GRID[-1], -1.0
    for t in GRID:
        rec, prec, fa = at(images, t)
        v = 2 * rec * prec / max(rec + prec, 1e-9) if rule == "max F1" else (rec if fa <= 10 else -1)
        if v > best_v:
            best, best_v = t, v
    return best


def main(runs):
    lines, out = ["E30: per-size calibration of 'drone' scores, fitted on Anti-UAV val, native frames."], {}
    for run in runs:
        name, _, sfx = run_files(run)
        d = json.loads((D / "runs" / name / f"dets_native{sfx}.json").read_text())
        sets = {k: [(g, p) for g, p in d[k]["images"]] for k in ("antiuav_val", "antiuav_test", "halmstad_val")}
        rows = {k: labelled(v) for k, v in sets.items()}
        lines.append(f"\n{run}\n  median score of CORRECT drone boxes by box width (count):")
        lines.append("  " + f"{'set':14s}" + "".join(f"{f'{lo:g}-{hi:g} px':>18s}" for lo, hi in zip(BINS, BINS[1:])))
        med = {}
        for k, r in rows.items():
            cells = []
            for lo, hi in zip(BINS, BINS[1:]):
                s = r[(r[:, 0] >= lo) & (r[:, 0] < hi) & (r[:, 2] == 1), 1]
                cells.append(f"{np.median(s):.2f} ({len(s)})" if len(s) else "-")
                med.setdefault(k, []).append(float(np.median(s)) if len(s) else None)
            lines.append("  " + f"{k:14s}" + "".join(f"{c:>18s}" for c in cells))
        maps = fit(rows["antiuav_val"])
        cal = {k: calibrate(v, maps) for k, v in sets.items()}
        lines.append("  threshold chosen on Anti-UAV val -> recall / precision / background FAs per 1000 frames:")
        res = {}
        for rule in ("max F1", "<= 10 FAs per 1000"):
            for label, data in (("raw score", sets), ("per-size calibrated", cal)):
                t = choose(data["antiuav_val"], rule)
                cells = {k: at(data[k], t) for k in ("antiuav_test", "halmstad_val")}
                res[f"{rule} | {label}"] = {"threshold": t, **{k: list(v) for k, v in cells.items()}}
                lines.append(f"    {rule:20s} {label:20s} t = {t:.2f}   test "
                             + " / ".join(f"{x:.3f}" if i < 2 else f"{x:.1f}" for i, x in enumerate(cells["antiuav_test"]))
                             + "   Halmstad "
                             + " / ".join(f"{x:.3f}" if i < 2 else f"{x:.1f}" for i, x in enumerate(cells["halmstad_val"])))
        t_h = choose(sets["halmstad_val"], "max F1")
        lines.append(f"    for reference, the raw threshold that is best ON Halmstad: {t_h:.2f} -> Halmstad "
                     + " / ".join(f"{x:.3f}" if i < 2 else f"{x:.1f}" for i, x in enumerate(at(sets["halmstad_val"], t_h))))
        out[run] = {"median_correct_score": med, "results": res, "halmstad_own_threshold": t_h}
        print("\n".join(lines[-12:]), flush=True)
    (D / "runs" / "e30_size_calibration.txt").write_text("\n".join(lines))
    (D / "runs" / "e30_size_calibration.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["E18_base_s1@last"])
