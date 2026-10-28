"""
E35: the scores this project had not been computing, from cached detections (no inference).

The standard evaluation reports recall, precision and false alarms at ONE threshold per model, chosen by maximum F1
on validation. Two models then sit at different thresholds (0.40 against 0.55), which has confused several
comparisons. This adds, per run and per set:

  * AP: area under the precision-recall curve of 'drone' boxes (IoU >= 0.3, boxes with score >= 0.05), which needs
    no threshold;
  * recall at a fixed false-alarm budget (10 and 30 background false drones per 1000 frames), read off the curve;
  * box error of correct detections at the run's threshold: width bias, MAE and RMSE in px and in % of the true
    width, and centre error. Range (09) and time to collision are computed from box width, so this is the error
    they inherit;
  * calibration of the raw score as "probability the box is a drone": ECE (10 bins) and Brier score.

    python experiments/e35_metrics.py E18_base_s1@last E22a_s1@last ...      (CPU, ~1 min)
    python experiments/e35_metrics.py --dut E22a_s0@last E22c_s0@last ...    (adds DUT; infers it once per run)

Sets: Anti-UAV test (all, and its daytime videos only: night is out of scope) and Halmstad val, native frames; DUT test (never trained on) where <run>/dets_dut<sfx>.json exists,
or with --dut, which runs the detector on DUT once and caches it (GPU if present, else CPU ~15 min a run); plus the
small-target sets where <run>/dets_small<sfx>.json exists (e33_center_match.py), for box error only.
Output: C:/cuas_data/runs/e35_metrics.txt/.json (e35_<run>.txt/.json for one run, e35_metrics_dut.* with --dut,
e35_<NAME>.* with --out NAME). Cached DUT detections are scored even without --dut.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuas.eval import evaluate, match_quality, run_files, split_pairs, video_of  # noqa: E402

D = Path(os.environ.get("CUAS_DATA", "C:/cuas_data"))
BINS = (0, 8, 16, 32, 64, float("inf"))
GRID = [round(0.05 + 0.01 * k, 2) for k in range(91)]


def pairs(images):
    """Every 'drone' box as (score, matched ground truth or None), greedy by score; and the number of drones."""
    rows, n_gt = [], 0
    for gts, preds in images:
        drones, used = [g for g in gts if g[4] == 0], set()
        n_gt += len(drones)
        for p in sorted((p for p in preds if p[4] == 0), key=lambda p: -p[5]):
            best, best_q = None, 0.0
            for i, g in enumerate(drones):
                q = 0.0 if i in used else match_quality(p, g)
                if q > best_q:
                    best, best_q = i, q
            if best is not None:
                used.add(best)
            rows.append((p, drones[best] if best is not None else None))
    return rows, n_gt


def average_precision(rows, n_gt):
    order = sorted(rows, key=lambda r: -r[0][5])
    tp = np.cumsum([r[1] is not None for r in order])
    prec, rec = tp / np.arange(1, len(order) + 1), tp / max(n_gt, 1)
    prec = np.maximum.accumulate(prec[::-1])[::-1]                    # precision envelope
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def recall_at_budget(images, budget):
    best = 0.0
    for t in GRID:
        r = evaluate([(g, [p for p in ps if p[5] >= t]) for g, ps in images])
        if 1000 * r["false_drone_on_background"] / r["n_images"] <= budget:
            best = max(best, sum(r["n_hit"]) / max(sum(r["n_gt"]), 1))
    return best


def calibration(rows, n_bins=10):
    s = np.array([r[0][5] for r in rows])
    y = np.array([r[1] is not None for r in rows], float)
    ece = sum(abs(y[m].mean() - s[m].mean()) * m.mean()
              for lo in np.linspace(0, 1, n_bins + 1)[:-1] for m in [(s >= lo) & (s < lo + 1 / n_bins)] if m.any())
    return float(ece), float(np.mean((s - y) ** 2))


def box_error(rows, conf):
    out = []
    for lo, hi in zip(BINS, BINS[1:]):
        e = np.array([(p[2] - g[2], (p[2] - g[2]) / g[2],
                       np.hypot(p[0] + p[2] / 2 - g[0] - g[2] / 2, p[1] + p[3] / 2 - g[1] - g[3] / 2))
                      for p, g in rows if g is not None and p[5] >= conf and lo <= g[2] < hi])
        out.append(None if len(e) < 20 else {
            "n": len(e), "bias_px": float(e[:, 0].mean()), "mae_px": float(np.abs(e[:, 0]).mean()),
            "rmse_px": float(np.sqrt((e[:, 0] ** 2).mean())), "bias_pct": float(100 * e[:, 1].mean()),
            "mae_pct": float(100 * np.abs(e[:, 1]).mean()), "centre_px": float(e[:, 2].mean())})
    return out


def day_videos(threshold: float = 50.0):
    """Anti-UAV TEST videos filmed by day. Night is out of scope (13 section 6), and night lights were 21-40% of the
    test's confident false drones (E36). Median frame brightness is bimodal (night 2-9, day 90-147 of 255), so the
    split is unambiguous; cached in runs/antiuav_test_brightness.json."""
    f = D / "runs" / "antiuav_test_brightness.json"
    if not f.exists():
        import cv2
        b = {}
        for ip, _ in split_pairs(D / "antiuav_test"):
            g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
            b.setdefault(video_of(ip), []).append(float(cv2.resize(g, (96, 54)).mean()))
        f.write_text(json.dumps({k: float(np.median(v)) for k, v in b.items()}, indent=0))
    return {k for k, v in json.loads(f.read_text()).items() if v >= threshold}


def dut(run, infer):
    """DUT test images [(gts, preds at conf >= 0.05)] from the cache, inferring them first if `infer`; else None."""
    name, _, sfx = run_files(run)
    if not infer and not (D / "runs" / name / f"dets_dut{sfx}.json").exists():
        return None
    import torch
    from e15_dut import SPLIT, Padded
    from e33_center_match import cached
    device = "0" if torch.cuda.is_available() else "cpu"
    return cached(run, "dut", {"dut_test": SPLIT}, device=device, wrap=Padded)["dut_test"]


def main(runs, infer_dut=False, out_key=None):
    lines, out = ["E35: threshold-free scores, box error and score calibration ('drone' boxes, IoU >= 0.3)."], {}
    head = f"{'run':20s} {'set':13s} {'AP':>6s} {'R@10':>6s} {'R@30':>6s} {'ECE':>6s} {'Brier':>6s}"
    lines += ["R@10 / R@30: drone recall with at most 10 / 30 background false drones per 1000 frames.", head]
    errs = []
    for run in runs:
        name, _, sfx = run_files(run)
        ev = D / "runs" / name / f"e12_eval{sfx}.json"      # its threshold; only the box-error table needs it
        conf = json.loads(ev.read_text())["conf"] if ev.exists() else None
        d = json.loads((D / "runs" / name / f"dets_native{sfx}.json").read_text())
        sets = {k: [(g, p) for g, p in d[k]["images"]] for k in ("antiuav_test", "halmstad_val")}
        pooled = [x for v in sets.values() for x in v]            # box error: the trained datasets only
        day = day_videos()
        sets["antiuav_day"] = [x for path, x in zip(d["antiuav_test"]["paths"], sets["antiuav_test"])
                               if video_of(path) in day]
        dut_images = dut(run, infer_dut)
        if dut_images is not None:
            sets["dut_test"] = dut_images
        small = D / "runs" / name / f"dets_small{sfx}.json"
        if small.exists():
            pooled += [(g, p) for v in json.loads(small.read_text()).values() for g, p in v]
        out[run] = {"conf": conf}
        for k, images in sets.items():
            rows, n_gt = pairs(images)
            ece, brier = calibration(rows)
            res = {"AP": average_precision(rows, n_gt), "R@10": recall_at_budget(images, 10),
                   "R@30": recall_at_budget(images, 30), "ECE": ece, "Brier": brier}
            out[run][k] = res
            lines.append(f"{run:20s} {k:13s} " + " ".join(f"{res[m]:6.3f}" for m in ("AP", "R@10", "R@30", "ECE", "Brier")))
        be = box_error(pairs(pooled)[0], conf) if conf is not None else []
        out[run]["box_error"] = be
        for (lo, hi), b in zip(zip(BINS, BINS[1:]), be):
            if b:
                errs.append(f"{run:20s} {f'{lo:g}-{hi:g}':>8s} {b['n']:6d} {b['bias_px']:+8.2f} {b['mae_px']:7.2f} "
                            f"{b['rmse_px']:7.2f} {b['bias_pct']:+8.1f} {b['mae_pct']:7.1f} {b['centre_px']:9.2f}")
    lines += ["", "Box error of correct detections at each run's threshold (native sets, plus the small-target sets "
              "where cached). Width error = predicted - true.",
              f"{'run':20s} {'true px':>8s} {'n':>6s} {'bias px':>8s} {'MAE px':>7s} {'RMSE px':>7s} {'bias %':>8s} "
              f"{'MAE %':>7s} {'centre px':>9s}"] + errs
    text = "\n".join(lines)
    key = runs[0].replace("@", "_") if len(runs) == 1 else "metrics"      # one run: e35_<run>.txt (the Colab queue)
    key += "_dut" if infer_dut and len(runs) > 1 else ""
    key = out_key or key                                                 # --out NAME: e35_NAME.txt/.json
    (D / "runs" / f"e35_{key}.txt").write_text(text)
    (D / "runs" / f"e35_{key}.json").write_text(json.dumps(out, indent=1))
    print(text)


if __name__ == "__main__":
    args = sys.argv[1:]
    out_key = None
    if "--out" in args:
        i = args.index("--out")
        out_key = args[i + 1]
        del args[i:i + 2]
    main([a for a in args if a != "--dut"] or ["E18_base_s1@last"], infer_dut="--dut" in args, out_key=out_key)
