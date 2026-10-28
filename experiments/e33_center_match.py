"""
E33: is the small-target recall (and w_min) partly an artefact of matching by IoU at a few pixels?

The evaluation matches a "drone" box to a labelled drone at IoU >= 0.3. On a 6-px target a box one pixel off
loses 0.2-0.4 of IoU while still marking the right object; the small-object tracking challenge scores by centre
distance for this reason (19 section 10.2). Here the smallest evaluation sets are scored twice with the same
detections: by IoU, and with drones narrower than 12 px matched by centre (cuas.eval.match_quality).

    C:\\cuas_venv\\Scripts\\python experiments/e33_center_match.py E18_base_s1@last E22a_s1@last     (CPU, ~4 min a run)

Detections are cached in <run>/dets_small<sfx>.json. Output: C:/cuas_data/runs/e33_center_match.txt/.json.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.eval import evaluate, run_files, run_split, w_min  # noqa: E402

D = Path(os.environ.get("CUAS_DATA", "C:/cuas_data"))
SETS = {"halmstad k4": D / "halmstad_zoom" / "k4", "halmstad k6": D / "halmstad_zoom" / "k6",
        "antiuav test k8": D / "antiuav_zoom_test" / "k8", "antiuav test k12": D / "antiuav_zoom_test" / "k12",
        "antiuav test k16": D / "antiuav_zoom_test" / "k16"}
BINS = (0, 4, 8, 12, 16, 24, float("inf"))
CENTER_BELOW = 12


def cached(run, tag, sets, device="cpu", tile=640, wrap=None):
    """{set name: [(gts, preds at conf >= 0.05), ...]} for a run spec, from <run>/dets_<tag><sfx>.json if present."""
    name, weights, sfx = run_files(run)
    f = D / "runs" / name / f"dets_{tag}{sfx}.json"
    if f.exists():
        return {k: [(g, p) for g, p in v] for k, v in json.loads(f.read_text()).items()}
    from cuas.detector import load as load_detector
    det = load_detector(str(D / "runs" / name / "weights" / weights), conf=0.05, device=device)
    out = {k: run_split(wrap(det) if wrap else det, path, conf=0.05, tile=tile) for k, path in sets.items()}
    f.write_text(json.dumps(out))
    return out


def main(runs):
    lines, out = [f"E33: drone recall by width, IoU >= 0.3 against centre matching for drones under {CENTER_BELOW} px. "
                  "Same detections, each run at its val-chosen threshold. Small sets pooled.",
                  f"{'run':22s} {'rule':7s}" + "".join(f"{f'{lo:g}-{hi:g}':>9s}" for lo, hi in zip(BINS, BINS[1:]))
                  + "   w_min  on-drone  background"], {}
    for run in runs:
        name, _, sfx = run_files(run)
        conf = json.loads((D / "runs" / name / f"e12_eval{sfx}.json").read_text())["conf"]
        images = [(g, [p for p in ps if p[5] >= conf]) for v in cached(run, "small", SETS).values() for g, ps in v]
        for rule, below in (("IoU", 0), ("centre", CENTER_BELOW)):
            r = evaluate(images, bins=BINS, center_below=below)
            cells = "".join(f"{(h / n if n else float('nan')):9.2f}" for h, n in zip(r["n_hit"], r["n_gt"]))
            lines.append(f"{run:22s} {rule:7s}{cells}   {str(w_min(r)):>5s}  {r['false_drone_on_drone']:8d}  "
                         f"{r['false_drone_on_background']:10d}")
            out[f"{run} {rule}"] = {"conf": conf, "n_gt": r["n_gt"], "n_hit": r["n_hit"], "w_min": w_min(r),
                                    "on_drone": r["false_drone_on_drone"], "background": r["false_drone_on_background"]}
            print(lines[-1], flush=True)
    lines.append(f"{'drones per bin':30s}" + "".join(f"{n:9d}" for n in r["n_gt"]))
    (D / "runs" / "e33_center_match.txt").write_text("\n".join(lines))
    (D / "runs" / "e33_center_match.json").write_text(json.dumps(out, indent=1))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:] or ["E18_base_s1@last"])
