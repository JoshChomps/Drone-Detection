"""
Paired comparison of two evaluated runs on the same native frames (e.g. E14 vs E12), with 95% per-video bootstrap
intervals of the DIFFERENCE. Paired resampling (the same videos for both runs in every resample) cancels most of
the video-to-video variation, so it detects smaller real differences than comparing two separate intervals.

Default: each run at its own operating threshold, the one experiments/e12_eval.py chose on validation. Run that
script on both runs first; it writes <run>/e12_eval.json and <run>/dets_native.json.
CAUTION (16 #50): own thresholds compare operating points, not models. E14's val-chosen threshold moved from 0.40
to 0.55, and at E12's 0.70 E12 matched E14 on test. Judge a model change with --match (B at A's native val recall,
the threshold chosen on val) and look at --sweep.

    C:\\cuas_venv\\Scripts\\python experiments/compare_runs.py E12_multidata_yolo26s E14_hardneg_yolo26s
    ... compare_runs.py --match A B    (B at the threshold giving A's native val recall; the fair model comparison)
    ... compare_runs.py --conf 0.4 A B (both at one threshold)      ... compare_runs.py --sweep A B (both at 0.1-0.7)
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.eval import bootstrap_ci, image_counts, recall_precision, run_files, video_of  # noqa: E402

D = Path(os.environ.get("CUAS_DATA", "C:/cuas_data")) / "runs"  # CUAS_DATA overrides it (e.g. on Colab)


def load(run):
    """`run` is a name (its best.pt results) or name@last (its final weights; e12_eval.py name@last first)."""
    name, _, sfx = run_files(run)
    conf = json.loads((D / name / f"e12_eval{sfx}.json").read_text())["conf"]
    return conf, json.loads((D / name / f"dets_native{sfx}.json").read_text())


def sweep(a, b, confs=(0.1, 0.25, 0.4, 0.55, 0.7)):
    """Both runs at the same thresholds, native frames, no intervals: shows whether a difference at the runs' own
    thresholds is the model or just where the val-chosen threshold landed."""
    (_, da), (_, db) = load(a), load(b)
    for name in da:
        print(f"\n{name} native (recall / precision / background FAs per 1000 frames)")
        print(f"   {'conf':>5}   {a[:28]:>28}   {b[:28]:>28}")
        for c in confs:
            cells = []
            for d in (da, db):
                tot = [sum(col) for col in zip(*[image_counts(g, [p for p in ps if p[5] >= c]) + [1]
                                                  for g, ps in d[name]["images"]])]
                r, p = recall_precision(tot[0:4])
                cells.append(f"{r:.3f} / {p:.3f} / {1000 * tot[3] / tot[4]:6.1f}")
            print(f"   {c:5.2f}   {cells[0]:>28}   {cells[1]:>28}")


def native_recall(d, set_name, c):
    tot = [sum(col) for col in zip(*[image_counts(g, [p for p in ps if p[5] >= c]) for g, ps in d[set_name]["images"]])]
    return recall_precision(tot)[0]


def matched_conf(d, target, set_name="antiuav_val"):
    """Highest threshold (0.01 grid) at which the run's native VAL recall still reaches `target`: the most precise
    operating point at the same recall. Chosen on validation only, so the test comparison stays clean."""
    ok = [c / 100 for c in range(5, 96) if native_recall(d, set_name, c / 100) >= target]
    return max(ok) if ok else 0.05


def main(a, b, conf=None, match=False):
    (ca, da), (cb, db) = load(a), load(b)
    if conf is not None:                                  # same threshold for both, instead of each run's own
        ca = cb = conf
    if match:                                             # B at the threshold giving A's native val recall
        target = native_recall(da, "antiuav_val", ca)
        cb = matched_conf(db, target)
        print(f"matched recall: A's native Anti-UAV val recall at {ca} is {target:.3f}; "
              f"B reaches it up to conf {cb:.2f} (val recall {native_recall(db, 'antiuav_val', cb):.3f})")
    print(f"A = {a} @ conf {ca}   B = {b} @ conf {cb}   (differences are B - A, 95% per-video bootstrap)")
    for name in da:
        assert da[name]["paths"] == db[name]["paths"], name
        keys = [video_of(p) for p in da[name]["paths"]]
        rows = []
        for (g, pa), (_, pb) in zip(da[name]["images"], db[name]["images"]):
            rows.append(image_counts(g, [p for p in pa if p[5] >= ca]) + image_counts(g, [p for p in pb if p[5] >= cb]) + [1])
        stats = {
            "recall": lambda s: recall_precision(s[4:8])[0] - recall_precision(s[0:4])[0],
            "precision": lambda s: recall_precision(s[4:8])[1] - recall_precision(s[0:4])[1],
            "background FAs per 1000 frames": lambda s: 1000 * (s[7] - s[3]) / s[8],
        }
        tot = [sum(col) for col in zip(*rows)]
        ra, pa_ = recall_precision(tot[0:4])
        rb, pb_ = recall_precision(tot[4:8])
        print(f"\n{name} native: {len(rows)} frames, {len(set(keys))} videos")
        print(f"   A: recall {ra:.3f} precision {pa_:.3f} background {tot[3]}   "
              f"B: recall {rb:.3f} precision {pb_:.3f} background {tot[7]}")
        for label, f in stats.items():
            d, lo, hi = bootstrap_ci(keys, rows, f)
            higher_is_better = not label.startswith("background")
            verdict = "no clear difference" if lo <= 0 <= hi else \
                      "B better" if (d > 0) == higher_is_better else "A better"
            print(f"   {label:32s} {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--sweep" in args:
        args.remove("--sweep")
        sweep(args[0], args[1])
    elif "--match" in args:
        args.remove("--match")
        main(args[0], args[1], match=True)
    elif "--conf" in args:
        i = args.index("--conf")
        c = float(args[i + 1])
        del args[i:i + 2]
        main(args[0], args[1], c)
    else:
        main(args[0], args[1])
