"""
E12 evaluation: did multi-dataset training fix the drone class?

1. Operating threshold chosen on VALIDATION only: the confidence that maximises F1 on pooled Anti-UAV val
   (native + zoom-out), where precision counts every false "drone" (on a real drone, on a distractor, on background).
2. At that threshold: Anti-UAV val and the held-out Anti-UAV TEST (scored once), plus Halmstad val (regression
   check against E2: it must not get worse on the original data).
Each set's detections are computed once at conf 0.05 and filtered per threshold, so the sweep is cheap.
3. 95% confidence intervals on the NATIVE frames by per-video (cluster) bootstrap: frames of a video are strongly
   correlated, so the videos, not the frames, are the samples. Zoom mosaics mix several videos, so they get no interval.
   Native detections are cached in <run>/dets_native.json for paired comparisons (experiments/compare_runs.py).

    C:\\cuas_venv\\Scripts\\python experiments/e12_eval.py [run_name]
    ... e12_eval.py run_name@last     (the run's FINAL weights instead of best.pt; writes e12_eval_last.json and
                                       dets_native_last.json, and every other script accepts the same spec)
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.detector import load as load_detector  # noqa: E402
from cuas.eval import (bootstrap_ci, evaluate, format_report, image_counts, recall_precision,  # noqa: E402
                       run_files, run_split, split_pairs, video_of, w_min)

D = Path(os.environ.get("CUAS_DATA", "C:/cuas_data"))  # CUAS_DATA overrides it (e.g. on Colab)
RUN = sys.argv[1] if len(sys.argv) > 1 else "E12_multidata_yolo26s"
NAME, WEIGHTS, SFX = run_files(RUN)
CONFS = (0.1, 0.25, 0.4, 0.55, 0.7)
SETS = {
    "antiuav_val": [D / "antiuav_val"] + [D / "antiuav_zoom_val" / f"k{k}" for k in (4, 8, 12, 16)],
    "antiuav_test": [D / "antiuav_test"] + [D / "antiuav_zoom_test" / f"k{k}" for k in (4, 8, 12, 16)],
    "halmstad_val": [D / "halmstad_yolo" / "val"] + [D / "halmstad_zoom" / f"k{k}" for k in (2, 3, 4, 6)],
}


def at(per_image, conf):
    return [(g, [p for p in ps if p[5] >= conf]) for g, ps in per_image]


def f1(r):
    hits = sum(r["n_hit"])
    false = r["false_drone_on_drone"] + sum(r["false_drone_on_distractor"].values()) + r["false_drone_on_background"]
    rec, prec = hits / max(sum(r["n_gt"]), 1), hits / max(hits + false, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9), prec, rec


def main():
    det = load_detector(str(D / "runs" / NAME / "weights" / WEIGHTS), conf=0.05, device=0)
    raw = {}
    for name, dirs in SETS.items():
        raw[name] = {d.name if d.name.startswith("k") else "native": run_split(det, d, conf=0.05, tile=640)
                     for d in dirs}
        print(f"inferred {name}: {sum(len(v) for v in raw[name].values())} images", flush=True)

    pooled = {n: [x for v in sets.values() for x in v] for n, sets in raw.items()}
    print("\nThreshold sweep on Anti-UAV VAL (pooled native + k4..k16):")
    best = None
    for c in CONFS:
        score, prec, rec = f1(evaluate(at(pooled["antiuav_val"], c)))
        print(f"  conf {c:.2f}: recall {rec:.3f}  precision {prec:.3f}  F1 {score:.3f}")
        if best is None or score > best[0]:
            best = (score, c)
    conf = best[1]
    print(f"-> operating threshold chosen on val: conf {conf}\n")

    summary, cache = {"run": RUN, "conf": conf}, {}
    for name in ("antiuav_val", "antiuav_test", "halmstad_val"):
        r = evaluate(at(pooled[name], conf))
        r["w_min_px"] = w_min(r)
        score, prec, rec = f1(r)
        print(f"=== {RUN} on {name} (pooled), conf {conf}: recall {rec:.3f}, precision {prec:.3f}, F1 {score:.3f} ===")
        print(format_report(r, det.names))
        print(f"w_min: {r['w_min_px']} px")
        for sub, per_image in raw[name].items():
            rs = evaluate(at(per_image, conf))
            n, h = sum(rs["n_gt"]), sum(rs["n_hit"])
            print(f"   {sub:6s} drones {n:5d} recall {h / max(n, 1):.3f} on-drone {rs['false_drone_on_drone']:4d} "
                  f"background {rs['false_drone_on_background']:4d} boxed-as-other {rs['missed_drone_boxed_as_other']}")
        native = raw[name]["native"]
        paths = [ip.name for ip, _ in split_pairs(SETS[name][0])]
        cache[name] = {"paths": paths, "images": [[g, [[float(v) for v in q] for q in ps]] for g, ps in native]}
        keys, rows = [video_of(q) for q in paths], [image_counts(g, p) + [1] for g, p in at(native, conf)]
        ci = {"recall": bootstrap_ci(keys, rows, lambda s: recall_precision(s)[0]),
              "precision": bootstrap_ci(keys, rows, lambda s: recall_precision(s)[1]),
              "background_per_1000_frames": bootstrap_ci(keys, rows, lambda s: 1000 * s[3] / s[4])}
        print(f"   native, 95% CI over {len(set(keys))} videos: " +
              ", ".join(f"{k} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for k, v in ci.items()))
        print()
        summary[name] = {"native_ci95": ci, "recall": rec, "precision": prec, "f1": score, "w_min_px": r["w_min_px"],
                         "recall_by_width": r["recall"], "n_by_width": r["n_gt"],
                         "background": r["false_drone_on_background"], "on_drone": r["false_drone_on_drone"],
                         "distractor": {str(k): v for k, v in r["false_drone_on_distractor"].items()},
                         "boxed_as_other": {str(k): v for k, v in r["missed_drone_boxed_as_other"].items()}}
    (D / "runs" / NAME / f"dets_native{SFX}.json").write_text(json.dumps(cache))
    out = D / "runs" / NAME / f"e12_eval{SFX}.json"
    out.write_text(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
