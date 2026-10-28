"""
E11: do the Halmstad-trained detectors (E1, E2) work on a different dataset (Anti-UAV300 visible)?

  E11a  native size: new camera, site and drones (median width ~110-120 px), incl. explicit target-absent frames
  E11b  simulated distance on UNSEEN footage: Anti-UAV frames shrunk k-fold into k x k mosaics, so drones drop to
        ~29 / 14 / 10 / 7 px at k = 4 / 8 / 12 / 16 (median). Tests whether E2's small-target ability is general
        or tied to Halmstad's mosaics.

Full HD frames are evaluated as 640-px tiles (run_split(..., tile=640)); nothing is downscaled.
Runs locally on the GPU venv (14 section 3); results go to C:/cuas_data/runs/E11/.

    C:\\cuas_venv\\Scripts\\python experiments/e11_antiuav.py [val] [test]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.detector import UltralyticsDetector  # noqa: E402
from cuas.eval import evaluate, format_report, run_split, w_min  # noqa: E402
from cuas.zoomout import make_split  # noqa: E402

DATA = Path("C:/cuas_data")
OUT = DATA / "runs" / "E11"
MODELS = {"E1": ("E1_halmstad_yolo26s", 0.25), "E2": ("E2_zoomtrain_yolo26s", 0.4)}   # each at its operating point
KS = (4, 8, 12, 16)


def zoom_sets(split):
    root = DATA / f"antiuav_zoom_{split}"
    for k in KS:
        if not (root / f"k{k}" / "images").exists():
            n = make_split(DATA / f"antiuav_{split}", root / f"k{k}", k)
            print(f"  built {split} k={k}: {n} mosaics")
    return {f"k{k}": root / f"k{k}" for k in KS}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in (sys.argv[1:] or ["val", "test"]):
        sets = {"native": DATA / f"antiuav_{split}", **zoom_sets(split)}
        for tag, (run, conf) in MODELS.items():
            det = UltralyticsDetector(str(DATA / "runs" / run / "weights" / "best.pt"), conf=0.05, device=0)
            pooled, t0, n_img = [], time.perf_counter(), 0
            for name, path in sets.items():
                per_image = run_split(det, path, conf=conf, tile=640)
                n_img += len(per_image)
                r = evaluate(per_image)
                n, hit = sum(r["n_gt"]), sum(r["n_hit"])
                summary[f"{tag}/{split}/{name}"] = {
                    "conf": conf, "images": r["n_images"], "drones": n, "recall": hit / max(n, 1),
                    "on_drone": r["false_drone_on_drone"], "background": r["false_drone_on_background"],
                    "missed_boxed_as_other": {str(k): v for k, v in r["missed_drone_boxed_as_other"].items()},
                    "recall_by_width": r["recall"], "n_by_width": r["n_gt"],
                    "bg_by_width": r["false_drone_background_by_width"]}
                print(f"{tag} {split:4s} {name:6s} images {r['n_images']:5d} drones {n:5d} recall {hit / max(n, 1):.3f} "
                      f"on-drone {r['false_drone_on_drone']:4d} background {r['false_drone_on_background']:4d} "
                      f"boxed-as-other {r['missed_drone_boxed_as_other']}")
                pooled += per_image
            r = evaluate(pooled)
            r["w_min_px"] = w_min(r)
            secs = time.perf_counter() - t0
            print(f"=== {tag} ({run}) on Anti-UAV {split}, conf {conf}, native + k{KS}: "
                  f"{secs:.0f} s for {n_img} images ===")
            print(format_report(r, det.names))
            print(f"w_min: {r['w_min_px']} px\n")
            summary[f"{tag}/{split}/pooled"] = {"conf": conf, "w_min_px": r["w_min_px"], "seconds": secs, **{
                k: r[k] for k in ("bins", "n_gt", "n_hit", "recall", "false_drone_on_drone",
                                  "false_drone_on_background", "false_drone_background_by_width")},
                "false_drone_on_distractor": {str(k): v for k, v in r["false_drone_on_distractor"].items()}}
    name = "e11_summary_" + "_".join(sys.argv[1:] or ["val", "test"]) + ".json"
    (OUT / name).write_text(json.dumps(summary, indent=2))
    print("wrote", OUT / name)


if __name__ == "__main__":
    main()
