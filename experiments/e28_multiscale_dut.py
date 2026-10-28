"""
E28: does a second, coarser look at the frame find the large, close drones that native-size tiles miss?

Native 640-px tiles keep a distant drone at its true pixel size (13 section 3.1), but a close drone can be hundreds
of pixels wide: larger than anything in training, and cut by tile edges. E15 found recall of only 0.06-0.12 for
DUT drones wider than 64 px. The 2025 Drone-vs-Bird winner ran the full frame as well as slices (19 section 9).
Here every DUT test image is run three ways with the SAME weights and threshold, and the detections are merged:

    tiles    native-size 640 tiles (what e15_dut.py and the program do now)
    half     the image shrunk 2x, then tiled
    coarse   the whole image shrunk to fit one 640 tile

No training. A close drone is the most urgent one, so this is a system question, not only a benchmark one.

    C:\\cuas_venv\\Scripts\\python experiments/e28_multiscale_dut.py [run or run@last]     (~10 min, local GPU)
"""
import json
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuas.detector import load as load_detector, detect_in_crops, nms  # noqa: E402
from cuas.eval import evaluate, read_yolo_labels, run_files, split_pairs  # noqa: E402
from cuas.tiling import grid_tiles  # noqa: E402
from e15_dut import SPLIT, TILE, Padded  # noqa: E402

D = Path(os.environ.get("CUAS_DATA", "C:/cuas_data"))
BINS = (0, 16, 32, 64, 128, 256, float("inf"))


def at_scale(img, det, s):
    """Detections on the image shrunk by factor s (tiled if still larger than a tile), in original pixels."""
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    h, w = small.shape[:2]
    dets = detect_in_crops(small, grid_tiles(w, h, TILE), det, TILE) if (w > TILE or h > TILE) else det([small])[0]
    return [(x / s, y / s, bw / s, bh / s, c, sc) for x, y, bw, bh, c, sc in dets]


def main(run):
    name, weights, sfx = run_files(run)
    conf = json.loads((D / "runs" / name / f"e12_eval{sfx}.json").read_text())["conf"]
    det = Padded(load_detector(str(D / "runs" / name / "weights" / weights), conf=0.05, device=0))
    per = {"tiles": [], "half": [], "coarse": []}
    gts_all = []
    for ip, lp in split_pairs(SPLIT):
        img = cv2.imread(str(ip))
        h, w = img.shape[:2]
        gts_all.append(read_yolo_labels(lp.read_text() if lp.exists() else "", w, h))
        coarse = min(1.0, TILE / max(w, h))
        per["tiles"].append([d for d in at_scale(img, det, 1.0) if d[5] >= conf])
        per["half"].append([d for d in at_scale(img, det, 0.5) if d[5] >= conf] if coarse < 0.5 else [])
        per["coarse"].append([d for d in at_scale(img, det, coarse) if d[5] >= conf] if coarse < 1 else [])
    variants = {"tiles (now)": ("tiles",), "coarse only": ("coarse",), "tiles + coarse": ("tiles", "coarse"),
                "tiles + half + coarse": ("tiles", "half", "coarse")}
    head = "".join(f"{f'{lo:g}-{hi:g}' if hi != float('inf') else f'{lo:g}+':>9s}" for lo, hi in zip(BINS, BINS[1:]))
    lines = [f"E28 multi-scale inference on DUT test: {run} at conf {conf} ({len(gts_all)} images). Drone recall by box "
             f"width (px), then false 'drone' boxes.", f"{'variant':24s} {'all':>6s}{head}   on-drone  background"]
    out = {}
    for label, keys in variants.items():
        images = [(g, nms([d for k in keys for d in per[k][i]])) for i, g in enumerate(gts_all)]
        r = evaluate(images, bins=BINS)
        rec = sum(r["n_hit"]) / max(sum(r["n_gt"]), 1)
        cells = "".join(f"{(h / n if n else float('nan')):9.2f}" for h, n in zip(r["n_hit"], r["n_gt"]))
        lines.append(f"{label:24s} {rec:6.3f}{cells}   {r['false_drone_on_drone']:8d}  {r['false_drone_on_background']:10d}")
        out[label] = {"recall": rec, "recall_by_width": r["recall"], "on_drone": r["false_drone_on_drone"],
                      "background": r["false_drone_on_background"],
                      "boxed_as_other": {str(k): v for k, v in r["missed_drone_boxed_as_other"].items()}}
    lines.append(f"{'drones per bin':24s} {sum(r['n_gt']):6d}" + "".join(f"{n:9d}" for n in r["n_gt"]))
    text = "\n".join(lines)
    key = run.replace("@", "_")
    (D / "runs" / f"e28_{key}.txt").write_text(text)
    (D / "runs" / f"e28_{key}.json").write_text(json.dumps({"run": run, "conf": conf, "results": out}, indent=1))
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "E12_multidata_yolo26s")
