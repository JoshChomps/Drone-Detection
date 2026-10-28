"""
Detector evaluation for this project's two questions (docs 10 section 4.1 and 13 section 2):
  1. How small can a drone be and still be found?  -> drone recall binned by box WIDTH in px, and w_min.
  2. How often is something else called a drone?  -> false "drone" boxes, split into those on a labelled
     distractor (bird, airplane, helicopter) and those on empty background.

Width matches doc 13's w_px = f_x * S / R, so a measured w_min drops straight into tools/geometry_budget.py.

    python -m cuas.eval --weights runs/E1/weights/best.pt --split data/halmstad_yolo/val --conf 0.25
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from cuas.tiling import read_yolo_labels

WIDTH_BINS = (0, 4, 8, 12, 16, 24, 32, 64, float("inf"))


def iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def match_quality(p, g, iou_thr: float = 0.3, center_below: float = 0.0, pad: float = 2.0) -> float:
    """How well prediction p matches ground truth g: > 0 is a match, larger is better.

    Normally IoU >= iou_thr. For a ground truth narrower than `center_below` px the rule is the centre instead:
    p matches if its centre lies inside g grown by `pad` px, scored by centre distance. At a few pixels a box
    one pixel off loses 0.2-0.4 of IoU while still marking the right object, which is why the small-object
    tracking challenge scores by centre distance (01 section 2.10)."""
    if g[2] < center_below:
        cx, cy = p[0] + p[2] / 2, p[1] + p[3] / 2
        if g[0] - pad <= cx <= g[0] + g[2] + pad and g[1] - pad <= cy <= g[1] + g[3] + pad:
            return 1.0 / (1.0 + np.hypot(cx - (g[0] + g[2] / 2), cy - (g[1] + g[3] / 2)))
        return 0.0
    v = iou(p, g)
    return v if v >= iou_thr else 0.0


def evaluate(images, drone_cls: int = 0, iou_thr: float = 0.3, bins=WIDTH_BINS, center_below: float = 0.0) -> dict:
    """images: iterable of (gts, preds); gts are (x, y, w, h, cls), preds (x, y, w, h, cls, score), already
    filtered to the operating confidence. IoU 0.3, not 0.5, because 1 px of box error on a 6-px target
    already costs ~30% IoU. center_below: drones narrower than this are matched by centre (match_quality).

    Drone recall: each drone GT is hit by at most one drone prediction (greedy, highest score first). A drone
    missed as "drone" but boxed (IoU >= iou_thr) by another class is counted in missed_drone_boxed_as_other:
    the detector saw it and got the class wrong (E11: Anti-UAV drones called "airplane").
    Every drone prediction left over is a false alarm of one of three kinds, checked in this order:
      * on a drone: it touches a real drone (a duplicate box, or one too misplaced to count). A box-merging
        problem, not a "what isn't a drone" problem;
      * on a distractor: IoU >= iou_thr with a labelled bird / airplane / helicopter;
      * on background: nothing labelled there (empty sky, clutter, or an unlabelled object).
    """
    n_bins = len(bins) - 1
    n_gt, n_hit, fa_w = np.zeros(n_bins, int), np.zeros(n_bins, int), np.zeros(n_bins, int)
    fa_distractor, fa_background, fa_on_drone, n_images = {}, 0, 0, 0
    drone_as_other = {}
    for gts, preds in images:
        n_images += 1
        drone_gts = [g for g in gts if g[4] == drone_cls]
        others = [g for g in gts if g[4] != drone_cls]
        used = set()
        for p in sorted((p for p in preds if p[4] == drone_cls), key=lambda p: -p[5]):
            best, best_q = None, 0.0
            for i, g in enumerate(drone_gts):
                q = 0.0 if i in used else match_quality(p, g, iou_thr, center_below)
                if q > best_q:
                    best, best_q = i, q
            if best is not None:
                used.add(best)
                continue
            if any(iou(p, g) > 0 or match_quality(p, g, iou_thr, center_below) for g in drone_gts):
                fa_on_drone += 1
                continue
            hit = max(others, key=lambda g: iou(p, g), default=None)
            if hit is not None and iou(p, hit) >= iou_thr:
                fa_distractor[hit[4]] = fa_distractor.get(hit[4], 0) + 1
            else:
                fa_background += 1
                fa_w[int(np.searchsorted(bins, p[2], side="right")) - 1] += 1
        for i, g in enumerate(drone_gts):
            b = int(np.searchsorted(bins, g[2], side="right")) - 1
            n_gt[b] += 1
            n_hit[b] += i in used
            if i not in used:        # missed as a drone: was it boxed as something else?
                other = max((p for p in preds if p[4] != drone_cls and iou(p, g) >= iou_thr),
                            key=lambda p: p[5], default=None)
                if other is not None:
                    drone_as_other[other[4]] = drone_as_other.get(other[4], 0) + 1
    return {
        "bins": [[bins[i], bins[i + 1]] for i in range(n_bins)],
        "n_gt": n_gt.tolist(), "n_hit": n_hit.tolist(),
        "recall": [h / n if n else None for h, n in zip(n_hit, n_gt)],
        "false_drone_on_drone": fa_on_drone,
        "missed_drone_boxed_as_other": drone_as_other,     # {class: count}: seen, but called e.g. airplane
        "false_drone_on_distractor": fa_distractor, "false_drone_on_background": fa_background,
        "false_drone_background_by_width": fa_w.tolist(),       # binned by PREDICTED box width
        "n_images": n_images,
    }


def w_min(result: dict, target: float = 0.8, min_count: int = 20):
    """Smallest bin lower edge from which every populated bin (>= min_count drones) has recall >= target.
    None if even the largest populated bin misses the target."""
    edge = None
    for (lo, _), n, r in reversed(list(zip(result["bins"], result["n_gt"], result["recall"]))):
        if n < min_count:
            continue
        if r < target:
            break
        edge = lo
    return edge


def run_files(spec: str) -> tuple:
    """(run folder, weights file, suffix of its result files) for a run spec. 'E14' is the run's best.pt, which
    Ultralytics picks by validation mAP; 'E14@last' is its final weights, with results kept apart in
    e12_eval_last.json / dets_native_last.json. The two differ by as much as the experiments do (19 section 3.4)."""
    name, _, which = spec.partition("@")
    if which not in ("", "best", "last"):
        raise ValueError(f"run spec {spec!r}: use NAME or NAME@last")
    return name, ("last.pt" if which == "last" else "best.pt"), ("_last" if which == "last" else "")


def video_of(path) -> str:
    """Source video of a frame file: Halmstad 'V_DRONE_001_00040_0_0' -> 'V_DRONE_001';
    Anti-UAV '20190925_111757_1_10_00040' -> '20190925_111757_1_10'."""
    stem = Path(path).stem
    m = re.match(r"(V_[A-Z]+_\d+)_", stem)
    return m.group(1) if m else stem.rsplit("_", 1)[0]


def image_counts(gts, preds, **kw) -> list:
    """[drones, drones found, false 'drone' boxes of any kind, of which on background] for one image."""
    r = evaluate([(gts, preds)], **kw)
    return [sum(r["n_gt"]), sum(r["n_hit"]),
            r["false_drone_on_drone"] + sum(r["false_drone_on_distractor"].values()) + r["false_drone_on_background"],
            r["false_drone_on_background"]]


def recall_precision(s) -> tuple:
    """From summed image_counts columns."""
    return s[1] / max(s[0], 1), s[1] / max(s[1] + s[2], 1)


def bootstrap_ci(keys, rows, stat, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple:
    """Cluster bootstrap: resample whole groups (videos), not frames. Frames of one video are strongly
    correlated, so a frame-level interval would be far too narrow (15 section 5: ~57 validation videos).

    keys: group of each row (e.g. video_of(path)); rows: per-image counts (n_images x m); stat: summed row -> float.
    For a paired comparison of two models, put both models' counts side by side in each row and let stat
    return the difference. Returns (point estimate, lower, upper) of the (1 - alpha) percentile interval."""
    rows = np.asarray(rows, float)
    groups = sorted(set(keys))
    index = {g: i for i, g in enumerate(groups)}
    per_group = np.zeros((len(groups), rows.shape[1]))
    np.add.at(per_group, [index[k] for k in keys], rows)
    idx = np.random.default_rng(seed).integers(0, len(groups), (n_boot, len(groups)))
    vals = [stat(s) for s in per_group[idx].sum(axis=1)]
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return stat(per_group.sum(axis=0)), float(lo), float(hi)


def format_report(result: dict, names=None) -> str:
    names = names or {}
    fa_w = result.get("false_drone_background_by_width") or [0] * len(result["bins"])
    lines = ["width px     drones   found   recall   bg false drones"]
    for (lo, hi), n, h, r, f in zip(result["bins"], result["n_gt"], result["n_hit"], result["recall"], fa_w):
        rng = f"{lo:g}-{hi:g}" if hi != float("inf") else f"{lo:g}+"
        lines.append(f"{rng:10} {n:8d} {h:7d}   " + (f"{r:.2f}" if r is not None else "  - ") + f"   {f:15d}")
    fd = ", ".join(f"{names.get(c, c)}: {n}" for c, n in sorted(result["false_drone_on_distractor"].items())) or "none"
    lines.append(f"Extra 'drone' boxes on real drones (duplicates / misplaced): {result.get('false_drone_on_drone', 0)}")
    ao = ", ".join(f"{names.get(c, c)}: {n}" for c, n in sorted(result.get("missed_drone_boxed_as_other", {}).items())) or "none"
    lines.append(f"Missed drones that were boxed as another class: {ao}")
    lines.append(f"False 'drone' boxes on distractors: {fd}")
    lines.append(f"False 'drone' boxes on background: {result['false_drone_on_background']} "
                 f"over {result['n_images']} images")
    return "\n".join(lines)


def split_pairs(split_dir):
    """(image_path, label_path) pairs of an Ultralytics split folder: <split>/images/*, <split>/labels/*.txt."""
    split_dir = Path(split_dir)
    for img in sorted((split_dir / "images").iterdir()):
        yield img, split_dir / "labels" / (img.stem + ".txt")


def run_split(detector, split_dir, conf: float = 0.25, tile: int = None) -> list:
    """(gts, preds) per image of a split, ready for evaluate().

    tile: for images larger than the detector's input (e.g. 1920x1080 Anti-UAV frames), detect on
    overlapping tile-sized crops and merge. Passing a large image straight to the detector would letterbox
    it DOWN to the input size and shrink every target, which is exactly what this project must not do.
    """
    import cv2
    from cuas.detector import detect_in_crops
    from cuas.tiling import grid_tiles
    out = []
    for img_path, lbl_path in split_pairs(split_dir):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gts = read_yolo_labels(lbl_path.read_text() if lbl_path.exists() else "", w, h)
        if tile and (w > tile or h > tile):
            preds = detect_in_crops(img, grid_tiles(w, h, tile), detector, tile)
        else:
            preds = detector([img])[0]
        out.append((gts, [p for p in preds if p[5] >= conf]))
    return out


def main():
    ap = argparse.ArgumentParser(description="Size-binned drone recall and distractor false alarms.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", required=True, help="Ultralytics split folder with images/ and labels/")
    ap.add_argument("--conf", type=float, default=0.25, help="operating confidence threshold")
    ap.add_argument("--target", type=float, default=0.8, help="recall that defines w_min")
    ap.add_argument("--json", help="also write the result here")
    ap.add_argument("--tile", type=int, help="tile images larger than this (e.g. 640 for Full HD frames)")
    args = ap.parse_args()

    from cuas.detector import UltralyticsDetector
    det = UltralyticsDetector(args.weights, conf=min(args.conf, 0.05))
    result = evaluate(run_split(det, args.split, args.conf, args.tile))
    result["w_min_px"] = w_min(result, args.target)
    print(format_report(result, det.names))
    print(f"w_min at recall >= {args.target}: {result['w_min_px']} px")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
