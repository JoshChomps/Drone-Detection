"""
E36: where are the false drones? A map of background false alarms by size, background and score, for the plain,
E22a and E22c recipes (three seeds each), on the Anti-UAV test (trained-on dataset, unseen videos) and DUT (never
trained on).

Question (user, 2026-10-05: "we need to cut the false drones"): E22c raises false drones at its own threshold
(16 #72-#73). Are they on plain sky, which the deployed camera sees (13 section 2.4), or on buildings, trees and
ground, which its sky mask removes? Small or large? Low-scoring (a threshold problem) or confident (a model
problem)? The answer decides the lever: threshold and tracking, a sky mask, or training data.

A background false alarm is a 'drone' box that touches no labelled drone and overlaps no labelled bird, airplane or
helicopter (IoU < 0.3), as in cuas.eval.evaluate. Background = E15's ring texture around the box (sky < 5.0).
Two operating points per run: the shared threshold 0.40, and the run's own threshold where background false
alarms are 30 per 1000 images (the R@30 point of E35), so the composition is compared at an equal count.
A contact sheet of the most confident false alarms of each recipe (seed 1) is written to look at.

    python experiments/e36_false_alarm_map.py        (CPU, ~5 min; needs <run>/dets_native_last.json and dets_dut_last.json)
Output: C:/cuas_data/runs/e36_false_alarm_map.txt / .json, e36_sheet_<set>.png
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuas.eval import iou, read_yolo_labels, split_pairs  # noqa: E402
from e15_dut import SKY_T, SPLIT as DUT_SPLIT, ring_texture  # noqa: E402

D = Path(os.environ.get("CUAS_DATA", "C:/cuas_data"))
RECIPES = {"plain": "E18_base", "E22a": "E22a", "E22c": "E22c"}
SEEDS = (1, 2, 0)
WIDTHS = (0, 8, 16, 32, 64, float("inf"))
BANDS = (0.10, 0.25, 0.40, 0.55, 0.70, 1.01)
GRID = [round(0.05 + 0.01 * k, 2) for k in range(91)]


def load_sets(run):
    nat = json.loads((D / "runs" / run / "dets_native_last.json").read_text())["antiuav_test"]
    dut = json.loads((D / "runs" / run / "dets_dut_last.json").read_text())["dut_test"]
    full = {Path(ip).name: str(ip) for ip, _ in split_pairs(D / "antiuav_test")}     # the cache keeps file names only
    return {"antiuav_test": list(zip([full[Path(n).name] for n in nat["paths"]], nat["images"])),
            "dut_test": list(zip([str(p) for p, _ in split_pairs(DUT_SPLIT)], dut))}


def background_fas(gts, preds):
    """'drone' boxes on nothing labelled (cuas.eval.evaluate's 'background' kind)."""
    drones, others = [g for g in gts if g[4] == 0], [g for g in gts if g[4] != 0]
    return [p for p in preds if p[4] == 0 and not any(iou(p, g) > 0 for g in drones)
            and not any(iou(p, g) >= 0.3 for g in others)]


def threshold_at(fas_scores, n_images, budget=30.0):
    """Lowest grid threshold with at most `budget` background false alarms per 1000 images."""
    s = np.sort(np.asarray(fas_scores))
    for t in GRID:
        if 1000 * (len(s) - np.searchsorted(s, t)) / n_images <= budget:
            return t
    return GRID[-1]


def check_order(set_name, items):
    """The cached ground truth must be the labels of the listed images (DUT relies on split order)."""
    label_of = {str(Path(ip)): lp for ip, lp in split_pairs(DUT_SPLIT if set_name == "dut_test" else D / "antiuav_test")}
    bad = 0
    for path, (gts, _) in items[:: max(1, len(items) // 200)]:
        img = cv2.imread(str(path))
        lp = label_of[str(Path(path))]
        labels = read_yolo_labels(lp.read_text() if lp.exists() else "", img.shape[1], img.shape[0])
        bad += len(labels) != len(gts) or any(abs(a[0] - b[0]) > 1 for a, b in zip(labels, gts))
    if bad:
        raise SystemExit(f"{set_name}: cached detections do not line up with the image list ({bad} mismatches)")


def main():
    runs = {f"{r}_s{s}": (name, s) for name, r in RECIPES.items() for s in SEEDS}
    data = {run: load_sets(f"{run}") for run in runs}
    first = next(iter(data.values()))
    for set_name, items in first.items():
        check_order(set_name, items)
    # Every background false alarm at score >= 0.10, per run and set; textures computed once per image
    fas = {run: {k: [] for k in first} for run in runs}         # (image index, box, score)
    for run, sets in data.items():
        for k, items in sets.items():
            for i, (_, (gts, preds)) in enumerate(items):
                fas[run][k] += [(i, p) for p in background_fas(gts, [p for p in preds if p[5] >= 0.10])]
    texture = {}
    for k, items in first.items():
        need = {}
        for run in runs:
            for i, p in fas[run][k]:
                need.setdefault(i, []).append(p)
        for i, boxes in need.items():
            g = cv2.cvtColor(cv2.imread(str(items[i][0])), cv2.COLOR_BGR2GRAY).astype(np.float32)
            for p in boxes:
                texture[(k, i, tuple(p[:4]))] = ring_texture(g, p)
        print(f"{k}: textures for {len(need)} images", flush=True)

    def table(run, k, t):
        sel = [(i, p) for i, p in fas[run][k] if p[5] >= t]
        n_img = len(first[k])
        sky = [texture[(k, i, tuple(p[:4]))] < SKY_T for i, p in sel]
        by_w = [sum(lo <= p[2] < hi for _, p in sel) for lo, hi in zip(WIDTHS, WIDTHS[1:])]
        by_s = [sum(lo <= p[5] < hi for _, p in sel) for lo, hi in zip(BANDS, BANDS[1:])]
        return {"t": t, "n": len(sel), "per_1000": 1000 * len(sel) / n_img, "sky": int(sum(sky)),
                "widths": by_w, "scores": by_s}

    out, lines = {}, ["E36: background false 'drone' boxes by size, background and score. Sky = E15 ring texture < "
                      f"{SKY_T}. Counts per run; seeds 1 / 2 / 0 summed per recipe in the tables."]
    for k in first:
        n_img = len(first[k])
        lines += ["", f"=== {k} ({n_img} images) ==="]
        for point in ("0.40", "30 per 1000"):
            lines += ["", f"At {'threshold 0.40' if point == '0.40' else 'each run’s own threshold for 30 false alarms per 1000 images'}:",
                      f"{'recipe':7s} {'thresholds':>17s} {'FAs':>5s} {'/1000':>6s} {'on sky':>7s}   "
                      + "".join(f"{f'{lo:g}-{hi:g}px':>9s}" for lo, hi in zip(WIDTHS, WIDTHS[1:]))
                      + "   " + "".join(f"{f'{lo:.2f}+':>7s}" for lo in BANDS[:-1])]
            for name in RECIPES:
                rows = []
                for run, (rname, s) in runs.items():
                    if rname != name:
                        continue
                    t = 0.40 if point == "0.40" else threshold_at([p[5] for _, p in fas[run][k]], n_img)
                    rows.append(table(run, k, t))
                    out.setdefault(k, {}).setdefault(point, {})[run] = rows[-1]
                n = sum(r["n"] for r in rows)
                ts = "/".join(f"{r['t']:.2f}" for r in rows)
                w = [sum(r["widths"][j] for r in rows) for j in range(len(WIDTHS) - 1)]
                sc = [sum(r["scores"][j] for r in rows) for j in range(len(BANDS) - 1)]
                sky = sum(r["sky"] for r in rows)
                lines.append(f"{name:7s} {ts:>17s} {n:5d} {1000 * n / n_img / len(rows):6.1f} "
                             f"{f'{100 * sky / max(n, 1):.0f}%':>7s}   " + "".join(f"{x:9d}" for x in w)
                             + "   " + "".join(f"{x:7d}" for x in sc))
        # Contact sheet: the 24 most confident false alarms of each recipe, seed 1, with context
        tiles = []
        for name, r in RECIPES.items():
            top = sorted(fas[f"{r}_s1"][k], key=lambda ip: -ip[1][5])[:24]
            for i, p in top:
                img = cv2.imread(str(first[k][i][0]))
                x, y, w, h = p[:4]
                m = max(24, 2 * max(w, h))
                cx, cy = x + w / 2, y + h / 2
                x0, y0 = int(max(0, cx - m)), int(max(0, cy - m))
                crop = img[y0:int(cy + m), x0:int(cx + m)].copy()
                cv2.rectangle(crop, (int(x - x0), int(y - y0)), (int(x + w - x0), int(y + h - y0)), (0, 0, 255), 1)
                crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_NEAREST if crop.shape[0] < 128 else cv2.INTER_AREA)
                sky = texture[(k, i, tuple(p[:4]))] < SKY_T
                cv2.putText(crop, f"{name} {p[5]:.2f} {w:.0f}px {'sky' if sky else 'clut'}", (2, 124),
                            cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 255, 255), 1)
                tiles.append(crop)
        rows_img = [np.hstack(tiles[j:j + 12]) for j in range(0, len(tiles), 12) if len(tiles[j:j + 12]) == 12]
        if rows_img:
            cv2.imwrite(str(D / "runs" / f"e36_sheet_{k}.png"), np.vstack(rows_img))
    text = "\n".join(lines)
    (D / "runs" / "e36_false_alarm_map.txt").write_text(text, encoding="utf-8")
    (D / "runs" / "e36_false_alarm_map.json").write_text(json.dumps(out, indent=1))
    print(text)


if __name__ == "__main__":
    main()
