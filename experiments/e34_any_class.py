"""
E34: what does it cost to let a box of ANY class start a track?

E13 showed that unseen drones are found by the airplane and helicopter heads far more often than by the drone head
(19 section 3.7), and the pipeline already tracks boxes of every class. This counts both sides of that route with
the same detections: drones reached, and candidate boxes on nothing labelled (each one a track the track classifier
then has to reject).

    C:\\cuas_venv\\Scripts\\python experiments/e34_any_class.py E18_base_s1@last E22a_s1@last
        (CPU. Halmstad val and Anti-UAV test come from the cached native detections; DUT is run once per run on
         the CPU, ~15 min, and cached in <run>/dets_dut<sfx>.json)

Output: C:/cuas_data/runs/e34_any_class.txt/.json.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuas.detector import nms  # noqa: E402
from cuas.eval import iou, run_files  # noqa: E402
from e15_dut import SPLIT, Padded  # noqa: E402
from e33_center_match import cached  # noqa: E402

D = Path(os.environ.get("CUAS_DATA", "C:/cuas_data"))
CONFS = (0.25, 0.40, 0.55)


def count(images, conf, any_class, iou_thr=0.3):
    """(drones, drones reached, candidates on a labelled non-drone, candidates on nothing labelled, images)."""
    n_gt = n_hit = on_other = on_nothing = 0
    for gts, preds in images:
        cands = [p for p in preds if p[5] >= conf and (any_class or p[4] == 0)]
        if any_class:
            cands = nms(cands, class_aware=False)             # one object, one candidate (as the pipeline does)
        drones = [g for g in gts if g[4] == 0]
        n_gt += len(drones)
        n_hit += sum(any(iou(p, g) >= iou_thr for p in cands) for g in drones)
        for p in cands:
            if any(iou(p, g) > 0 for g in drones):
                continue
            if any(iou(p, g) >= iou_thr for g in gts if g[4] != 0):
                on_other += 1
            else:
                on_nothing += 1
    return n_gt, n_hit, on_other, on_nothing, len(images)


def main(runs):
    lines, out = ["E34: candidates from 'drone' boxes only against boxes of any class (class-agnostic NMS). Per set and "
                  "threshold: drones reached, candidates on a labelled bird/airplane/helicopter, candidates on nothing "
                  "labelled per 1000 images."], {}
    for run in runs:
        name, _, sfx = run_files(run)
        native = json.loads((D / "runs" / name / f"dets_native{sfx}.json").read_text())
        sets = {k: [(g, p) for g, p in native[k]["images"]] for k in ("halmstad_val", "antiuav_test")}
        sets["dut_test"] = cached(run, "dut", {"dut_test": SPLIT}, wrap=Padded)["dut_test"]
        lines.append(f"\n{run}\n{'set':14s} {'conf':>5s}   {'drone boxes: reached / on other / on nothing per 1000':>56s}   "
                     f"{'any class: reached / on other / on nothing per 1000':>54s}")
        for set_name, images in sets.items():
            for conf in CONFS:
                cells = []
                for any_class in (False, True):
                    n, h, o, b, m = count(images, conf, any_class)
                    cells.append((h / max(n, 1), o, 1000 * b / m))
                    out[f"{run} {set_name} {conf} {'any' if any_class else 'drone'}"] = {
                        "drones": n, "reached": h, "on_other": o, "on_nothing": b, "images": m}
                lines.append(f"{set_name:14s} {conf:5.2f}   " + "   ".join(
                    f"{f'{r:.3f} / {o:5d} / {b:7.1f}':>55s}" for r, o, b in cells))
                print(lines[-1], flush=True)
    (D / "runs" / "e34_any_class.txt").write_text("\n".join(lines))
    (D / "runs" / "e34_any_class.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:] or ["E18_base_s1@last"])
