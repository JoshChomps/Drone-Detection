"""
E37: the threshold-free DUT scores (E35) on the part of DUT this project's camera sees: images whose every labelled
drone is against plain sky (E15's ring texture < 5.0), and the clutter remainder for contrast.

Question (2026-10-06, 16 #81): E24 (`--freeze 10`) raised DUT AP by ~0.12 over E22c in every seed, but at its own
threshold most of the gain was on drones in front of buildings and trees, which the deployed camera masks out
(13 section 2.4). How much of the gain is on sky?

Images without drones are kept in both subsets (they only add false alarms), so R@10 / R@30 keep the meaning of E35.

    python experiments/e37_dut_sky.py E22c_s1@last E24_s1@last ...     (CPU, cached dets_dut_last.json, ~5 min)
Output: C:/cuas_data/runs/e37_dut_sky.txt / .json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuas.eval import run_files  # noqa: E402
from e15_dut import backgrounds  # noqa: E402
from e35_metrics import D, average_precision, pairs, recall_at_budget  # noqa: E402

GROUPS = {"E18_base": "plain", "E22a": "E22a", "E22c": "E22c", "E24": "E24", "E24b": "E24b", "E24c": "E24c",
          "E24d": "E24d", "E27": "E27"}


def main(runs):
    bgs = backgrounds()                                   # per image, per drone: "sky" / "clutter", in split order
    sky = [all(b == "sky" for b in bg) for bg in bgs]     # drone-free images count in both subsets
    clutter = [not bg or any(b == "clutter" for b in bg) for bg in bgs]
    lines = [f"E37: DUT scores without a threshold by background. sky = images whose every drone is on sky "
             f"({sum(1 for b in bgs if b and all(x == 'sky' for x in b))} images with drones); clutter = the rest "
             f"({sum(1 for b in bgs if any(x == 'clutter' for x in b))}); {sum(1 for b in bgs if not b)} drone-free "
             f"images in both.", f"{'run':16s} {'subset':8s} {'AP':>6s} {'R@10':>6s} {'R@30':>6s}"]
    out = {}
    for run in runs:
        name, _, sfx = run_files(run)
        images = json.loads((D / "runs" / name / f"dets_dut{sfx}.json").read_text())["dut_test"]
        assert len(images) == len(bgs), (run, len(images), len(bgs))
        out[run] = {}
        for k, keep in (("sky", sky), ("clutter", clutter)):
            sub = [x for x, m in zip(images, keep) if m]
            rows, n_gt = pairs(sub)
            res = {"AP": average_precision(rows, n_gt), "R@10": recall_at_budget(sub, 10),
                   "R@30": recall_at_budget(sub, 30), "n_drones": n_gt}
            out[run][k] = res
            lines.append(f"{run:16s} {k:8s} {res['AP']:6.3f} {res['R@10']:6.3f} {res['R@30']:6.3f}")
    lines += ["", "Means over seeds:"]
    for prefix, label in GROUPS.items():
        rs = [r for r in out if r.startswith(prefix + "_s")]
        if rs:
            for k in ("sky", "clutter"):
                m = {q: np.mean([out[r][k][q] for r in rs]) for q in ("AP", "R@10", "R@30")}
                lines.append(f"{label:16s} {k:8s} {m['AP']:6.3f} {m['R@10']:6.3f} {m['R@30']:6.3f}   (n={len(rs)})")
    text = "\n".join(lines)
    (D / "runs" / "e37_dut_sky.txt").write_text(text)
    (D / "runs" / "e37_dut_sky.json").write_text(json.dumps(out, indent=1))
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
