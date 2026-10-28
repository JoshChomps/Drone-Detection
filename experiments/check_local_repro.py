"""
Does the local laptop pipeline reproduce the Colab numbers? Run this after any environment change.

Re-evaluates E1 (conf 0.25) and E2 (conf 0.4) on the rebuilt Halmstad val (native + zoom-out k2/k3/k4/k6),
exactly as the Colab "clean re-evaluation" did (14 section 5, 2026-09-24), and compares recall per width bin
with the Colab reference. Differences come only from library versions and GPU non-determinism, so every bin
should agree within TOL.

    C:\\cuas_venv\\Scripts\\python -m cuas.datasets.halmstad --src C:/cuas_data/halmstad/Data/Video_V --out C:/cuas_data/halmstad_yolo --every 10
    C:\\cuas_venv\\Scripts\\python -m cuas.zoomout --src C:/cuas_data/halmstad_yolo/val --dst C:/cuas_data/halmstad_zoom --k 2 3 4 6
    C:\\cuas_venv\\Scripts\\python experiments/check_local_repro.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.detector import UltralyticsDetector  # noqa: E402
from cuas.eval import evaluate, format_report, run_split  # noqa: E402

DATA = Path("C:/cuas_data")
TOL = 0.03
# Colab T4 clean re-evaluation, 2026-09-24 (bins 4-8, 8-12, 12-16, 16-24, 24-32, 32-64), plus image count
COLAB = {
    "E1_halmstad_yolo26s": (0.25, [0.03, 0.38, 0.72, 0.91, 0.96, 0.99]),
    "E2_zoomtrain_yolo26s": (0.4, [0.79, 0.96, 0.92, 0.93, 0.96, 0.91]),
}
COLAB_IMAGES = 2550


def main():
    sets = [DATA / "halmstad_yolo" / "val"] + [DATA / "halmstad_zoom" / f"k{k}" for k in (2, 3, 4, 6)]
    ok = True
    for run, (conf, ref) in COLAB.items():
        det = UltralyticsDetector(str(DATA / "runs" / run / "weights" / "best.pt"), conf=0.05, device=0)
        per_image = [x for s in sets for x in run_split(det, s, conf=conf)]
        r = evaluate(per_image)
        print(f"=== {run} @ {conf}, local ===")
        print(format_report(r, det.names))
        local = [round(float(v), 2) for v in r["recall"][1:7]]
        diffs = [abs(a - b) for a, b in zip(local, ref)]
        good = max(diffs) <= TOL and r["n_images"] == COLAB_IMAGES
        ok &= good
        print(f"images {r['n_images']} (Colab {COLAB_IMAGES})")
        print(f"recall 4-64 px  local {local}\n                colab {ref}\n  max diff {max(diffs):.3f} -> "
              + ("MATCH" if good else "MISMATCH"))
        print()
    print("LOCAL PIPELINE REPRODUCES COLAB" if ok else "LOCAL PIPELINE DOES NOT MATCH COLAB: investigate before using it")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
