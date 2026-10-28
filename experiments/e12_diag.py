"""E12 diagnostics (log: 16 #34): (1) same-threshold Halmstad comparison E2 vs E12; (2) contact sheet of E12 background false alarms
on Anti-UAV test native frames, so we can see what they are.

    C:\\cuas_venv\\Scripts\\python experiments/e12_diag.py      (contact sheets -> C:/cuas_data/runs/e12_diag/)
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.detector import UltralyticsDetector, detect_in_crops  # noqa: E402
from cuas.eval import evaluate, iou, read_yolo_labels, split_pairs  # noqa: E402
from cuas.tiling import grid_tiles  # noqa: E402

D = Path("C:/cuas_data")
OUT = D / "runs" / "e12_diag"
OUT.mkdir(parents=True, exist_ok=True)
HALM = [D / "halmstad_yolo" / "val"] + [D / "halmstad_zoom" / f"k{k}" for k in (2, 3, 4, 6)]


def infer(det, split):
    rows = []
    for ip, lp in split_pairs(split):
        img = cv2.imread(str(ip))
        h, w = img.shape[:2]
        gts = read_yolo_labels(lp.read_text() if lp.exists() else "", w, h)
        preds = detect_in_crops(img, grid_tiles(w, h, 640), det, 640) if (w > 640 or h > 640) else det([img])[0]
        rows.append((ip, gts, preds))
    return rows


def summary(rows, conf):
    r = evaluate([(g, [p for p in ps if p[5] >= conf]) for _, g, ps in rows])
    n, h = sum(r["n_gt"]), sum(r["n_hit"])
    return f"recall {h / n:.3f}  on-drone {r['false_drone_on_drone']:4d}  background {r['false_drone_on_background']:4d}  " \
           f"distractor {sum(r['false_drone_on_distractor'].values()):3d}  as-other {sum(r['missed_drone_boxed_as_other'].values())}"


def background_fas(rows, conf):
    fas = []
    for ip, gts, ps in rows:
        for p in ps:
            if p[4] == 0 and p[5] >= conf and not any(iou(p, g) > 0 for g in gts):
                fas.append((ip, p))
    return fas


def sheet(fas, path, n=48, ctx=96):
    random.seed(0)
    pick = random.sample(fas, min(n, len(fas)))
    cells = []
    for ip, p in pick:
        img = cv2.imread(str(ip))
        H, W = img.shape[:2]
        cx, cy, s = int(p[0] + p[2] / 2), int(p[1] + p[3] / 2), max(ctx, int(max(p[2], p[3]) * 2))
        x0, y0 = max(0, cx - s // 2), max(0, cy - s // 2)
        crop = img[y0:min(H, y0 + s), x0:min(W, x0 + s)].copy()
        cv2.rectangle(crop, (int(p[0] - x0), int(p[1] - y0)), (int(p[0] + p[2] - x0), int(p[1] + p[3] - y0)), (0, 0, 255), 1)
        crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, f"{p[5]:.2f} w{int(p[2])}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(crop, ip.stem[:22], (3, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        cells.append(crop)
    while len(cells) % 8:
        cells.append(np.zeros((160, 160, 3), np.uint8))
    grid = np.vstack([np.hstack(cells[i:i + 8]) for i in range(0, len(cells), 8)])
    cv2.imwrite(str(path), grid)


e12 = UltralyticsDetector(str(D / "runs/E12_multidata_yolo26s/weights/best.pt"), conf=0.05, device=0)
e2 = UltralyticsDetector(str(D / "runs/E2_zoomtrain_yolo26s/weights/best.pt"), conf=0.05, device=0)

print("Halmstad val pooled, same thresholds:")
h12 = [x for s in HALM for x in infer(e12, s)]
h2 = [x for s in HALM for x in infer(e2, s)]
for c in (0.1, 0.25, 0.4):
    print(f"  conf {c:.2f}  E2 : {summary(h2, c)}")
    print(f"  conf {c:.2f}  E12: {summary(h12, c)}")
nat12 = [x for x in h12 if "halmstad_yolo" in str(x[0])]
nat2 = [x for x in h2 if "halmstad_yolo" in str(x[0])]
print("Halmstad native only @0.4: E2", summary(nat2, 0.4), "| E12", summary(nat12, 0.4))

t12 = infer(e12, D / "antiuav_test")
fas = background_fas(t12, 0.4)
imgs = {ip for ip, _ in fas}
print(f"\nAnti-UAV test native @0.4: {len(fas)} background FAs on {len(imgs)} of {len(t12)} frames; "
      f"videos with FAs: {len({ip.stem.rsplit('_', 1)[0] for ip in imgs})}")
by_img = {}
for ip, _ in fas:
    by_img[ip.stem.rsplit('_', 1)[0]] = by_img.get(ip.stem.rsplit('_', 1)[0], 0) + 1
print("top videos by FA count:", sorted(by_img.items(), key=lambda kv: -kv[1])[:10])
print("FA score quantiles:", np.quantile([p[5] for _, p in fas], [0.1, 0.5, 0.9]).round(2).tolist())
print("FA width quantiles:", np.quantile([p[2] for _, p in fas], [0.1, 0.5, 0.9]).round(1).tolist())
print("FA y-centre / H quantiles:", np.quantile([(p[1] + p[3] / 2) / 1080 for _, p in fas], [0.1, 0.5, 0.9]).round(2).tolist())
sheet(fas, OUT / "e12_antiuav_test_bg_fas.png")
sheet([f for f in fas if f[1][5] >= 0.6], OUT / "e12_antiuav_test_bg_fas_high.png")
hfas = background_fas(nat12 + [x for x in h12 if "halmstad_zoom" in str(x[0])], 0.4)
sheet(hfas, OUT / "e12_halmstad_bg_fas.png", n=24)
print("sheets written")
