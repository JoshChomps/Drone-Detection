"""Generates experiments/E1_baseline_detector.ipynb. Edit the cells here, then run:
    python experiments/make_e1_notebook.py experiments/E1_baseline_detector.ipynb
"""
import ast
import json
import sys

OUT = sys.argv[1]

CELLS = [
("md", """# E1: baseline detector on Halmstad (visible)
Trains YOLO26s on the Halmstad drone / bird / airplane / helicopter videos, then measures **drone recall by box
width** (-> `w_min`, doc 13) and **how often birds and aircraft get called "drone"**. Plan and results: doc 14.

Runtime -> Change runtime type -> **T4 GPU**. Run the cells in order. Cell 2 asks you to upload
`experiments/cuas_code.zip` from the project folder."""),

("code", """# CELL 1 - GPU check and installs (~1 min)
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install ultralytics mcos-decoder"""),

("code", """# CELL 2 - upload the project code: choose experiments/cuas_code.zip from the project folder
import sys, zipfile
from google.colab import files
up = files.upload()
zipfile.ZipFile(next(iter(up)), 'r').extractall('/content/code')
sys.path.insert(0, '/content/code')
import cuas.eval, cuas.tiling, cuas.detector
print('cuas code loaded OK')"""),

("code", """# CELL 3 - Google Drive for results (weights survive a disconnect) and paths
from google.colab import drive
drive.mount('/content/drive')
RUNS = '/content/drive/MyDrive/cuas_runs'
DATA = '/content/halmstad_yolo'
ZOOM = '/content/halmstad_zoom'
SRC = '/content/halmstad/Data/Video_V'"""),

("code", """# CELL 4 - download the dataset (357 MB, CC0). Expect 570 files (285 videos + 285 label files)
!git clone -q --depth 1 https://github.com/DroneDetectionThesis/Drone-detection-dataset /content/halmstad
!ls {SRC} | wc -l"""),

("code", """# CELL 5 - sanity check. PASTE THIS OUTPUT BACK. Look at the picture: the red box must surround the drone.
import glob, collections, cv2, numpy as np
import matplotlib.pyplot as plt
from mcos_decoder import load_groundtruth

problems, widths, frames_total, sizes = [], collections.defaultdict(list), 0, collections.Counter()
for vid in sorted(glob.glob(f'{SRC}/V_*.mp4')):
    labels = load_groundtruth(vid.replace('.mp4', '_LABELS.mat'))
    cap = cv2.VideoCapture(vid)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sizes[(int(cap.get(3)), int(cap.get(4)))] += 1
    cap.release()
    frames_total += n
    if abs(n - len(labels)) > 1:
        problems.append(f'{vid.split("/")[-1]}: {n} frames vs {len(labels)} labels')
    cls = vid.split('V_')[-1].split('_')[0]
    widths[cls] += [b[2] for b in labels if b is not None]

print('total frames', frames_total, '| frame sizes', dict(sizes))
for cls, ws in sorted(widths.items()):
    ws = np.array(ws)
    print(f'{cls:10s} {len(ws):6d} boxes | width px: median {np.median(ws):5.1f}, '
          f'10th pct {np.percentile(ws, 10):5.1f}, under 8 px {np.mean(ws < 8):.0%}')
if problems:
    print('PROBLEMS:')
    for p in problems[:10]:
        print('  ', p)
else:
    print('Label/frame counts: OK')

vid = f'{SRC}/V_DRONE_001.mp4'
labels = load_groundtruth(vid.replace('.mp4', '_LABELS.mat'))
labelled = [k for k, b in enumerate(labels) if b is not None]
i = labelled[len(labelled) // 2]
cap = cv2.VideoCapture(vid)
cap.set(cv2.CAP_PROP_POS_FRAMES, i)
ok, f = cap.read()
x, y, w, h = map(int, labels[i])
cv2.rectangle(f, (x - 6, y - 6), (x + w + 6, y + h + 6), (0, 0, 255), 2)
plt.figure(figsize=(9, 7)); plt.imshow(f[..., ::-1]); plt.axis('off')
plt.title(f'V_DRONE_001 frame {i}: the red box should surround the drone'); plt.show()"""),

("code", """# CELL 6 - build the training set (split by VIDEO, every 10th labelled frame) and the "simulated distance"
# validation copies: val frames shrunk 2x/3x/4x/6x into mosaics, because Halmstad has no drones under ~20 px (~3-5 min)
!cd /content/code && python -m cuas.datasets.halmstad --src {SRC} --out {DATA} --every 10
!cd /content/code && python -m cuas.zoomout --src {DATA}/val --dst {ZOOM} --k 2 3 4 6
!cat {DATA}/data.yaml"""),

("code", """# CELL 7 - evaluation helper (defines a function; runs nothing yet). It pools the normal val set with the
# shrunken copies, so the recall-by-width table reaches down to ~4 px.
import json
from cuas.detector import UltralyticsDetector
from cuas.eval import evaluate, run_split, w_min, format_report

def evaluate_run(name, confs=(0.1, 0.25, 0.4, 0.55)):
    det = UltralyticsDetector(f'{RUNS}/{name}/weights/best.pt', conf=0.05)
    splits = [f'{DATA}/val'] + [f'{ZOOM}/k{k}' for k in (2, 3, 4, 6)]
    per_image = [x for s in splits for x in run_split(det, s, conf=0.05)]
    for conf in confs:
        r = evaluate([(g, [p for p in ps if p[5] >= conf]) for g, ps in per_image])
        r['w_min_px'], r['conf'] = w_min(r), conf
        print(f'=== {name}, conf {conf} (native + simulated distance) ===')
        print(format_report(r, det.names))
        print(f"w_min: {r['w_min_px']} px")
        print()
        json.dump(r, open(f'{RUNS}/{name}/cuas_eval_conf{conf}.json', 'w'), indent=2)
print('evaluate_run ready')"""),

("code", """# CELL 8 - SMOKE TEST (~5-10 min): 2 epochs on 10% of the data. The numbers will be bad; this only proves
# the whole chain works before spending an hour. Note the time per epoch that Ultralytics prints.
from ultralytics import YOLO
_ = YOLO('yolo26s.pt').train(data=f'{DATA}/data.yaml', imgsz=640, epochs=2, batch=16, fraction=0.1, seed=0,
                             project=RUNS, name='smoke', exist_ok=True, plots=False)
evaluate_run('smoke')"""),

("code", """# CELL 9 - E1 FULL RUN (~1-2 h). If Colab disconnects, run this instead to resume:
#   YOLO(f'{RUNS}/E1_halmstad_yolo26s/weights/last.pt').train(resume=True)
_ = YOLO('yolo26s.pt').train(data=f'{DATA}/data.yaml', imgsz=640, epochs=40, batch=16, seed=0, patience=15,
                             project=RUNS, name='E1_halmstad_yolo26s', exist_ok=True)   # _ = hides a huge metrics printout"""),

("code", """# CELL 10 - E1 results. PASTE THIS OUTPUT BACK.
evaluate_run('E1_halmstad_yolo26s')"""),
("md", """# E2: train on simulated distance
Same as E1 except the zoom-out mosaics of the **train** split (k = 2, 3, 4, 6) are added to training (~+45% images).
Same schedule as E1 (40 epochs, patience 15) so the only change is the data. Compare its cell 13 tables with E1's cell 10."""),

("code", """# CELL 11 - E2 data: zoom-out mosaics of the TRAIN split, added to training via a second data yaml (~2 min)
ZTRAIN = '/content/halmstad_zoom_train'
!cd /content/code && python -m cuas.zoomout --src {DATA}/train --dst {ZTRAIN} --k 2 3 4 6
import yaml
cfg = yaml.safe_load(open(f'{DATA}/data.yaml'))
cfg['train'] = ['train/images'] + [f'{ZTRAIN}/k{k}/images' for k in (2, 3, 4, 6)]
yaml.safe_dump(cfg, open(f'{DATA}/data_e2.yaml', 'w'), sort_keys=False)
print(open(f'{DATA}/data_e2.yaml').read())"""),

("code", """# CELL 12 - E2 training (~1.5-2 h). Resume after a disconnect with:
#   YOLO(f'{RUNS}/E2_zoomtrain_yolo26s/weights/last.pt').train(resume=True)
_ = YOLO('yolo26s.pt').train(data=f'{DATA}/data_e2.yaml', imgsz=640, epochs=40, batch=16, seed=0, patience=15,
                             project=RUNS, name='E2_zoomtrain_yolo26s', exist_ok=True)"""),

("code", """# CELL 13 - E2 results. PASTE THIS OUTPUT BACK (same tables as E1's cell 10).
evaluate_run('E2_zoomtrain_yolo26s')"""),

("md", """# Diagnostics: where do false alarms come from?
Run after E1 (and again after E2). The pooled tables mix normal-size and shrunken images, so these cells split them apart
and show what the false 'drone' boxes actually look like."""),

("code", """# CELL 14 - per-set breakdown at one confidence. PASTE THIS OUTPUT BACK.
from cuas.eval import iou, split_pairs

def split_breakdown(name, conf=0.25):
    det = UltralyticsDetector(f'{RUNS}/{name}/weights/best.pt', conf=0.05)
    print(f'{name} at conf {conf}')
    print('set      images drones recall  on-drone  bg-FA/img  bird->drone  plane->drone  heli->drone')
    for s in ['native', 'k2', 'k3', 'k4', 'k6']:
        path = f'{DATA}/val' if s == 'native' else f'{ZOOM}/{s}'
        r = evaluate([(g, [p for p in ps if p[5] >= conf]) for g, ps in run_split(det, path, conf=0.05)])
        n, hit, fd = sum(r['n_gt']), sum(r['n_hit']), r['false_drone_on_distractor']
        print(f"{s:8s} {r['n_images']:6d} {n:6d} {hit / max(n, 1):6.2f} {r['false_drone_on_drone']:9d} {r['false_drone_on_background'] / r['n_images']:10.3f}"
              f" {fd.get(1, 0):12d} {fd.get(2, 0):13d} {fd.get(3, 0):12d}")

split_breakdown('E1_halmstad_yolo26s')
print()
split_breakdown('E2_zoomtrain_yolo26s')"""),

("code", """# CELL 15 - LOOK at background false alarms (red box = false 'drone', green = labelled objects).
# Tell me what most of them are: mosaic seams? unlabelled birds/drones? clouds, trees, compression blocks?
import random
import cv2
import matplotlib.pyplot as plt
from cuas.detector import UltralyticsDetector
from cuas.eval import iou, split_pairs
from cuas.tiling import read_yolo_labels

def show_false_alarms(name, split='native', conf=0.25, n=12, seed=0):
    det = UltralyticsDetector(f'{RUNS}/{name}/weights/best.pt', conf=conf)
    path = f'{DATA}/val' if split == 'native' else f'{ZOOM}/{split}'
    pairs = list(split_pairs(path))
    random.Random(seed).shuffle(pairs)
    found = []
    for img_path, lbl_path in pairs:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gts = read_yolo_labels(lbl_path.read_text() if lbl_path.exists() else '', w, h)
        for p in det([img])[0]:
            if p[4] == 0 and p[5] >= conf and all(iou(p, g) < 0.3 for g in gts):
                found.append((img, p, gts))
        if len(found) >= n:
            break
    cols = 4
    rows = max(1, (len(found) + cols - 1) // cols)
    plt.figure(figsize=(16, 4 * rows))
    for i, (img, p, gts) in enumerate(found[:n]):
        x, y, bw, bh = map(int, p[:4])
        cx, cy, half = x + bw // 2, y + bh // 2, max(48, bw * 2)
        vis = img.copy()
        for g in gts:
            gx, gy, gw, gh = map(int, g[:4])
            cv2.rectangle(vis, (gx - 2, gy - 2), (gx + gw + 2, gy + gh + 2), (0, 255, 0), 1)
        cv2.rectangle(vis, (x - 2, y - 2), (x + bw + 2, y + bh + 2), (0, 0, 255), 1)
        crop = vis[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(crop[..., ::-1])
        plt.title(f'score {p[5]:.2f}, {bw}x{bh} px')
        plt.axis('off')
    plt.suptitle(f'{name}: background false alarms on {split}')
    plt.show()

show_false_alarms('E2_zoomtrain_yolo26s', 'native', conf=0.4)
show_false_alarms('E2_zoomtrain_yolo26s', 'k3', conf=0.4)"""),

("code", """# CELL 16 - are the 'background' false alarms really unlabelled drones? PASTE THIS OUTPUT BACK.
# Halmstad frames with no label were used as 'empty' examples (23 in val, 135 in train). Count how many of the
# false alarms land on those images, and how many 'empty' TRAINING images the model finds a drone in.
import cv2
from cuas.detector import UltralyticsDetector
from cuas.eval import iou, split_pairs
from cuas.tiling import read_yolo_labels

det = UltralyticsDetector(f'{RUNS}/E2_zoomtrain_yolo26s/weights/best.pt', conf=0.4)

def scan(split):
    on_empty, on_labelled, n_empty, empty_with_drone = 0, 0, 0, []
    for img_path, lbl_path in split_pairs(f'{DATA}/{split}'):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gts = read_yolo_labels(lbl_path.read_text() if lbl_path.exists() else '', w, h)
        fas = [p for p in det([img])[0] if p[4] == 0 and all(iou(p, g) == 0 for g in gts)]
        if not gts:
            n_empty += 1
            on_empty += len(fas)
            if fas:
                empty_with_drone.append(img_path.name)
        else:
            on_labelled += len(fas)
    print(f'{split}: {n_empty} "empty" images, {len(empty_with_drone)} of them contain a detected drone')
    print(f'   false drones on "empty" images: {on_empty}, on labelled images: {on_labelled}')
    print('   examples:', ', '.join(empty_with_drone[:8]))

scan('val')
scan('train')"""),
]

cells = []
for kind, src in CELLS:
    lines = src.splitlines(True)
    if kind == "md":
        cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
    else:
        ast.parse("".join(l for l in lines if not l.lstrip().startswith("!")))   # syntax check
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines})

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", OUT, len(cells), "cells; all code cells parse")
