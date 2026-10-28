"""
Build the E14 training set: E12's data plus hard negatives mined from Anti-UAV TRAIN frames (14 section 4, E14).

E12 saw only Anti-UAV tiles that contain a drone, so it learned that dataset's clutter (lamps, buildings, cranes,
foliage) as drone context (16 #34). Here E12 runs over the Anti-UAV train frames (Full HD, tiled at 640, never
downscaled). Every "drone" box that touches no labelled drone is a false alarm, and a 640-px tile centred on it
becomes a training tile carrying the true labels (usually none).

Safety rules, because Anti-UAV labels one drone per sequence and "no label" does not prove "no drone":
  * drop false alarms narrower than MIN_W px: a small blob on sky may be a real, unlabelled drone (16 #34);
  * drop false alarms on the burned-in bottom-left caption, a dataset artefact the overlay mask misses
    (16 #37): ~60% of E12's train false alarms. It is absent from Anti-UAV test and from a deployed camera, so
    spending negatives on it would teach nothing useful;
  * skip a tile that would cut a labelled drone (it would be half-labelled);
  * cap each sequence (one night video must not dominate) and the total at SHARE of the training images.
Anti-UAV TEST is never touched.

    C:\\cuas_venv\\Scripts\\python experiments/build_e14.py
Mining pool: the E12 train frames (frames 0, 40, 80, ...) plus frames 13, 33, 53, ... of the same train videos (every
20th from offset 13, one decode pass; built here from the zip and never used as positives). Detections are cached per folder in
C:/cuas_data/e14/mined_<folder>.json, so re-running with other caps skips inference.
"""
import json
import random
import shutil
import zipfile
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.datasets.antiuav import caption_template, convert as convert_antiuav, find_caption, inpaint_split  # noqa: E402
from cuas.eval import iou, split_pairs  # noqa: E402
from cuas.tiling import crops_around, grid_tiles, read_yolo_labels, yolo_lines  # noqa: E402

D = Path("C:/cuas_data")
E14 = D / "e14"
FRAMES = D / "antiuav_train_frames"
EXTRA = D / "antiuav_train_mine"   # frames 13, 33, 53, ...: between the E12 frames, 0.65 s / 1.65 s after them
MINE_DIRS = [FRAMES, EXTRA]
ZIP = (Path.home() / "Downloads/Anti-UAV-RGBT.zip")
WEIGHTS = D / "runs" / "E12_multidata_yolo26s" / "weights" / "best.pt"
MINE_CONF = 0.25     # below the 0.4 operating point, so borderline clutter is mined too
MIN_W = 28           # px; narrower false alarms may be unlabelled drones (24 let a 24-px blob through, 16 #38)
CAPTION_REF = "20190926_195921_1_2_00000.jpg"   # a train frame that shows the caption, for the template
PER_SEQ = 40         # tiles per sequence at most
SHARE = 0.15         # negatives as a share of the final training set
TILE = 640


def seq_of(stem):
    return stem.rsplit("_", 1)[0]


def build_mining_frames():
    if (EXTRA / "images").exists():
        return
    raw = D / "antiuav_raw"
    z = zipfile.ZipFile(ZIP)
    members = [n for n in z.namelist() if n.startswith("train/") and "/visible." in n]
    print(f"0. extracting {len(members)} train visible files for extra mining frames", flush=True)
    for n in members:
        z.extract(n, raw)
    mask = cv2.imread(str(D / "antiuav_overlay_mask.png"), cv2.IMREAD_GRAYSCALE)
    convert_antiuav(raw, "train", EXTRA, every=20, absent_every=0, offset=13)
    print("   overlay inpainted", inpaint_split(EXTRA, mask), flush=True)
    shutil.rmtree(raw)
    print("   deleted extracted train video")


def mine(frames):
    cache = E14 / f"mined_{frames.name}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from cuas.detector import UltralyticsDetector, detect_in_crops
    det = UltralyticsDetector(str(WEIGHTS), conf=MINE_CONF, device=0)
    rows = []
    pairs = list(split_pairs(frames))
    for n, (ip, lp) in enumerate(pairs):
        img = cv2.imread(str(ip))
        h, w = img.shape[:2]
        gts = read_yolo_labels(lp.read_text() if lp.exists() else "", w, h)
        preds = detect_in_crops(img, grid_tiles(w, h, TILE), det, TILE)
        fas = [p[:4] + (p[5],) for p in preds
               if p[4] == 0 and p[5] >= MINE_CONF and not any(iou(p, g) > 0 for g in gts)]
        rows.append({"frame": str(ip), "fas": [[round(float(v), 2) for v in f] for f in fas]})
        if n % 500 == 0:
            print(f"   {frames.name}: mined {n}/{len(pairs)} frames", flush=True)
    E14.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    return rows


def cuts_a_drone(gts, ox, oy):
    """True if a labelled box is partly (not wholly) inside the tile."""
    for x, y, w, h, _ in gts:
        ix = max(0.0, min(x + w, ox + TILE) - max(x, ox))
        iy = max(0.0, min(y + h, oy + TILE) - max(y, oy))
        if 0 < ix * iy < 0.999 * w * h:
            return True
    return False


def sheet(items, path, n=64):
    """items: (frame_name, box). 8-column grid of 160-px cells, box in red, score and width on top."""
    random.seed(0)
    cells = []
    for name, p in random.sample(items, min(n, len(items))):
        img = cv2.imread(name)
        H, W = img.shape[:2]
        cx, cy, s = int(p[0] + p[2] / 2), int(p[1] + p[3] / 2), max(96, int(max(p[2], p[3]) * 2))
        x0, y0 = max(0, cx - s // 2), max(0, cy - s // 2)
        crop = img[y0:min(H, y0 + s), x0:min(W, x0 + s)].copy()
        cv2.rectangle(crop, (int(p[0] - x0), int(p[1] - y0)), (int(p[0] + p[2] - x0), int(p[1] + p[3] - y0)), (0, 0, 255), 1)
        crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, f"{p[4]:.2f} w{int(p[2])}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(crop, Path(name).stem[:22], (3, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        cells.append(crop)
    while len(cells) % 8:
        cells.append(np.zeros((160, 160, 3), np.uint8))
    cv2.imwrite(str(path), np.vstack([np.hstack(cells[i:i + 8]) for i in range(0, len(cells), 8)]))


def main():
    build_mining_frames()
    print("1. mining E12 false alarms on Anti-UAV train frames (conf >=", MINE_CONF, ")")
    rows = [r for d in MINE_DIRS for r in mine(d)]
    all_fas = [(r["frame"], f) for r in rows for f in r["fas"]]
    tpl = caption_template(cv2.imread(str(FRAMES / "images" / CAPTION_REF), cv2.IMREAD_GRAYSCALE))
    captions = {n: find_caption(cv2.imread(n, cv2.IMREAD_GRAYSCALE), tpl)
                for n in {n for n, _ in all_fas}}
    on_caption = [(n, f) for n, f in all_fas if captions[n] and iou(f, captions[n]) > 0]
    rest = [(n, f) for n, f in all_fas if not (captions[n] and iou(f, captions[n]) > 0)]
    kept = [(n, f) for n, f in rest if f[2] >= MIN_W]
    small = [(n, f) for n, f in rest if f[2] < MIN_W]
    print(f"   {len(all_fas)} false alarms on {sum(bool(r['fas']) for r in rows)} of {len(rows)} frames; "
          f"{len(on_caption)} on the burned-in caption dropped; "
          f"{len(small)} narrower than {MIN_W} px dropped; {len(kept)} kept; "
          f"{sum(f[4] >= 0.4 for _, f in kept)} of the kept score >= 0.4")
    if kept:
        print("   kept widths p10/50/90:", np.quantile([f[2] for _, f in kept], [0.1, 0.5, 0.9]).round(1).tolist(),
              " scores p10/50/90:", np.quantile([f[4] for _, f in kept], [0.1, 0.5, 0.9]).round(2).tolist())

    cfg = yaml.safe_load((D / "e12" / "data_e12.yaml").read_text())
    n_train = sum(len(list(Path(p).iterdir())) for p in cfg["train"])
    cap = int(round(SHARE / (1 - SHARE) * n_train))

    by_frame = {}
    for n, f in kept:
        by_frame.setdefault(n, []).append(f)
    cands, n_cut = [], 0
    for name, fas in by_frame.items():
        img_h, img_w = cv2.imread(name).shape[:2]
        lbl = Path(name).parent.parent / "labels" / (Path(name).stem + ".txt")
        gts = read_yolo_labels(lbl.read_text() if lbl.exists() else "", img_w, img_h)
        for ox, oy in crops_around(sorted(fas, key=lambda f: -f[4]), img_w, img_h, TILE):
            if cuts_a_drone(gts, ox, oy):
                n_cut += 1
                continue
            inside = [g for g in gts if ox <= g[0] and g[0] + g[2] <= ox + TILE and oy <= g[1] and g[1] + g[3] <= oy + TILE]
            cands.append({"frame": name, "ox": ox, "oy": oy, "gts": inside,
                          "score": max(f[4] for f in fas if ox <= f[0] + f[2] / 2 < ox + TILE and oy <= f[1] + f[3] / 2 < oy + TILE)})
    rng = random.Random(0)
    by_seq = {}
    for c in cands:
        by_seq.setdefault(seq_of(Path(c["frame"]).stem), []).append(c)
    chosen = []
    for s in sorted(by_seq):
        chosen += rng.sample(by_seq[s], min(PER_SEQ, len(by_seq[s])))
    if len(chosen) > cap:
        chosen = rng.sample(chosen, cap)
    print(f"2. {len(cands)} candidate tiles ({n_cut} skipped: would cut a labelled drone) from {len(by_seq)} sequences; "
          f"per-sequence cap {PER_SEQ} and total cap {cap} ({SHARE:.0%} of {n_train} + negatives) -> {len(chosen)} tiles; "
          f"{sum(bool(c['gts']) for c in chosen)} of them also contain the labelled drone")

    out = E14 / "antiuav_hardneg_tiles"
    for sub in ("images", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for old in (out / sub).iterdir():
            old.unlink()
    for c in chosen:
        img = cv2.imread(c["frame"])
        crop = img[c["oy"]:c["oy"] + TILE, c["ox"]:c["ox"] + TILE]
        stem = f"{Path(c['frame']).stem}_neg_{c['ox']}_{c['oy']}"
        cv2.imwrite(str(out / "images" / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        boxes = [(x - c["ox"], y - c["oy"], w, h, cls) for x, y, w, h, cls in c["gts"]]
        (out / "labels" / f"{stem}.txt").write_text(yolo_lines(boxes, TILE, TILE))
    (E14 / "chosen.json").write_text(json.dumps(chosen))
    print(f"3. wrote {len(chosen)} tiles to {out}")

    sheet(kept, E14 / "sheet_kept_random.png")
    sheet([k for k in kept if k[1][4] >= 0.5], E14 / "sheet_kept_high.png")
    sheet(small, E14 / "sheet_dropped_small.png", n=48)
    sheet(on_caption, E14 / "sheet_dropped_caption.png", n=32)
    print("4. contact sheets in", E14)

    cfg["train"] = cfg["train"] + [str(out / "images").replace("\\", "/")]
    (E14 / "data_e14.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("5. wrote", E14 / "data_e14.yaml", f"(train {n_train} + {len(chosen)} images)")


if __name__ == "__main__":
    main()
