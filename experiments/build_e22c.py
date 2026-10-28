"""
Build the E22c training set: E14's data plus tiles with a REAL drone pasted in from the other dataset.

Question: is the drone class tied to the scenes it was filmed in? Every run so far finds 12-16% of the drones of a
dataset it never saw (DUT), whatever the architecture, class set or degradation (19 sections 3.6, 3.7; 15 section
6.14). In training, a Halmstad drone only ever appears in Halmstad sky and an Anti-UAV drone in an Anti-UAV scene.
Here each dataset's drones are cut out (cuas/paste.py) and pasted into the other's training images:

    Anti-UAV drones  ->  Halmstad frames (also the bird, airplane and helicopter videos: drone and bird together)
    Halmstad drones  ->  Anti-UAV tiles and the mined clutter tiles (drone over buildings, lights, cranes)

at a random place (not only sky) and a random size no larger than the original. The pasted tile keeps its own
labels and gains one drone box. Degradation is left to train_local.py --degrade, on the composed image.

What it can and cannot show: it varies the BACKGROUND behind known drones. It adds no new drone models, so if DUT
fails because its drones look different, this will not fix it; own target-free footage as background and the DUT
train split are the next steps either way (19 section 3.9). DUT test and Anti-UAV test are never touched.

    python experiments/build_e22c.py            (CPU, ~3 min)  ->  C:/cuas_data/e22c/paste_tiles, data_e22c.yaml, sheet.png
"""
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.eval import iou, split_pairs, video_of  # noqa: E402
from cuas.paste import cut, paste  # noqa: E402
from cuas.tiling import read_yolo_labels, yolo_lines  # noqa: E402

D = Path("C:/cuas_data")
OUT = D / "e22c"
SOURCES = {"antiuav": [D / "antiuav_train_frames"], "halmstad": [D / "halmstad_yolo" / "train"]}
TARGETS = {"antiuav": [D / "halmstad_yolo" / "train"],                      # Anti-UAV drones go INTO Halmstad frames
           "halmstad": [D / "e12" / "antiuav_train_tiles", D / "e14" / "antiuav_hardneg_tiles"]}
PER_DIRECTION = 1500      # 3,000 pasted tiles in all: about 16% of the training set
MIN_W, PER_VIDEO = 14, 12  # source drones at least this wide (a matte needs pixels); patches kept per source video
SCALE = (0.35, 1.0)       # never enlarged: an upscaled crop is a blurred crop
MIN_PASTED_W = 8
NIGHT = 60                # grey level of the old background below which a patch counts as filmed at night


def patches(folders, rng):
    """Matted drone patches, at most PER_VIDEO per source video so one long video cannot dominate."""
    by_video, tried = {}, 0
    for folder in folders:
        pairs = list(split_pairs(folder))
        rng.shuffle(pairs)
        for ip, lp in pairs:
            v = video_of(ip)
            if len(by_video.get(v, [])) >= PER_VIDEO or not lp.exists():
                continue
            img = cv2.imread(str(ip))
            h, w = img.shape[:2]
            for box in read_yolo_labels(lp.read_text(), w, h):
                if box[4] == 0 and box[2] >= MIN_W:
                    tried += 1
                    p = cut(img, box)
                    if p is not None:
                        by_video.setdefault(v, []).append(p)
    out = [p for ps in by_video.values() for p in ps]
    return out, tried, len(by_video)


def compose(img, labels, patch, rng, tries=10):
    """(image, pasted box) with the patch somewhere it is visible and touches no labelled object, or None."""
    h, w = img.shape[:2]
    _, _, (_, _, bw, bh), _ = patch
    for _ in range(tries):
        s = rng.uniform(*SCALE)
        if bw * s < MIN_PASTED_W:
            s = min(1.0, MIN_PASTED_W / bw)
        x, y = rng.uniform(0, w - bw * s), rng.uniform(0, h - bh * s)
        grown = (x - 8, y - 8, bw * s + 16, bh * s + 16)
        if any(iou(grown, g) > 0 for g in labels):
            continue
        r = paste(img, patch, x, y, s)
        if r is not None:
            return r
    return None


def main(seed=0):
    rng = random.Random(seed)
    tiles = OUT / "paste_tiles"
    if tiles.exists():
        shutil.rmtree(tiles)
    (tiles / "images").mkdir(parents=True)
    (tiles / "labels").mkdir()
    samples, widths = [], []
    for src, folders in SOURCES.items():
        ps, tried, n_videos = patches(folders, rng)
        print(f"{src} drones: {tried} boxes of >= {MIN_W} px tried, {len(ps)} clean mattes from {n_videos} videos "
              f"(the rest had clutter behind them or did not stand out)")
        if src == "antiuav":                                   # Halmstad is daylight: a drone filmed at night is
            day = [p for p in ps if p[3].mean() >= NIGHT]      # lights on black, and has no place in a day scene
            print(f"   {len(ps) - len(day)} night patches left out (old background darker than {NIGHT})")
            ps = day
        targets = [pair for folder in TARGETS[src] for pair in split_pairs(folder)]
        rng.shuffle(targets)
        made = 0
        for ip, lp in targets:
            if made >= PER_DIRECTION:
                break
            img = cv2.imread(str(ip))
            h, w = img.shape[:2]
            labels = read_yolo_labels(lp.read_text() if lp.exists() else "", w, h)
            r = compose(img, labels, rng.choice(ps), rng)
            if r is None:
                continue
            out, box = r
            name = f"{ip.stem}__paste_{src}"
            cv2.imwrite(str(tiles / "images" / f"{name}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (tiles / "labels" / f"{name}.txt").write_text(yolo_lines(labels + [tuple(box) + (0,)], w, h))
            made += 1
            widths.append(box[2])
            if len(samples) < 48 and made % 60 == 1:
                samples.append((out, box))
        print(f"   -> {made} tiles: {src} drones pasted into {', '.join(f.name for f in TARGETS[src])}")
    print(f"pasted drone width px, 10th / 50th / 90th percentile: {np.percentile(widths, [10, 50, 90]).round(0).tolist()}")
    base = yaml.safe_load((D / "e14" / "data_e14.yaml").read_text())
    base["train"].append((tiles / "images").as_posix())
    (OUT / "data_e22c.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    cells = []
    for img, (x, y, bw, bh) in samples:                       # contact sheet: 160-px windows around the pasted drone
        cx, cy = int(x + bw / 2), int(y + bh / 2)
        x0, y0 = int(np.clip(cx - 80, 0, img.shape[1] - 160)), int(np.clip(cy - 80, 0, img.shape[0] - 160))
        cells.append(img[y0:y0 + 160, x0:x0 + 160])
    rows = [np.hstack(cells[i:i + 8]) for i in range(0, len(cells) - len(cells) % 8, 8)]
    cv2.imwrite(str(OUT / "sheet.png"), np.vstack(rows))
    print(f"wrote {tiles}, {OUT / 'data_e22c.yaml'}, {OUT / 'sheet.png'}")


if __name__ == "__main__":
    main()
