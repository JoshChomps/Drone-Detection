"""
E39 data: the DUT Anti-UAV detection TRAIN split as extra training data (19 section 9: the 2025 Drone-vs-Bird winner
trained on DUT; 19 section 3.9: the third lever for varied terrain). Training on it ends DUT's role as the unseen
dataset: from then on DUT test is "seen dataset, unseen images", like the Anti-UAV test, and the unseen yardstick must
be another set (DUT's tracking videos, Drone-vs-Bird, own footage; doc 20). So E39 runs LAST, after every arm that is
chosen on DUT test.

Tiles: one 640-px tile per drone at true pixel size (cuas.tiling.training_tiles, no background tiles: DUT labels only
UAVs, so empty parts may hold unlabelled birds or aircraft). Images or tiles smaller than 640 are padded with grey
(114, Ultralytics' letterbox value) to 640 x 640, bottom/right, so Ultralytics never scales a small image up (E15 pads
its test images the same way).

    1. download train.zip and val.zip from the DUT repo's Google Drive links into ~/Downloads
    2. python experiments/build_e39.py            (~10 min, CPU)
Output: C:/cuas_data/dut_train, dut_val (converted), C:/cuas_data/e39/dut_train_tiles, data_e39.yaml
(E22c's training list + the DUT tiles; validation as E14).
"""
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.datasets.dut import convert  # noqa: E402
from cuas.eval import split_pairs  # noqa: E402
from cuas.tiling import read_yolo_labels, training_tiles, yolo_lines  # noqa: E402

D = Path("C:/cuas_data")
DL = (Path.home() / "Downloads")
TILE = 640


def unpack(split):
    out = D / f"dut_{split}"
    if (out / "images").exists():
        return out
    z = DL / f"{split}.zip"
    if not z.exists():
        raise SystemExit(f"{z} not found: download the DUT {split} split (repo wangdongdut/DUT-Anti-UAV, Google Drive)")
    raw = D / "dut_raw" / f"_{split}"            # its own folder: convert() searches it recursively
    with zipfile.ZipFile(z) as zf:
        zf.extractall(raw)
    print(f"{split}:", convert(raw, out))
    return out


def tiles(src, dst, seed=0):
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    rng, n, widths = np.random.default_rng(seed), 0, []
    for ip, lp in split_pairs(src):
        img = cv2.imread(str(ip))
        h, w = img.shape[:2]
        boxes = read_yolo_labels(lp.read_text() if lp.exists() else "", w, h)
        for ox, oy, tb in training_tiles(w, h, boxes, TILE, n_negative=0, rng=rng):
            if not tb:
                continue
            crop = img[oy:oy + TILE, ox:ox + TILE]
            canvas = np.full((TILE, TILE, 3), 114, np.uint8)
            canvas[:crop.shape[0], :crop.shape[1]] = crop
            stem = f"dut_{ip.stem}_{ox}_{oy}"
            cv2.imwrite(str(dst / "images" / f"{stem}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (dst / "labels" / f"{stem}.txt").write_text(yolo_lines(tb, TILE, TILE))
            widths += [b[2] for b in tb]
            n += 1
    w = np.array(widths)
    print(f"{n} tiles, {len(w)} drones; width px median {np.median(w):.0f}, "
          f"<16: {(w < 16).mean():.2f}, 16-64: {((w >= 16) & (w < 64)).mean():.2f}, >=64: {(w >= 64).mean():.2f}")
    return n


def main():
    train = unpack("train")
    unpack("val")                                  # converted for later use (threshold or check); not trained on
    out = D / "e39" / "dut_train_tiles"
    if not (out / "images").exists():
        tiles(train, out)
    base = yaml.safe_load((D / "e22c" / "data_e22c.yaml").read_text())
    base["train"] = base["train"] + [str(out / "images").replace("\\", "/")]
    (D / "e39" / "data_e39.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    print("wrote", D / "e39" / "data_e39.yaml")


if __name__ == "__main__":
    main()
