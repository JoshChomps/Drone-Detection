"""
Build the E12 multi-dataset training set (Halmstad clean + Anti-UAV300 train split) on the local machine.

Rules applied (03 section 2.1): split by video/sequence (Anti-UAV's own train/val/test sequence split);
no background examples from Anti-UAV (it labels only drones); tile at true size (never downscale); the
burned-in on-screen display is inpainted with the same mask as val/test; small drones via zoom-out mosaics.
Anti-UAV TEST is never touched here: it stays held out for the final E12 evaluation.

    C:\\cuas_venv\\Scripts\\python experiments/build_e12.py
"""
import collections
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.datasets.antiuav import convert as convert_antiuav, inpaint_split  # noqa: E402
from cuas.tiling import tile_split  # noqa: E402
from cuas.zoomout import make_split  # noqa: E402

D = Path("C:/cuas_data")
ZIP = (Path.home() / "Downloads/Anti-UAV-RGBT.zip")
E12 = D / "e12"
NAMES = {0: "drone", 1: "bird", 2: "airplane", 3: "helicopter"}


def class_counts(split_dirs):
    c = collections.Counter()
    n_img = 0
    for d in split_dirs:
        for lbl in (Path(d) / "labels").glob("*.txt"):
            n_img += 1
            for line in lbl.read_text().splitlines():
                if line.strip():
                    c[NAMES[int(line.split()[0])]] += 1
    return n_img, dict(c)


def main():
    E12.mkdir(parents=True, exist_ok=True)
    raw, frames = D / "antiuav_raw", D / "antiuav_train_frames"

    if not (frames / "images").exists():
        z = zipfile.ZipFile(ZIP)
        members = [n for n in z.namelist() if n.startswith("train/") and "/visible." in n]
        print(f"1. extracting {len(members)} train visible files")
        for n in members:
            z.extract(n, raw)
        print("2. building train frames (every 40th frame with a drone, no absent frames)")
        convert_antiuav(raw, "train", frames, every=40, absent_every=0)
        shutil.rmtree(raw)                      # frames are built; the raw video can be re-extracted from the zip
        print("   deleted extracted train video")
        print("3. inpainting the on-screen display (same mask as val/test)")
        mask = cv2.imread(str(D / "antiuav_overlay_mask.png"), cv2.IMREAD_GRAYSCALE)
        print("  ", inpaint_split(frames, mask))

    tiles = E12 / "antiuav_train_tiles"
    if not (tiles / "images").exists():
        print("4. tiling train frames (positive tiles only):", tile_split(frames, tiles), "tiles")
    val_tiles = E12 / "antiuav_val_tiles"
    if not (val_tiles / "images").exists():
        print("7. tiling val frames for Ultralytics' in-training validation:", tile_split(D / "antiuav_val", val_tiles), "tiles")

    zoom_train = []
    for k in (2, 4, 8, 16):                     # Anti-UAV drones ~110 px -> ~55 / 28 / 14 / 7 px
        out = E12 / "antiuav_zoom_train" / f"k{k}"
        if not (out / "images").exists():
            print(f"5. Anti-UAV zoom k={k}:", make_split(tiles, out, k), "mosaics")
        zoom_train.append(out)
    for k in (2, 3, 4, 6):                      # same as E2
        out = D / "halmstad_zoom_train" / f"k{k}"
        if not (out / "images").exists():
            print(f"6. Halmstad zoom k={k}:", make_split(D / "halmstad_yolo" / "train", out, k), "mosaics")
        zoom_train.append(out)

    train = [D / "halmstad_yolo" / "train", tiles] + zoom_train
    val = [D / "halmstad_yolo" / "val", val_tiles]
    cfg = {"train": [str(p / "images").replace("\\", "/") for p in train],
           "val": [str(p / "images").replace("\\", "/") for p in val],
           "names": NAMES}
    (E12 / "data_e12.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("8. wrote", E12 / "data_e12.yaml")
    for tag, dirs in (("train", train), ("val", val)):
        n_img, c = class_counts(dirs)
        print(f"   {tag}: {n_img} images, instances {c}")
        for d in dirs:
            print(f"      {len(list((Path(d) / 'images').iterdir())):6d}  {d}")


if __name__ == "__main__":
    main()
