"""
Simulated distance: shrink real frames k-fold and tile k x k of them into one image of the original size.

A drone that is 23 px wide in the source becomes ~4-12 px at k = 2-6, and the detector sees it at that true
pixel size (a single shrunken frame would be scaled back UP by the detector's letterbox, defeating the point).
Used to measure recall below the sizes a dataset contains (Halmstad has no drones under ~20 px), and as
training data for small targets (experiment E2).

Caveat: area-downscaling averages away sensor noise and blur, so shrunken targets are cleaner than real
distant ones and w_min measured this way is OPTIMISTIC. `noise` adds Gaussian noise back as a partial check.

    python -m cuas.zoomout --src data/halmstad_yolo/val --dst data/zoom_val --k 2 3 4 6
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from cuas.eval import split_pairs
from cuas.tiling import read_yolo_labels, yolo_lines


def mosaic(images, box_lists, k: int, out_w: int, out_h: int):
    """k*k images, each shrunk to (out_w // k, out_h // k), in a grid. Returns (image, boxes)."""
    cw, ch = out_w // k, out_h // k
    canvas = np.zeros((out_h, out_w, 3), np.uint8)
    boxes = []
    for n, (img, bl) in enumerate(zip(images, box_lists)):
        r, c = divmod(n, k)
        ih, iw = img.shape[:2]
        canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
        sx, sy = cw / iw, ch / ih
        boxes += [(x * sx + c * cw, y * sy + r * ch, w * sx, h * sy, cls) for x, y, w, h, cls in bl]
    return canvas, boxes


def make_split(src, dst, k: int, seed: int = 0, noise: float = 0.0, jpg_quality: int = 95, jpg_params=None) -> int:
    """Write a zoomed-out copy of an Ultralytics split. Returns the number of mosaics written.
    jpg_params: full cv2.imwrite parameters instead of the quality alone (E23 frame stacks need 4:4:4 sampling)."""
    pairs = list(split_pairs(src))
    np.random.default_rng(seed).shuffle(pairs)
    dst = Path(dst)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 1)
    n_out = 0
    for start in range(0, len(pairs) - k * k + 1, k * k):          # drop the incomplete last group
        imgs, boxes = [], []
        for img_path, lbl_path in pairs[start:start + k * k]:
            img = cv2.imread(str(img_path))
            imgs.append(img)
            boxes.append(read_yolo_labels(lbl_path.read_text() if lbl_path.exists() else "",
                                          img.shape[1], img.shape[0]))
        h, w = imgs[0].shape[:2]
        canvas, bl = mosaic(imgs, boxes, k, w, h)
        if noise:
            canvas = np.clip(canvas + rng.normal(0, noise, canvas.shape), 0, 255).astype(np.uint8)
        stem = f"zoom{k}_{n_out:05d}"
        cv2.imwrite(str(dst / "images" / f"{stem}.jpg"), canvas, jpg_params or [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        (dst / "labels" / f"{stem}.txt").write_text(yolo_lines(bl, w, h))
        n_out += 1
    return n_out


def main():
    ap = argparse.ArgumentParser(description="Make zoomed-out (simulated distance) copies of a YOLO split.")
    ap.add_argument("--src", required=True, help="split folder with images/ and labels/")
    ap.add_argument("--dst", required=True, help="output root; one sub-folder per k (k2, k3, ...)")
    ap.add_argument("--k", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--noise", type=float, default=0.0, help="Gaussian noise sigma added after shrinking")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for k in a.k:
        n = make_split(a.src, Path(a.dst) / f"k{k}", k, a.seed, a.noise)
        print(f"k={k}: {n} mosaics -> {Path(a.dst) / f'k{k}'}")


if __name__ == "__main__":
    main()
