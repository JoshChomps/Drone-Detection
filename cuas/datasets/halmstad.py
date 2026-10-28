"""
Halmstad multi-sensor drone dataset (Svanstrom et al. 2021, CC0) -> Ultralytics YOLO tiles.
https://github.com/DroneDetectionThesis/Drone-detection-dataset  (357 MB; visible videos in Data/Video_V)

Files are Data/Video_V/V_<CLASS>_<NNN>.mp4 with V_<CLASS>_<NNN>_LABELS.mat, one labelled target per frame,
CLASS in DRONE / BIRD / AIRPLANE / HELICOPTER. The .mat labels need `pip install mcos-decoder`.

The split is by VIDEO, stratified per class: neighbouring frames are near-duplicates, so a frame-level
split would leak the validation set into training and overstate every number.

A frame with no label does NOT mean "no target visible": inspecting E2's false alarms (2026-09-24) showed
real, unlabelled drones in such frames. So no background examples are taken from Halmstad by default
(`neg_every=0`); true empty-sky negatives must come from footage verified to be empty.

    python -m cuas.datasets.halmstad --src <clone>/Data/Video_V --out <data>/halmstad_yolo
"""
import argparse
import re
from pathlib import Path

import numpy as np

from cuas.tiling import training_tiles, yolo_lines

CLASSES = ["drone", "bird", "airplane", "helicopter"]     # drone is class 0 everywhere in this project
NAME_RE = re.compile(r"V_(DRONE|BIRD|AIRPLANE|HELICOPTER)_(\d+)\.mp4$", re.I)


def split_videos(paths, val_frac: float = 0.2, seed: int = 0) -> dict:
    """{path: 'train' | 'val'}, stratified by class, deterministic for a given seed."""
    by_cls = {}
    for p in sorted(paths, key=str):
        m = NAME_RE.search(Path(p).name)
        if m:
            by_cls.setdefault(m.group(1).lower(), []).append(p)
    rng, out = np.random.default_rng(seed), {}
    for cls in sorted(by_cls):
        vids = list(by_cls[cls])
        rng.shuffle(vids)
        n_val = max(1, round(len(vids) * val_frac)) if len(vids) > 1 else 0
        out.update({v: ("val" if i < n_val else "train") for i, v in enumerate(vids)})
    return out


def convert(src, out, every: int = 5, val_frac: float = 0.2, seed: int = 0, tile: int = 640,
            neg_every: int = 0, jpg_quality: int = 95):
    import cv2
    from mcos_decoder import load_groundtruth

    src, out = Path(src), Path(out)
    videos = sorted(src.glob("V_*.mp4"))
    if not videos:
        raise SystemExit(f"No V_*.mp4 videos in {src}")
    split = split_videos(videos, val_frac, seed)
    rng = np.random.default_rng(seed)
    counts, short = {"train": 0, "val": 0}, []
    for vid in videos:
        m = NAME_RE.search(vid.name)
        if not m:
            continue
        cls = CLASSES.index(m.group(1).lower())
        labels = load_groundtruth(str(vid.with_name(vid.stem + "_LABELS.mat")))
        part = split[vid]
        (out / part / "images").mkdir(parents=True, exist_ok=True)
        (out / part / "labels").mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(vid))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames - len(labels) > 1:
            short.append(f"{vid.name} ({len(labels)}/{n_frames})")
        i = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i >= len(labels):
                # Some videos stop being labelled early (V_BIRD_005: 1 label, 302 frames). Past the end the
                # target may still be visible, so these frames are neither positives nor background.
                break
            box = labels[i]
            # Every `every`-th labelled frame. Unlabelled frames are NOT reliable background (they can hold
            # unlabelled drones), so they are only used when explicitly asked for, and never from bird videos.
            background = (box is None and neg_every > 0 and cls != CLASSES.index("bird")
                          and i % neg_every == 0)
            if (box is not None and i % every) or (box is None and not background):
                continue
            h, w = frame.shape[:2]
            boxes = [(*box, cls)] if box is not None else []
            for ox, oy, tb in training_tiles(w, h, boxes, tile, n_negative=0 if boxes else 1, rng=rng):
                crop = frame[oy:oy + tile, ox:ox + tile]
                stem = f"{vid.stem}_{i:05d}_{ox}_{oy}"
                cv2.imwrite(str(out / part / "images" / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
                (out / part / "labels" / f"{stem}.txt").write_text(yolo_lines(tb, crop.shape[1], crop.shape[0]))
                counts[part] += 1
        cap.release()
    if short:
        print(f"[!] {len(short)} videos have fewer labels than frames; their unlabelled tails were skipped: "
              + ", ".join(short[:8]) + (" ..." if len(short) > 8 else ""))
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: train/images\nval: val/images\n"
        + "names:\n" + "".join(f"  {k}: {n}\n" for k, n in enumerate(CLASSES)))
    print(f"Wrote {counts['train']} train / {counts['val']} val tiles to {out}")


def main():
    ap = argparse.ArgumentParser(description="Convert Halmstad visible videos to a YOLO tile dataset.")
    ap.add_argument("--src", required=True, help="path to Data/Video_V")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=5, help="keep every Nth labelled frame")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neg_every", type=int, default=0,
                    help="take every Nth UNLABELLED frame as background (0 = off; unlabelled != empty in Halmstad)")
    a = ap.parse_args()
    convert(a.src, a.out, a.every, a.val_frac, a.seed, neg_every=a.neg_every)


if __name__ == "__main__":
    main()
