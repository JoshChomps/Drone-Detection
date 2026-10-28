"""
Anti-UAV (Anti-UAV300 "RGB-T" release; repo MIT) visible videos -> full frames + YOLO labels, for evaluation.
https://github.com/ZhaoJ9014/Anti-UAV  (Google Drive zip "Anti-UAV-RGBT.zip", 6.0 GB)

Zip layout (checked 2026-09-24): {train,val,test}/<sequence>/visible.mp4 + visible.json, plus infrared.*,
label_new/ and framecut.py. visible.json = {"exist": [0/1 per frame], "gt_rect": [[x, y, w, h] per frame]}.
Visible video is 1920x1080 at 20 fps, ~1,000 frames per sequence, one drone per sequence.
Sequences: train 160, val 67, test 91.

What this dataset is (measured 2026-09-24):
  * Drones are LARGE: median width 109-122 px, and only 43 of ~280k visible boxes are under 16 px. It tests
    generalisation to a new camera/site at normal size, not small-target detection.
  * A tracking reticle is burned into the RGB video and the camera follows the drone (drones sit near the
    centre); heavy compression blocks are visible.
  * exist == 0 marks frames where the target is explicitly absent (unlike Halmstad's missing labels), so
    a sample of those is kept as background. Other objects (birds) are never labelled.

  * The burned-in on-screen display (Chinese status text, date/time, reticle) is detected as "airplane" by
    Halmstad-trained models (conf up to 0.8). A deployed camera would have its OSD off, so remove it with
    --inpaint after building (fixed-position mask from persistent edges, Telea inpainting).

Full frames are written, not tiles: evaluate with run_split(..., tile=640) so nothing is downscaled.
Extract only the visible files first, e.g.:  unzip -q Anti-UAV-RGBT.zip 'val/*/visible*' 'test/*/visible*'

    python -m cuas.datasets.antiuav --src <extracted root> --split val --out <data>/antiuav_val --every 40
"""
import argparse
import json
from pathlib import Path

from cuas.tiling import yolo_lines


def convert(src, split, out, every: int = 40, absent_every: int = 200, jpg_quality: int = 95, offset: int = 0) -> dict:
    """Frames i with (i - offset) % every == 0 that show the drone, plus every absent_every-th target-absent frame.
    offset > 0 picks frames between an earlier offset-0 build (E14 mines false alarms on those, 16 #38)."""
    import cv2
    src, out = Path(src), Path(out)
    seqs = sorted(p for p in (src / split).iterdir() if (p / "visible.mp4").exists())
    if not seqs:
        raise SystemExit(f"No */visible.mp4 under {src / split}")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    counts = {"sequences": len(seqs), "with_drone": 0, "absent": 0}
    for seq in seqs:
        lab = json.loads((seq / "visible.json").read_text())
        exist, rects = lab["exist"], lab["gt_rect"]
        cap = cv2.VideoCapture(str(seq / "visible.mp4"))
        i = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i >= len(exist):
                break
            r = rects[i] if i < len(rects) else None
            present = bool(exist[i]) and r is not None and len(r) == 4 and r[2] > 0 and r[3] > 0
            if present and (i - offset) % every == 0:
                boxes = [(*r, 0)]
            elif not exist[i] and absent_every and i % absent_every == 0:
                boxes = []
            else:
                continue
            stem = f"{seq.name}_{i:05d}"
            h, w = frame.shape[:2]
            cv2.imwrite(str(out / "images" / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
            (out / "labels" / f"{stem}.txt").write_text(yolo_lines(boxes, w, h))
            counts["with_drone" if boxes else "absent"] += 1
        cap.release()
    print(f"{split}: {counts['sequences']} sequences -> {counts['with_drone']} frames with a drone, "
          f"{counts['absent']} target-absent frames -> {out}")
    return counts


def overlay_mask(image_paths, edge_thr: float = 25, min_frac: float = 0.3, dilate: int = 7):
    """The burned-in on-screen display (text top-left/right and bottom-left, tracking reticle) as a uint8 mask.

    A pixel is overlay if it shows a strong edge (|Laplacian| > edge_thr) in at least `min_frac` of many frames
    from different sequences: the overlay never moves, while sky, clouds and drones do (a drone crosses any one
    pixel in only a few frames). Dilation merges changing digits into solid text blocks and covers anti-aliasing.
    A percentile threshold on the average edge map was tried first and rejected: when the overlay is small, the
    quota fills up with moving-target trails, and the mask would erase drones.
    """
    import cv2
    import numpy as np
    count, n = None, 0
    for p in image_paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        strong = np.abs(cv2.Laplacian(g, cv2.CV_32F)) > edge_thr
        count = strong.astype(np.int32) if count is None else count + strong
        n += 1
    mask = (count >= min_frac * max(n, 1)).astype(np.uint8) * 255
    return cv2.dilate(mask, np.ones((dilate, dilate), np.uint8))


CAPTION_REGION = (0, 940, 320, 1080)     # x0, y0, x1, y1 of the Full HD frame searched for the caption


def caption_template(gray_frame, thr: int = 150, pad: int = 4):
    """Crop the white bottom-left caption (Chinese 'civil aviation aircraft') from a frame known to show it.
    The overlay mask misses it: it appears in only some night sequences (train 31 of 160, val 12 of 67, test 0 of 91,
    measured 2026-09-27), so its pixels are not persistent edges across the dataset (16 #37)."""
    import numpy as np
    x0, y0, x1, y1 = CAPTION_REGION
    reg = gray_frame[y0:y1, x0:x1]
    ys, xs = np.nonzero(reg > thr)
    return reg[max(ys.min() - pad, 0):ys.max() + pad + 1, max(xs.min() - pad, 0):xs.max() + pad + 1]


def find_caption(gray_frame, template, thr: float = 0.6):
    """(x, y, w, h) of the caption in frame pixels, or None. Normalised template match in CAPTION_REGION; real
    frames score > 0.6 or < 0.3 with nothing in between (train sample: 159 vs 734, one borderline)."""
    import cv2
    x0, y0, x1, y1 = CAPTION_REGION
    res = cv2.matchTemplate(gray_frame[y0:y1, x0:x1], template, cv2.TM_CCOEFF_NORMED)
    _, score, _, (mx, my) = cv2.minMaxLoc(res)
    return (x0 + mx, y0 + my, template.shape[1], template.shape[0]) if score > thr else None


def inpaint_split(split_dir, mask, jpg_quality: int = 95) -> dict:
    """Remove the overlay from every image of a split in place (Telea inpainting inside the mask).
    Returns how many labelled drones the mask touches, so any damage to targets is visible."""
    import cv2
    from cuas.eval import split_pairs
    from cuas.tiling import read_yolo_labels
    touched = total = 0
    for img_path, lbl_path in split_pairs(split_dir):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        for x, y, bw, bh, _ in read_yolo_labels(lbl_path.read_text() if lbl_path.exists() else "", w, h):
            total += 1
            touched += bool(mask[int(y):int(y + bh) + 1, int(x):int(x + bw) + 1].any())
        cv2.imwrite(str(img_path), cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA), [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
    return {"drones": total, "drones_touched_by_mask": touched}


def main():
    ap = argparse.ArgumentParser(description="Anti-UAV visible videos -> full frames + YOLO labels (evaluation).")
    ap.add_argument("--src", help="extracted root containing train/ val/ test/")
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--out")
    ap.add_argument("--every", type=int, default=40, help="keep every Nth frame with a drone")
    ap.add_argument("--absent_every", type=int, default=200, help="keep every Nth target-absent frame (0 = none)")
    ap.add_argument("--inpaint", nargs="+", metavar="SPLIT_DIR",
                    help="instead of converting: remove the burned-in overlay from these built splits in place")
    ap.add_argument("--mask", help="overlay mask PNG to use/create (built from the first SPLIT_DIR if missing)")
    a = ap.parse_args()
    if a.inpaint:
        import cv2
        from cuas.eval import split_pairs
        if a.mask and Path(a.mask).exists():
            mask = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
        else:
            mask = overlay_mask([p for p, _ in split_pairs(a.inpaint[0])][::5])
            if a.mask:
                cv2.imwrite(a.mask, mask)
        print(f"overlay mask covers {100 * (mask > 0).mean():.2f}% of the frame")
        for d in a.inpaint:
            print(d, inpaint_split(d, mask))
        return
    if not (a.src and a.split and a.out):
        ap.error("--src, --split and --out are required unless --inpaint is used")
    convert(a.src, a.split, a.out, a.every, a.absent_every)


if __name__ == "__main__":
    main()
