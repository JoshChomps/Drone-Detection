"""
Cutting large frames into detector-sized tiles without shrinking the targets (doc 13 section 3.1).

Training and inference use the same tile size, so a 10-px drone in the frame is a 10-px drone for the
network. Downscaling a 4K frame to 640 px instead would shrink it to under 2 px.
"""
import numpy as np


def tile_origins(length: int, tile: int, overlap: float = 0.2) -> list:
    """Start offsets covering [0, length) with tiles of `tile` px; the last tile is flush with the end."""
    if length <= tile:
        return [0]
    stride = max(1, int(tile * (1 - overlap)))
    starts = list(range(0, length - tile, stride))
    return starts + [length - tile]


def grid_tiles(w: int, h: int, tile: int = 640, overlap: float = 0.2) -> list:
    """(x0, y0) origins of a full scan of a w x h frame (a 3840x2160 frame gives 32 tiles at 640)."""
    return [(x, y) for y in tile_origins(h, tile, overlap) for x in tile_origins(w, tile, overlap)]


def crops_around(boxes, w: int, h: int, tile: int = 640, margin: int = 32) -> list:
    """One tile origin per box, centred on it and clipped to the frame. A box already inside an earlier
    tile (with `margin` px to spare) reuses that tile, so clustered proposals cost one crop."""
    origins = []
    for x, y, bw, bh, *_ in boxes:
        if any(ox + margin <= x and x + bw <= ox + tile - margin and oy + margin <= y and y + bh <= oy + tile - margin
               for ox, oy in origins):
            continue
        ox = int(np.clip(x + bw / 2 - tile / 2, 0, max(w - tile, 0)))
        oy = int(np.clip(y + bh / 2 - tile / 2, 0, max(h - tile, 0)))
        if (ox, oy) not in origins:     # clipping at a frame edge can repeat an origin (16 #40)
            origins.append((ox, oy))
    return origins


def boxes_in_tile(boxes, x0: int, y0: int, tile_w: int, tile_h: int, min_visible: float = 0.5) -> list:
    """Boxes clipped to a tile, in tile coordinates. Drop a box if less than `min_visible` of it is inside."""
    out = []
    for x, y, w, h, *rest in boxes:
        cx0, cy0 = max(x, x0), max(y, y0)
        cx1, cy1 = min(x + w, x0 + tile_w), min(y + h, y0 + tile_h)
        if cx1 <= cx0 or cy1 <= cy0 or (cx1 - cx0) * (cy1 - cy0) < min_visible * w * h:
            continue
        out.append((cx0 - x0, cy0 - y0, cx1 - cx0, cy1 - cy0, *rest))
    return out


def training_tiles(w: int, h: int, boxes, tile: int = 640, n_negative: int = 1, rng=None) -> list:
    """Tiles for one annotated frame: [(x0, y0, boxes_in_tile_coords)].

    One tile per box, placed at a random offset so targets are not always centred, plus `n_negative`
    random tiles that touch no box (background: sky, clouds, trees) so the detector learns what to ignore.
    """
    rng = rng or np.random.default_rng()
    tw, th = min(tile, w), min(tile, h)
    tiles, covered = [], []
    for x, y, bw, bh, *_ in boxes:
        if any(ox <= x and x + bw <= ox + tw and oy <= y and y + bh <= oy + th for ox, oy in covered):
            continue
        # A box may start off-frame (a target leaving the view): place the tile around the part inside (16 #54).
        lo, hi = max(0, int(min(x + bw, w)) - tw), min(int(max(x, 0)), w - tw)
        ox = int(rng.integers(lo, max(hi, lo) + 1)) if bw < tw else int(np.clip(x, 0, w - tw))
        lo, hi = max(0, int(min(y + bh, h)) - th), min(int(max(y, 0)), h - th)
        oy = int(rng.integers(lo, max(hi, lo) + 1)) if bh < th else int(np.clip(y, 0, h - th))
        covered.append((ox, oy))
    for _ in range(n_negative):
        for _attempt in range(20):
            ox, oy = int(rng.integers(0, w - tw + 1)), int(rng.integers(0, h - th + 1))
            if not boxes_in_tile(boxes, ox, oy, tw, th, min_visible=1e-9):
                covered.append((ox, oy))
                break
    for ox, oy in covered:
        tiles.append((ox, oy, boxes_in_tile(boxes, ox, oy, tw, th)))
    return tiles


def yolo_lines(boxes, tile_w: int, tile_h: int) -> str:
    """Ultralytics label file text: one 'cls cx cy w h' line per box, normalised to the tile size."""
    return "".join(f"{int(c)} {(x + w / 2) / tile_w:.6f} {(y + h / 2) / tile_h:.6f} {w / tile_w:.6f} {h / tile_h:.6f}\n"
                   for x, y, w, h, c in boxes)


def tile_split(src, dst, tile: int = 640, seed: int = 0, jpg_quality: int = 95) -> int:
    """Cut every labelled image of a split (images/ + labels/) into training tiles at true pixel size: one tile per
    target at a random offset, and NO background tiles. For datasets that label only drones (e.g. Anti-UAV), the
    empty parts of a frame can hold unlabelled birds or aircraft, so they must not be taught as "nothing".
    Returns the number of tiles written."""
    import cv2
    from pathlib import Path
    src, dst = Path(src), Path(dst)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    rng, n = np.random.default_rng(seed), 0
    for img_path in sorted((src / "images").iterdir()):
        lbl = src / "labels" / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        boxes = read_yolo_labels(lbl.read_text() if lbl.exists() else "", w, h)
        if not boxes:
            continue
        for ox, oy, tb in training_tiles(w, h, boxes, tile, n_negative=0, rng=rng):
            if not tb:      # the target is mostly outside the frame: an unlabelled tile would teach it as "nothing"
                continue
            crop = img[oy:oy + tile, ox:ox + tile]
            stem = f"{img_path.stem}_{ox}_{oy}"
            cv2.imwrite(str(dst / "images" / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
            (dst / "labels" / f"{stem}.txt").write_text(yolo_lines(tb, crop.shape[1], crop.shape[0]))
            n += 1
    return n


def read_yolo_labels(text: str, img_w: int, img_h: int) -> list:
    """Inverse of yolo_lines: pixel boxes (x, y, w, h, cls)."""
    out = []
    for line in text.split("\n"):
        if line.strip():
            c, cx, cy, w, h = line.split()[:5]
            w, h = float(w) * img_w, float(h) * img_h
            out.append((float(cx) * img_w - w / 2, float(cy) * img_h - h / 2, w, h, int(c)))
    return out
