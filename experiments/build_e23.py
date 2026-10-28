"""
E23 data: every training and evaluation image of the E14 recipe rebuilt as a temporal stack (cuas/temporal.py):
grey frames t-2k, t-k, t, the past two registered to t, as the three channels. Same file names, crops and labels as
the single-frame sets, so runs compare image for image; the zoom-out mosaics are rebuilt from the stacked sources
with the same seed, so they have the same layouts.

Sources: Halmstad Video_V/*.mp4 (30 fps) and Anti-UAV-RGBT.zip visible.mp4 per sequence (frames inpainted with the
same overlay mask as the single-frame sets). Frame i of a tile name is the i-th frame decoded by OpenCV, as in the
converters. Frames before 0 are clamped to 0 (no motion shown). Check: channel 2 of every stack (frame t) is compared
with the grey of the existing image; mismatches are listed (the existing images are JPEG, so a mean absolute
difference of ~1-3 grey levels is expected).

The control (E23ctl, frame t in all three channels) is derived from the stacks with --static, on Colab, so only one
set is uploaded.

    C:\\cuas_venv\\Scripts\\python experiments/build_e23.py --k 3            (~30-60 min, 12 processes)
    python experiments/build_e23.py --static C:/cuas_data/e23_k3 C:/cuas_data/e23_k0
Output: C:/cuas_data/e23_k<k>/ mirroring C:/cuas_data (halmstad_yolo/train, e12/antiuav_train_tiles, ...),
data_e23.yaml, build_e23.json (counts, registration shifts, check)
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.eval import split_pairs  # noqa: E402
from cuas.temporal import JPEG_444, stack, static  # noqa: E402
from cuas.zoomout import make_split  # noqa: E402

D = Path("C:/cuas_data")
ZIP = (Path.home() / "Downloads/Anti-UAV-RGBT.zip")
HALMSTAD = D / "halmstad" / "Data" / "Video_V"
MASK = D / "antiuav_overlay_mask.png"
TILE = 640
SPLITS = ["halmstad_yolo/train", "halmstad_yolo/val", "e12/antiuav_train_tiles", "e12/antiuav_val_tiles",
          "e14/antiuav_hardneg_tiles", "antiuav_val", "antiuav_test"]
ZOOMS = [("e12/antiuav_zoom_train", "e12/antiuav_train_tiles", (2, 4, 8, 16)),
         ("halmstad_zoom_train", "halmstad_yolo/train", (2, 3, 4, 6)),
         ("halmstad_zoom", "halmstad_yolo/val", (2, 3, 4, 6)),
         ("antiuav_zoom_val", "antiuav_val", (4, 8, 12, 16)),
         ("antiuav_zoom_test", "antiuav_test", (4, 8, 12, 16))]
TRAIN = ["halmstad_yolo/train", "e12/antiuav_train_tiles"] + \
        [f"e12/antiuav_zoom_train/k{k}" for k in (2, 4, 8, 16)] + \
        [f"halmstad_zoom_train/k{k}" for k in (2, 3, 4, 6)] + ["e14/antiuav_hardneg_tiles"]   # = data_e14.yaml
VAL = ["halmstad_yolo/val", "e12/antiuav_val_tiles"]
NAMES = {0: "drone", 1: "bird", 2: "airplane", 3: "helicopter"}
HAL_RE = re.compile(r"^(V_[A-Z]+_\d+)_(\d{5})_(\d+)_(\d+)$")
AUV_RE = re.compile(r"^(\d{8}_\d{6}_\d+_\d+)_(\d{5})(?:_neg)?(?:_(\d+)_(\d+))?$")


def jobs_by_video():
    """{video key: [(split, image path, frame i, ox, oy or None for a whole frame)]}."""
    out = {}
    for split in SPLITS:
        for ip, _ in split_pairs(D / split):
            m = HAL_RE.match(ip.stem) or AUV_RE.match(ip.stem)
            if not m:
                raise SystemExit(f"cannot parse {ip}")
            key = ("halmstad" if m.re is HAL_RE else "antiuav", m.group(1))
            ox = None if m.group(3) is None else (int(m.group(3)), int(m.group(4)))
            out.setdefault(key, []).append((split, str(ip), int(m.group(2)), ox))
    return out


def video_file(key, tmp):
    kind, name = key
    if kind == "halmstad":
        return HALMSTAD / f"{name}.mp4"
    with zipfile.ZipFile(ZIP) as z:
        member = next(n for n in z.namelist() if n.endswith(f"/{name}/visible.mp4"))
        return Path(z.extract(member, tmp))


def build_video(args):
    key, jobs, k, root = args
    root, mask = Path(root), cv2.imread(str(MASK), cv2.IMREAD_GRAYSCALE)
    need = {max(0, i - d) for _, _, i, _ in jobs for d in (0, k, 2 * k)}
    with tempfile.TemporaryDirectory() as tmp:
        cap = cv2.VideoCapture(str(video_file(key, tmp)))
        frames, i = {}, -1
        while i < max(need):
            ok, f = cap.read()
            if not ok:
                break
            i += 1
            if i in need:
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                frames[i] = cv2.inpaint(g, mask, 3, cv2.INPAINT_TELEA) if key[0] == "antiuav" else g
        cap.release()
    stacks, res = {}, {"written": 0, "missing_frame": [], "mad": [], "shifts": []}
    for split, ip, i, ox in jobs:
        if i not in frames:
            res["missing_frame"].append(ip)
            continue
        if i not in stacks:
            stacks[i], sh = stack(frames[max(0, i - 2 * k)], frames[max(0, i - k)], frames[i])
            res["shifts"].append([abs(v) for s in sh for v in s])
        img = stacks[i] if ox is None else stacks[i][ox[1]:ox[1] + TILE, ox[0]:ox[0] + TILE]
        old = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        res["mad"].append((float(np.abs(img[:, :, 2].astype(np.int16) - old).mean()) if old.shape == img.shape[:2]
                           else 255.0, ip))
        dst = root / split
        (dst / "images").mkdir(parents=True, exist_ok=True)
        (dst / "labels").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst / "images" / Path(ip).name), img, JPEG_444)
        lbl = Path(ip).parents[1] / "labels" / (Path(ip).stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, dst / "labels" / lbl.name)
        res["written"] += 1
    return key, res


CUT_THR = 8.0   # grey levels: median |frame t - past frame| above this = a cut, zoom or exposure jump (16 #84)


def reset_cuts(root, thr=CUT_THR):
    """Stacks whose past frames differ from frame t over the whole image (scene cuts and zoom changes in Halmstad,
    exposure jumps in Anti-UAV; ~1%) carry no usable motion, only a full-frame change. A live pipeline restarts its
    frame buffer on such a jump, so these become static stacks (frame t in all three channels). Source splits only;
    the zoom-out mosaics are built afterwards from them."""
    n = total = 0
    for split in SPLITS:
        for ip in sorted((root / split / "images").glob("*.jpg")):
            im = cv2.imread(str(ip))
            t = im[:, :, 2].astype(np.int16)
            total += 1
            if max(np.median(np.abs(t - im[:, :, 1])), np.median(np.abs(t - im[:, :, 0]))) > thr:
                cv2.imwrite(str(ip), static(im), JPEG_444)
                n += 1
    print(f"reset {n} of {total} stacks to static (a cut or exposure jump; median change > {thr} grey levels)")
    return n


def write_yaml(root):
    import yaml
    cfg = {"train": [f"{root.as_posix()}/{s}/images" for s in TRAIN],
           "val": [f"{root.as_posix()}/{s}/images" for s in VAL], "names": NAMES}
    (root / "data_e23.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def zooms(root):
    for dst, src, ks in ZOOMS:
        for k in ks:
            out = root / dst / f"k{k}"
            if not (out / "images").exists():
                n = make_split(root / src, out, k, jpg_params=JPEG_444)
                ref = len(list((D / dst / f"k{k}" / "images").glob("*.jpg")))
                print(f"  zoom {dst}/k{k}: {n} mosaics (single-frame set: {ref})", flush=True)


def build(k, workers, limit=0):
    root = D / (f"e23_k{k}" + ("_smoke" if limit else ""))
    jobs = jobs_by_video()
    if limit:                                   # smoke test: a few videos of each kind, no zoom sets
        hal = [j for j in jobs if j[0] == "halmstad"][:limit]
        jobs = {j: jobs[j] for j in hal + [j for j in jobs if j[0] == "antiuav"][:limit]}
    print(f"{sum(map(len, jobs.values()))} images from {len(jobs)} videos -> {root}", flush=True)
    summary = {"k": k, "videos": len(jobs), "written": 0, "missing_frame": [], "worst_mad": [], "shift_px": {}}
    mads, shifts, done = [], [], 0
    with ProcessPoolExecutor(workers) as ex:
        for key, res in ex.map(build_video, [(key, j, k, str(root)) for key, j in jobs.items()]):
            done += 1
            summary["written"] += res["written"]
            summary["missing_frame"] += res["missing_frame"]
            mads += res["mad"]
            shifts += res["shifts"]
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} videos", flush=True)
    m = np.array([a for a, _ in mads])
    s = np.array(shifts)
    summary["mad"] = {"median": float(np.median(m)), "p99": float(np.percentile(m, 99)), "max": float(m.max()),
                      "over_6": int((m > 6).sum())}
    summary["worst_mad"] = sorted(mads, reverse=True)[:20]
    summary["shift_px"] = {"median": float(np.median(s)), "p95": float(np.percentile(s, 95)), "max": float(s.max())}
    summary["reset_to_static"] = reset_cuts(root)
    if not limit:
        zooms(root)
    write_yaml(root)
    (root / "build_e23.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({x: summary[x] for x in ("written", "mad", "shift_px")}, indent=1))
    print(f"missing frames: {len(summary['missing_frame'])}; worst check: {summary['worst_mad'][:3]}")


def make_static(src, dst):
    """E23ctl: every image of a stack root with frame t in all three channels; labels and yaml copied."""
    src, dst = Path(src), Path(dst)
    n = 0
    for ip in src.rglob("images/*.jpg"):
        out = dst / ip.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), static(cv2.imread(str(ip))), JPEG_444)
        n += 1
    for lp in src.rglob("labels/*.txt"):
        out = dst / lp.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(lp, out)
    for y in src.glob("*.yaml"):
        (dst / y.name).write_text(y.read_text().replace(src.as_posix(), dst.as_posix()))
    print(f"{n} static images -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="frame interval (Halmstad is 30 fps: 3 = 0.1 s)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0, help="smoke test: this many videos of each source")
    ap.add_argument("--static", nargs=2, metavar=("SRC", "DST"), help="derive the E23ctl set from a stack root")
    a = ap.parse_args()
    if a.static:
        make_static(*a.static)
    else:
        build(a.k, a.workers, a.limit)
