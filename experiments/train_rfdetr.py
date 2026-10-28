"""
E25: train RF-DETR (Apache-2.0 sizes N-L; windowed DINOv2 backbone pretrained far more broadly than COCO) on the same
data as a YOLO run, so it is scored by the same scripts (cuas.detector.load reads cuas_rfdetr.json in the run folder).

Question (19 section 5 rank 5, after E24 showed that keeping pretrained features is what transfers to DUT): does a
foundation-model backbone find more drones in an unseen dataset than YOLO26s with a frozen COCO backbone?
Caveats checked at the source (2026-10-08, rf-detr 1.11.2): patch 16, so an 8-px drone is half a token (small targets
stay YOLO's job; DUT drones are median 55 px); resolution 640 (divisible by 16); every input is resized to a square,
so the 640 x 512 Halmstad frames are stretched 1.25x in training (the detector wrapper pads at inference instead).

RF-DETR reads one YOLO folder per split (Roboflow layout), so the Ultralytics yaml's folder lists are merged into
<CUAS_DATA>/rfdetr_ds/<name>/{train,valid}/{images,labels} as symlinks (Linux/Colab; copies on Windows). Not in the
run folder: on Colab that is on Google Drive, which has no symlinks. RF-DETR's own checkpoints (~0.9 GB a run) are
deleted once the final weights are copied, for the same reason; --keep-checkpoints leaves them.
Resume: a run folder with RF-DETR's last.ckpt (written every epoch) and no cuas_rfdetr.json is an interrupted run, and
the same command continues it from that epoch (an unreadable last.ckpt is set aside and the run starts again).

    python experiments/train_rfdetr.py --data <ultralytics yaml> --name E25_s1 --seed 1 [--variant small]
         [--epochs 20] [--batch 8 --grad-accum 2] [--freeze-encoder] [--fraction 0.02 for a smoke test]
Output: <CUAS_DATA>/runs/<name>/weights/last.pt (final EMA weights), cuas_rfdetr.json (also the training time, which
colab_e18.py uses to fit the epoch count to a time budget), metrics.csv
"""
import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import yaml


def merge(cfg_lists, dst, fraction=1.0, seed=0):
    """Symlink (or copy) every image of the listed Ultralytics folders, and its label, into one folder."""
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    imgs = sorted((i, p) for i, d in enumerate(cfg_lists) for p in Path(d).iterdir() if p.suffix.lower() in {".jpg", ".png"})
    if fraction < 1:
        imgs = random.Random(seed).sample(imgs, max(1, int(len(imgs) * fraction)))
    link = os.symlink if os.name != "nt" else shutil.copy
    for i, p in imgs:
        name = f"{i:02d}_{p.name}"              # folder index: mosaics are named zoom<k>_<n> in several folders
        lbl = p.parent.parent / "labels" / (p.stem + ".txt")
        out = dst / "images" / name
        if not out.exists():
            link(p, out)
            lo = dst / "labels" / (Path(name).stem + ".txt")
            if lbl.exists():
                link(lbl, lo)
            else:
                lo.write_text("")
    return len(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="an Ultralytics data yaml (lists of image folders)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", default="small", choices=["nano", "small", "medium", "large"])
    ap.add_argument("--resolution", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze-encoder", action="store_true", help="the E24 idea for RF-DETR: train only the decoder")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-checkpoints", action="store_true")
    a = ap.parse_args()

    import rfdetr
    from cuas.detector import RFDETRDetector
    root = Path(os.environ.get("CUAS_DATA", "C:/cuas_data"))
    out = root / "runs" / a.name
    cfg = yaml.safe_load(Path(a.data).read_text())
    names = [cfg["names"][k] for k in sorted(cfg["names"])]
    ds = root / "rfdetr_ds" / a.name
    shutil.rmtree(ds, ignore_errors=True)       # a re-run with another --fraction must not inherit old files
    n_train = merge(cfg["train"], ds / "train", a.fraction, a.seed)
    n_val = merge(cfg["val"], ds / "valid", a.fraction, a.seed)
    (ds / "data.yaml").write_text(yaml.safe_dump({"names": names, "nc": len(names), "train": "train/images",
                                                  "val": "valid/images"}, sort_keys=False))
    print(f"E25 RF-DETR {a.variant} @ {a.resolution}: {n_train} train / {n_val} val images, {len(names)} classes",
          flush=True)
    model = getattr(rfdetr, RFDETRDetector.VARIANTS[a.variant])(resolution=a.resolution,
                                                                freeze_encoder=a.freeze_encoder)
    resume = {}
    if (out / "last.ckpt").exists() and not (out / "cuas_rfdetr.json").exists():
        import torch
        try:
            ep = torch.load(out / "last.ckpt", map_location="cpu", weights_only=False).get("epoch")
            resume = {"resume": str(out / "last.ckpt")}
            print(f"E25: resuming {a.name} from last.ckpt (epochs finished: {ep})", flush=True)
        except Exception as e:                  # cut short by a dead runtime
            (out / "last.ckpt").replace(out / "last.corrupt")
            print(f"E25: last.ckpt is unreadable ({type(e).__name__}); starting again", flush=True)
    t0 = time.time()
    model.train(dataset_dir=str(ds), output_dir=str(out), epochs=a.epochs, batch_size=a.batch,
                grad_accum_steps=a.grad_accum, seed=a.seed, num_workers=a.workers, class_names=names,
                early_stopping=False, tensorboard=False, progress_bar=None, **resume)
    secs = time.time() - t0
    # The final EMA weights (last_ema.pth), as YOLO runs are scored on their final weights (19 section 3.4); RF-DETR
    # also writes checkpoint_best_*.pth, chosen on its validation set, which the protocol does not use
    ck = [out / "last_ema.pth"] if (out / "last_ema.pth").exists() else sorted(out.glob("*.pth"), key=lambda p: -p.stat().st_mtime)
    (out / "weights").mkdir(exist_ok=True)
    shutil.copy(ck[0], out / "weights" / "last.pt")
    (out / "cuas_rfdetr.json").write_text(json.dumps({"variant": a.variant, "resolution": a.resolution,
                                                      "names": names, "checkpoint": ck[0].name,
                                                      "n_train": n_train, "epochs": a.epochs, "train_seconds": secs}))
    print(f"E25: final weights {ck[0].name} -> {out / 'weights' / 'last.pt'}; {a.epochs} epoch(s) of {n_train} images "
          f"in {secs / 60:.1f} min", flush=True)
    if not a.keep_checkpoints:
        for f in list(out.glob("*.pth")) + list(out.glob("*.ckpt")) + list(out.glob("*.corrupt")):
            f.unlink()
        shutil.rmtree(ds, ignore_errors=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
