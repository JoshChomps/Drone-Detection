"""
Train a detector on the local GPU with the same settings as the Colab runs (E1/E2), so results stay comparable.
Windows needs the __main__ guard below for multi-worker data loading.

    C:\\cuas_venv\\Scripts\\python experiments/train_local.py --data C:/cuas_data/e12/data_e12.yaml --name E12_smoke --epochs 2 --fraction 0.1
    C:\\cuas_venv\\Scripts\\python experiments/train_local.py --data C:/cuas_data/e12/data_e12.yaml --name E12_multidata_yolo26s

Resume after an interruption:  --resume C:/cuas_data/runs/<name>/weights/last.pt
E18 architecture add-ons (cuas/arch.py):  --arch aelan+mpda+rmpda  (the spec is stored in the checkpoint, so
--resume needs no --arch). --seed changes the training seed (default 0, as in every earlier run).
The optimizer is PINNED to what `optimizer=auto` chose for every run so far (AdamW, lr 0.00125, momentum 0.9, no
bias warm-up). Left on auto, Ultralytics switches to MuSGD at lr 0.01 above 10,000 nominal iterations, and E14 is
at 9,680: a few hundred more training images would have changed the optimizer along with the data (16 #54).
E22:  --degrade  blurs / adds noise to / JPEG-compresses the training images (cuas/degrade.py; a marker file in the
run folder makes --resume do the same);  --mosaic 0.1 --scale 0.3  is the small-target augmentation arm (19 section 9).
E24c:  --freeze 10 --unfreeze-at 30  trains the head on a frozen backbone, then the whole network from epoch 30 on
(LP-FT, arXiv 2202.10054; the learning rate has decayed to ~0.26 of lr0 by then). Ultralytics' optimizer already holds
the frozen parameters (they just get no gradient), so releasing them needs no rebuild; a marker file makes --resume
do the same.
E23:  --temporal  for frame-stack input (cuas/temporal.py): no hue/saturation augmentation and no albumentations
(ToGray, CLAHE), which would mix the channels; brightness (hsv_v) scales all channels alike and stays. A marker file
makes --resume do the same.
Log every run in 16_training_run_log.md.
"""
import argparse
import os
import platform

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--name")
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", help="last.pt to resume from")
    ap.add_argument("--arch", default="", help="E18 add-ons, e.g. aelan+mpda+rmpda (cuas/arch.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optimizer", default="AdamW", help="'auto' restores the Ultralytics choice (see above)")
    ap.add_argument("--lr0", type=float, default=0.00125)
    ap.add_argument("--freeze", type=int, help="freeze the first N layers (e.g. 10 = the whole backbone)")
    ap.add_argument("--unfreeze-at", type=int, help="E24c: release the --freeze layers from this epoch on (0-based)")
    ap.add_argument("--device", default="0", help="'cpu' for a local smoke test")
    ap.add_argument("--temporal", action="store_true", help="E23: frame-stack input; no channel-mixing augmentation")
    ap.add_argument("--degrade", action="store_true", help="E22: degrade training images (cuas/degrade.py)")
    ap.add_argument("--mosaic", type=float, help="mosaic probability (Ultralytics default 1.0)")
    ap.add_argument("--scale", type=float, help="scale jitter, +/- this fraction (Ultralytics default 0.5)")
    ap.add_argument("--classes", help="train on these class ids only, e.g. 0 (E13: drone-only; other labelled objects become background)")
    a = ap.parse_args()
    if not a.resume and not (a.data and a.name):
        ap.error("--data and --name are required unless --resume is given")

    if platform.system() == "Windows":
        # Keep Windows from idle-sleeping while this process trains (like a video player does). Changes no
        # settings and ends with the process. A closed lid or choosing Sleep still sleeps.
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("Windows idle sleep blocked for the duration of training")

    from pathlib import Path
    runs_root = os.environ.get("CUAS_DATA", "C:/cuas_data") + "/runs"
    temporal_marker = (Path(a.resume).parents[1] if a.resume else Path(runs_root) / a.name) / "cuas_temporal.txt"
    if a.temporal or (a.resume and temporal_marker.exists()):
        import sys as _sys
        _sys.modules["albumentations"] = None      # Ultralytics then skips its albumentations transform
        temporal_marker.parent.mkdir(parents=True, exist_ok=True)
        temporal_marker.write_text("hsv_h=0 hsv_s=0, no albumentations")
        print("E23 temporal input: hue/saturation augmentation and albumentations off")
    import torch
    import ultralytics
    from ultralytics import YOLO
    print(f"python {platform.python_version()} | torch {torch.__version__} | ultralytics {ultralytics.__version__} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'}")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runs = os.environ.get("CUAS_DATA", "C:/cuas_data") + "/runs"     # CUAS_DATA overrides it (e.g. on Colab)
    marker = (Path(a.resume).parents[1] if a.resume else Path(runs) / a.name) / "cuas_degrade.txt"
    if a.degrade or (a.resume and marker.exists()):
        from cuas.degrade import install
        degrade = install()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(repr(degrade))
    unfreeze_marker = marker.with_name("cuas_unfreeze_at.txt")
    if a.unfreeze_at is not None:
        if a.freeze is None:
            ap.error("--unfreeze-at needs --freeze")
        unfreeze_marker.parent.mkdir(parents=True, exist_ok=True)
        unfreeze_marker.write_text(str(a.unfreeze_at))
    unfreeze_at = int(unfreeze_marker.read_text()) if unfreeze_marker.exists() else None

    def release_frozen(tr):
        """on_train_epoch_start, before the epoch's _model_train(): from `unfreeze_at` on, every layer trains and
        its BatchNorm statistics update again (on resume Ultralytics re-freezes first, so this re-releases)."""
        if unfreeze_at is None or tr.epoch < unfreeze_at or tr.freeze_layer_names == [".dfl"]:
            return
        tr.freeze_layer_names = [".dfl"]                 # what Ultralytics always freezes
        n = 0
        for k, v in tr.model.named_parameters():
            if ".dfl" not in k and v.dtype.is_floating_point and not v.requires_grad:
                v.requires_grad, n = True, n + 1
        print(f"E24c: released {n} frozen parameter tensors at epoch {tr.epoch} (0-based)", flush=True)
        probe = next(v for k, v in tr.model.named_parameters() if k.startswith("model.0."))
        release_frozen.probe = (probe, probe.detach().clone())

    def check_released(tr):
        """on_train_epoch_end, once: did the released backbone actually change during its first epoch?"""
        probe = getattr(release_frozen, "probe", None)
        if probe is not None:
            print(f"E24c: first backbone layer changed during epoch {tr.epoch}: "
                  f"{not torch.equal(probe[0].detach(), probe[1])}", flush=True)
            release_frozen.probe = None

    trainer = None
    if a.arch or a.resume:
        # ArchTrainer rebuilds add-ons named in the model yaml; without any it trains exactly like the default.
        from cuas.arch import ArchTrainer, parse_spec
        trainer = ArchTrainer
    if a.resume:
        model = YOLO(a.resume)
        model.add_callback("on_train_epoch_start", release_frozen)
        model.add_callback("on_train_epoch_end", check_released)
        model.train(resume=True, trainer=trainer)
    else:
        model = YOLO(a.model)
        model.add_callback("on_train_epoch_start", release_frozen)
        model.add_callback("on_train_epoch_end", check_released)
        if a.arch:
            model.model.yaml["cuas_arch"] = "+".join(parse_spec(a.arch))
            print(f"E18 add-ons: {model.model.yaml['cuas_arch']}")
        model.train(data=a.data, imgsz=640, epochs=a.epochs, batch=a.batch, seed=a.seed, patience=a.patience,
                    fraction=a.fraction, workers=a.workers, device=0 if a.device == "0" else a.device, project=runs,
                    name=a.name, exist_ok=True, trainer=trainer,
                    optimizer=a.optimizer, lr0=a.lr0, momentum=0.9, warmup_bias_lr=0.0,
                    **({"freeze": a.freeze} if a.freeze is not None else {}),
                    **({"hsv_h": 0.0, "hsv_s": 0.0} if a.temporal else {}),
                    **({"mosaic": a.mosaic} if a.mosaic is not None else {}),
                    **({"scale": a.scale} if a.scale is not None else {}),
                    **({"classes": [int(c) for c in a.classes.split(",")]} if a.classes else {}))
