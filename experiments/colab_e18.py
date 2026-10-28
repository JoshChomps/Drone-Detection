# Colab, ONE cell: the training queue. It began as E13 + E18 (architecture add-ons; both finished and closed,
# 16 #60-#61) and now runs the seeds of the plain recipe and E22 (training on degraded images).
# From 2026-10-06 the new runs are E24b/c/d (freeze variants of the provisional base E24; ~1.7 h each, 9 runs).
# From 2026-10-08 (16 #83-#87) also E23 / E23ctl (temporal input, 6 runs), E25 (RF-DETR Small, 3 runs) and, last,
# E39 (E24 + DUT train split, 3 runs), and E24 at 80 epochs (3 runs, the candidate final model). Seed 1 of each arm
# comes before the other seeds. Runtime is not capped (user, 2026-10-08: accuracy first): ~24 runs, of the order of
# 70-100 L4-hours with scoring; RF-DETR's share is unknown until its first run. Every stage is marked on Drive and
# every training resumes, so just re-run the cell after each disconnect. Send the output whenever a run finishes.
#
# Upload first, to MyDrive/cuas/:
#   cuas_code.zip       REPLACE the old one (build: python experiments/pack_for_colab.py --code)
#   cuas_data_e14.zip   already there from E14 (unchanged)
#   cuas_data_e22c.zip  already there from E22c (unchanged)
#   cuas_data_e23.zip   NEW, 1.49 GB (E23); a run whose data is missing is skipped with a note
#   cuas_data_e39.zip   NEW, 0.43 GB (E39)
# Runtime: L4 GPU. Paste this whole file into one cell and run it. Safe to re-run after a disconnect: finished
# stages are skipped (DONE_E18_* markers on Drive) and an interrupted training resumes from its last.pt.
#
# Every run: train 40 epochs with no early stopping -> its FINAL weights go through e12_eval.py -> paired comparison
# at matched recall with the plain run of the SAME seed -> E15 on DUT -> E21 (degraded images, incl. held-out kinds).
# When all members of a soup (SOUPS) are trained, their weights are averaged and the average is scored the same way.
# Roughly on an L4: a training run ~1.7-2 h, its scoring ~35 min. The whole list is ~25 GPU-hours; it is in
# priority order, so stop whenever you like and send back this cell's output. Logs are in MyDrive/cuas_runs/.
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive")
UP, RUNS_DRIVE = DRIVE / "cuas", DRIVE / "cuas_runs"
REPO, DATA = Path("/content/repo"), Path("/content/cuas_data")
BASE = "E14_hardneg_yolo26s_L4"         # seed 0 of the plain model, from colab_run_all.py
BASE_LAST = "E18_base_s1@last"          # base for FINAL-weights comparisons: a plain run that went the full 40 epochs
                                        # (E14 stopped early at 28, so its last.pt is not an end-of-schedule model)
DEG = "--degrade --patience 100"          # E22a: blur / noise / JPEG on the training images (cuas/degrade.py)
AUG = DEG + " --mosaic 0.1 --scale 0.3"   # E22b: plus the 2025 Drone-vs-Bird winner's small-target augmentation
RUNS = [                                # (name, add-ons, seed, extra train args), priority order; edit freely
    ("E18_base_s1", "", 1, ""),                   # plain recipe, seed 1: trained (16 #55); its final weights get scored
    ("E18_base_s2", "", 2, "--patience 100"),     # plain, seed 2
    ("E22a_s1", "", 1, DEG),                      # degraded training, same seeds as the plain runs
    ("E22a_s2", "", 2, DEG),
    ("E18_base_s0", "", 0, "--patience 100"),     # plain, seed 0 again: E14 stopped early at epoch 28
    ("E22a_s0", "", 0, DEG),
    ("E22b_s1", "", 1, AUG),
    ("E18_base_s1_e80", "", 1, "--epochs 80 --patience 100"),   # is 40 epochs enough? pairs with E18_base_s1
    # E22c: E22a plus 3,000 tiles with a real drone pasted in from the other dataset (build_e22c.py). Needs
    # cuas_data_e22c.zip on Drive; skipped with a note if it is not there.
    ("E22c_s1", "", 1, DEG + " --data {E22C}"),
    ("E22c_s2", "", 2, DEG + " --data {E22C}"),
    ("E22c_s0", "", 0, DEG + " --data {E22C}"),
    # E22b showed no gain over E22a in its first seed (16 #67-#68), so its other seeds are last. Delete to skip.
    ("E22b_s2", "", 2, AUG),
    ("E22b_s0", "", 0, AUG),
    # E22c is the base recipe from 2026-10-05 (16 #75). Next, two ways to keep more of the pretraining, aimed at the
    # unseen dataset (DUT), each compared with E22c of the same seed:
    # E27: start from the official Objects365-pretrained weights (365 classes, in ultralytics/assets v8.4.0) instead
    # of the COCO ones. E24: freeze the first 10 layers (the backbone) so fine-tuning cannot distort them.
    ("E27_s1", "", 1, DEG + " --data {E22C} --model yolo26s-objv1-150.pt"),
    ("E27_s2", "", 2, DEG + " --data {E22C} --model yolo26s-objv1-150.pt"),
    ("E27_s0", "", 0, DEG + " --data {E22C} --model yolo26s-objv1-150.pt"),
    ("E24_s1", "", 1, DEG + " --data {E22C} --freeze 10"),
    ("E24_s2", "", 2, DEG + " --data {E22C} --freeze 10"),
    ("E24_s0", "", 0, DEG + " --data {E22C} --freeze 10"),
    # E24 is the base recipe from 2026-10-06, provisionally (16 #81): DUT AP 0.195 -> 0.311, daytime Anti-UAV recall at
    # 10 false drones per 1000 frames 0.895 -> 0.871. Three ways to keep the transfer and win back the in-domain score,
    # each compared with E24 of the same seed. Seed 1 of all three first, so stopping early still answers something.
    # E24b: freeze half the backbone. E24c: freeze 10 for 30 epochs, then train everything (LP-FT, arXiv 2202.10054).
    # E24d: the plain recipe (no degradation, no pasted drones) with freeze 10: do E22a/E22c still help once frozen?
    ("E24b_s1", "", 1, DEG + " --data {E22C} --freeze 5"),
    ("E24c_s1", "", 1, DEG + " --data {E22C} --freeze 10 --unfreeze-at 30"),
    ("E24d_s1", "", 1, "--patience 100 --freeze 10"),
    # E23 (19 section 5 rank 2): temporal input, three registered grey frames t-6, t-3, t as the channels
    # (build_e23.py, cuas/temporal.py), against E23ctl: the same images with frame t in all three channels. Plain
    # recipe on the E14 data (no degradation: its JPEG step mixes channels; no freeze: the backbone must learn motion),
    # so E23ctl against E18_base shows what dropping colour costs. Scored on the stacked sets only (no DUT: it has
    # no video). Needs cuas_data_e23.zip on Drive.
    ("E23ctl_s1", "", 1, "--patience 100 --temporal --data {E23S}"),
    ("E23_s1", "", 1, "--patience 100 --temporal --data {E23}"),
    # E25 (19 section 5 rank 5): RF-DETR Small (Apache-2.0, windowed DINOv2 backbone) at 640 on E22c's data, trained
    # by experiments/train_rfdetr.py and scored by the same scripts (cuas.detector.load). Its own augmentation, not
    # cuas.degrade, so it is a model-family comparison against E24 and E22c, not a one-change arm. RFDETR_EPOCHS below.
    ("E25_s1", "rfdetr", 1, "--data {E22C}"),
    # E24 at 80 epochs: the candidate FINAL model. On the plain recipe 80 epochs halved the in-domain false alarms at
    # the same recall and cut DUT false drones (16 #70, #75); runtime is no longer a reason to stop at 40 (user,
    # 2026-10-08). Compared with E24 of the same seed (40 epochs). Experiments between arms stay at 40.
    ("E24_s1_e80", "", 1, DEG + " --data {E22C} --freeze 10 --epochs 80"),
    ("E24b_s2", "", 2, DEG + " --data {E22C} --freeze 5"),
    ("E24c_s2", "", 2, DEG + " --data {E22C} --freeze 10 --unfreeze-at 30"),
    ("E24d_s2", "", 2, "--patience 100 --freeze 10"),
    ("E24b_s0", "", 0, DEG + " --data {E22C} --freeze 5"),
    ("E24c_s0", "", 0, DEG + " --data {E22C} --freeze 10 --unfreeze-at 30"),
    ("E24d_s0", "", 0, "--patience 100 --freeze 10"),
    ("E23ctl_s2", "", 2, "--patience 100 --temporal --data {E23S}"),
    ("E23_s2", "", 2, "--patience 100 --temporal --data {E23}"),
    ("E23ctl_s0", "", 0, "--patience 100 --temporal --data {E23S}"),
    ("E23_s0", "", 0, "--patience 100 --temporal --data {E23}"),
    ("E24_s2_e80", "", 2, DEG + " --data {E22C} --freeze 10 --epochs 80"),
    ("E24_s0_e80", "", 0, DEG + " --data {E22C} --freeze 10 --epochs 80"),
    ("E25_s2", "rfdetr", 2, "--data {E22C}"),
    ("E25_s0", "rfdetr", 0, "--data {E22C}"),
    # E39: the base recipe (E24) plus DUT's train split (build_e39.py). LAST on purpose: once a model has trained on
    # DUT, DUT test is no longer an unseen dataset, so every arm chosen on DUT test runs before it. Needs
    # cuas_data_e39.zip on Drive. Compared with E24 of the same seed.
    ("E39_s1", "", 1, DEG + " --data {E39} --freeze 10"),
    ("E39_s2", "", 2, DEG + " --data {E39} --freeze 10"),
    ("E39_s0", "", 0, DEG + " --data {E39} --freeze 10"),
]
RFDETR_EPOCHS = 20                      # E25: the full schedule whatever it takes (user, 2026-10-08: accuracy before
                                        # runtime); an interrupted run resumes from its last finished epoch (16 #88)
SOUPS = [                               # E31: the mean of several seeds' final weights as ONE model (soup.py)
    ("E31_soup_base_s1s2", ["E18_base_s1", "E18_base_s2"]),     # the test: can our seeds be averaged at all?
    # A local soup of two DIFFERENT recipes (E12 + E14) ran, but was worse than either member at matched recall
    # (16 #62), so the rest wait for the result above. Uncomment them if that soup holds up against its members:
    # ("E31_soup_base", ["E18_base_s1", "E18_base_s2", "E18_base_s0"]),
    # ("E31_soup_E22a", ["E22a_s1", "E22a_s2", "E22a_s0"]),
    # ("E31_soup_E22b", ["E22b_s1", "E22b_s2", "E22b_s0"]),
]
os.environ["CUAS_DATA"] = str(DATA)


def sh(cmd, log=None, show=r".", quiet=False, check=True):
    """Run a shell command, stream its output (split on \\r too, for progress bars), keep a full log on
    Drive, print only lines matching `show`. Stops the whole script if the command fails (unless check=False)."""
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=REPO if REPO.exists() else None,
                         start_new_session=True)     # own process group, so a cell interrupt can kill all of it
    try:
        return _stream(p, cmd, log, show, quiet, check)
    except KeyboardInterrupt:                       # Colab's stop button: don't leave training running headless
        try:
            os.killpg(p.pid, 9)
        except ProcessLookupError:                  # it had already exited
            pass
        print("  interrupted: child processes killed. Re-run the cell to continue.", flush=True)
        raise


def _stream(p, cmd, log, show, quiet, check):
    fh = open(log, "a") if log else None
    buf, last_progress, tail = b"", 0.0, deque(maxlen=20)
    for chunk in iter(lambda: p.stdout.read(4096), b""):
        buf += chunk
        *lines, buf = re.split(rb"[\r\n]", buf)
        for ln in lines:
            s = ln.decode(errors="replace")
            if fh:
                fh.write(s + "\n")
            if s.strip() and "it/s" not in s and "s/it" not in s:
                tail.append(s)                      # shown if the command fails, whatever `show` filters
            if quiet or not s.strip():
                continue
            if "it/s" in s or "s/it" in s:          # progress bar: print at most every 5 min
                if time.time() - last_progress > 300:
                    print("  " + s.strip()[-110:], flush=True)
                    last_progress = time.time()
            elif re.search(show, s):
                print("  " + s, flush=True)
    code = p.wait()
    if fh:
        fh.close()
    if code and check:
        print("  --- last output lines ---\n  " + "\n  ".join(tail), flush=True)
        raise SystemExit(f"FAILED ({code}): {cmd}" + (f"\nfull log: {log}" if log else ""))
    return code


def done(name):
    return (RUNS_DRIVE / f"DONE_E18_{name}").exists()


def mark(name):
    (RUNS_DRIVE / f"DONE_E18_{name}").write_text(time.ctime())


def evaluate(spec, base, py=sys.executable):
    """e12_eval -> paired comparison with `base` at matched recall (+ sweep) -> E15 on DUT, for one set of weights:
    "run" is its best.pt, "run@last" its final weights (16 #57). Every step runs once (markers on Drive).
    py: the interpreter (E25 runs use the RF-DETR environment)."""
    key = spec.replace("@", "_")
    if not done(f"eval_{key}"):
        sh(f"{py} experiments/e12_eval.py {spec}", log=RUNS_DRIVE / f"eval_{key}.txt")
        mark(f"eval_{key}")
    b_name, _, b_which = (base or "").partition("@")
    b_dets = RUNS_DRIVE / b_name / ("dets_native_last.json" if b_which == "last" else "dets_native.json")
    if base and base != spec and b_dets.exists() and not done(f"cmp_{key}"):
        # --match: the run at the base's native val recall (a model comparison, not a threshold comparison; 16 #50)
        sh(f"{py} experiments/compare_runs.py --match {base} {spec}", log=RUNS_DRIVE / f"cmp_{key}.txt")
        sh(f"{py} experiments/compare_runs.py --sweep {base} {spec}", log=RUNS_DRIVE / f"cmp_{key}.txt")
        mark(f"cmp_{key}")
    # E15 right away (DUT, never trained on; with the sky/clutter split), so each run's answer arrives with it
    if not done(f"e15_{key}"):
        sh(f"{py} experiments/e15_dut.py --out e15_{key}.json {spec}", log=RUNS_DRIVE / f"e15_{key}.txt")
        mark(f"e15_{key}")
    # E21: recall under blur, noise and compression, including kinds E22 never trains on (writes e21_<key>.txt)
    if not done(f"e21_{key}"):
        sh(f"{py} experiments/e21_degradation.py {spec}", log=RUNS_DRIVE / f"e21_{key}_log.txt")
        mark(f"e21_{key}")
    # E35: the scores with no threshold (drone AP, recall at a false-alarm budget), box error, calibration, now on
    # DUT too (--dut caches <run>/dets_dut_last.json). These are the headline comparison (19 section 11)
    if not done(f"e35_{key}"):
        sh(f"{py} experiments/e35_metrics.py --dut {spec}", log=RUNS_DRIVE / f"e35_{key}_log.txt")
        mark(f"e35_{key}")


def evaluate_e23(spec, base, root):
    """E23 / E23ctl: the E12 evaluation, the paired comparison and E35 on the stacked copies of the same images
    (CUAS_DATA = the stack root, whose runs/ links to the real runs folder). No E15 (DUT has no video) and no E21
    (its JPEG step subsamples colour, which would smear the frame differences)."""
    key, env = spec.replace("@", "_"), f"CUAS_DATA={root}"
    if not done(f"eval_{key}"):
        sh(f"{env} {sys.executable} experiments/e12_eval.py {spec}", log=RUNS_DRIVE / f"eval_{key}.txt")
        mark(f"eval_{key}")
    if base and not done(f"cmp_{key}"):
        # The base's detections were made on its own input (frame t for E23ctl, colour for E18_base) of the same
        # images in the same order, so the comparison pairs image by image
        sh(f"{env} {sys.executable} experiments/compare_runs.py --match {base} {spec}", log=RUNS_DRIVE / f"cmp_{key}.txt")
        sh(f"{env} {sys.executable} experiments/compare_runs.py --sweep {base} {spec}", log=RUNS_DRIVE / f"cmp_{key}.txt")
        mark(f"cmp_{key}")
    if not done(f"e35_{key}"):
        sh(f"{env} {sys.executable} experiments/e35_metrics.py {spec}", log=RUNS_DRIVE / f"e35_{key}_log.txt")
        mark(f"e35_{key}")


def soups():
    """Average and score every soup whose members have all finished training."""
    for name, members in SOUPS:
        if done(f"soup_{name}") or not all(done(f"train_{m}") for m in members):
            continue
        sh(f"{sys.executable} experiments/soup.py {name} {' '.join(members)}", log=RUNS_DRIVE / f"{name}_soup.log")
        mark(f"soup_{name}")
    for name, members in SOUPS:
        if done(f"soup_{name}"):
            evaluate(f"{name}@last", BASE_LAST)


def checkpoint(name):
    """("done" | "resume" | "fresh", checkpoint path or None) for a run's training.

    done: last.pt belongs to a finished run. Ultralytics' final_eval strips it (epoch -1), and resuming a stripped
    checkpoint fails ("nothing to resume"), e.g. if the runtime died after training, before the marker.
    resume: from last.pt, or from best.pt (a full checkpoint of an earlier epoch) when last.pt cannot be read.
    Ultralytics writes last.pt in place, so a kill or a dead runtime during the save leaves it cut short; it is set
    aside as last.corrupt and never loaded again (16 #54)."""
    import torch
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))     # the checkpoint may pickle cuas.arch modules
    for f in (RUNS_DRIVE / name / "weights" / "last.pt", RUNS_DRIVE / name / "weights" / "best.pt"):
        if not f.exists():
            continue
        try:
            epoch = torch.load(f, map_location="cpu", weights_only=False).get("epoch", -1)
        except Exception as e:            # truncated or otherwise unreadable
            print(f"[{name}] {f.name} is unreadable ({type(e).__name__}); set aside as {f.stem}.corrupt", flush=True)
            f.replace(f.with_suffix(".corrupt"))
            continue
        return ("done" if epoch == -1 else "resume"), f
    return "fresh", None


# 1. Setup (same as colab_run_all.py; skips what this runtime already has) -----------------------------------
from google.colab import drive  # noqa: E402
drive.mount("/content/drive")
# A run stopped by an older version of this cell (or of colab_run_all.py) may still be training in the background
subprocess.run("pkill -9 -f 'experiments/(train_local|e12_eval|e15_dut|compare_runs)|unittest[ ]discover'; sleep 2",
               shell=True)
RUNS_DRIVE.mkdir(parents=True, exist_ok=True)
for f in ("cuas_data_e14.zip", "cuas_code.zip"):
    if not (UP / f).exists():
        raise SystemExit(f"Upload {f} to MyDrive/cuas/ first")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
sh(f"{sys.executable} -m pip -q install ultralytics==8.4.162 onnx onnxruntime onnxscript", quiet=True)
REPO.mkdir(exist_ok=True)
sh(f"unzip -q -o {UP / 'cuas_code.zip'} -d {REPO}")
if not (REPO / "cuas" / "arch.py").exists():
    raise SystemExit("cuas_code.zip on Drive is the old one: upload the new one (python experiments/pack_for_colab.py --code)")
if not (DATA / "e14" / "data_e14.yaml").exists():
    DATA.mkdir(exist_ok=True)
    sh(f"unzip -q -o {UP / 'cuas_data_e14.zip'} -d {DATA}")
if not (DATA / "runs").is_symlink():
    sh(f"mkdir -p {RUNS_DRIVE} && cp -rn {DATA}/runs/. {RUNS_DRIVE}/ && rm -rf {DATA}/runs && ln -s {RUNS_DRIVE} {DATA}/runs")
yaml = DATA / "e14" / "data_e14_colab.yaml"
yaml.write_text((DATA / "e14" / "data_e14.yaml").read_text().replace("C:/cuas_data", str(DATA)))
E22C = DATA / "e22c" / "data_e22c_colab.yaml"       # E22c's data: E14's plus the pasted-drone tiles
if (UP / "cuas_data_e22c.zip").exists():
    if not (DATA / "e22c" / "data_e22c.yaml").exists():
        sh(f"unzip -q -o {UP / 'cuas_data_e22c.zip'} -d {DATA}")
    E22C.write_text((DATA / "e22c" / "data_e22c.yaml").read_text().replace("C:/cuas_data", str(DATA)))
E23, E23S = DATA / "e23_k3", DATA / "e23_k0"       # E23: frame stacks; E23ctl: frame t in all three channels
if (UP / "cuas_data_e23.zip").exists():
    if not (E23 / "data_e23.yaml").exists():
        sh(f"unzip -q -o {UP / 'cuas_data_e23.zip'} -d {DATA}")
    if not (E23S / "data_e23.yaml").exists():      # derived here so only one set is uploaded (~10 min, CPU)
        sh(f"{sys.executable} experiments/build_e23.py --static {E23} {E23S}")
    for r in (E23, E23S):
        if not (r / "runs").exists():
            (r / "runs").symlink_to(RUNS_DRIVE)
    yml = (E23 / "data_e23.yaml").read_text()
    (E23 / "data_e23_colab.yaml").write_text(yml.replace("C:/cuas_data", str(DATA)))
    (E23S / "data_e23_colab.yaml").write_text(yml.replace("C:/cuas_data/e23_k3", str(E23S)))
E39 = DATA / "e39" / "data_e39_colab.yaml"         # E22c's data plus DUT train tiles
if (UP / "cuas_data_e39.zip").exists() and E22C.exists():
    if not (DATA / "e39" / "data_e39.yaml").exists():
        sh(f"unzip -q -o {UP / 'cuas_data_e39.zip'} -d {DATA}")
    E39.write_text((DATA / "e39" / "data_e39.yaml").read_text().replace("C:/cuas_data", str(DATA)))
have_base = (RUNS_DRIVE / BASE / "dets_native.json").exists()
print(f"setup ok | base {BASE} evaluated: {have_base}" + ("" if have_base else
      "  (run colab_run_all.py to the end first; paired comparisons are skipped until then)"))
# Data loading, not the GPU, is the likely limit for a model this small (and --degrade adds CPU work per image), so
# use up to 8 loader workers. The batch stays 16: it is part of the recipe the plain seed-1 run was trained with.
train = f"{sys.executable} experiments/train_local.py --data {yaml} --workers {min(8, os.cpu_count() or 4)}"
print(f"CPU cores: {os.cpu_count()}")

# 2. Checks: the add-ons behave on Colab's torch/CUDA, and a 1-epoch GPU run of the full model trains -------
if not done("checks"):
    sh(f"{sys.executable} -m unittest discover -s tests -p test_arch.py", log=RUNS_DRIVE / "E18_checks.log",
       show=r"^Ran |^OK|FAIL|Error")
    sh(f"{train} --name E18_smoke --arch aelan+mpda+rmpda --epochs 1 --fraction 0.05",
       log=RUNS_DRIVE / "E18_checks.log", show=r"^python |E18 add-ons|Transferred|^ +all ")
    mark("checks")

if not done("checks_e22"):                       # the E22 options train and resume on this runtime (~3 min)
    sh(f"{train} --name E22_smoke --seed 0 --epochs 2 --fraction 0.02 {AUG}", log=RUNS_DRIVE / "E22_checks.log",
       show=r"degraded|^ +all |epochs completed")
    mark("checks_e22")

if E22C.exists() and not done("checks_e22c"):    # the E22c data loads and trains (~3 min)
    sh(f"{train} --name E22c_smoke --seed 0 --epochs 2 --fraction 0.02 {DEG} --data {E22C}",
       log=RUNS_DRIVE / "E22_checks.log", show=r"degraded|images, |^ +all |epochs completed")
    mark("checks_e22c")

O365 = REPO / "yolo26s-objv1-150.pt"              # E27's start; verified in ultralytics/assets v8.4.0 (2026-10-05)
if not O365.exists():
    sh(f"curl -sSL -o {O365} https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-objv1-150.pt")
if E22C.exists() and not done("checks_e27"):     # the Objects365 weights download and load (~3 min). Look for the
    sh(f"{train} --name E27_smoke --seed 0 --epochs 2 --fraction 0.02 {DEG} --data {E22C} --model yolo26s-objv1-150.pt",
       log=RUNS_DRIVE / "E22_checks.log", show=r"Remapped|Transferred|Download|^ +all |epochs completed")
    mark("checks_e27")                           # "Remapped k/4 cls head rows" line: which classes start from it
# E25: RF-DETR lives in its own environment on top of the system packages, so whatever its training extras pull in
# (numpy, a headless OpenCV; 16 #85) cannot break the YOLO runs. The environment is per runtime: rebuilt when missing.
PY_RF = Path("/content/rfdetr_env/bin/python")
if any(arch == "rfdetr" and not done(f"e35_{n}_last") for n, arch, *_ in RUNS) and E22C.exists():
    if not PY_RF.exists():
        sh(f"{sys.executable} -m pip -q install virtualenv", quiet=True)
        sh(f"{sys.executable} -m virtualenv -q --system-site-packages {PY_RF.parents[1]}", quiet=True)
        sh(f'{PY_RF} -m pip -q install "rfdetr[train]"', quiet=True)
        sh(f"{PY_RF} -c \"import torch, rfdetr, pytorch_lightning; print('rfdetr env ok, torch', torch.__version__, "
           f"'cuda', torch.cuda.is_available())\"")
    if not done("checks_e25"):                   # trains and scores a small sample; prints its timing
        t_rf = time.time()
        sh(f"{PY_RF} experiments/train_rfdetr.py --data {E22C} --name E25_smoke --seed 0 --epochs 1 "
           f"--fraction 0.05", log=RUNS_DRIVE / "E25_checks.log", show=r"E25|mAP|\d\.\d{4} |Error|error")
        t_rf = time.time()
        sh(f"{PY_RF} experiments/e12_eval.py E25_smoke@last", log=RUNS_DRIVE / "E25_checks.log",
           show=r"recall|conf|Error|error")
        print(f"E25 smoke: e12_eval took {(time.time() - t_rf) / 60:.1f} min", flush=True)
        mark("checks_e25")
if (E23 / "data_e23_colab.yaml").exists() and not done("checks_e23"):   # stacks load; expect "E23 temporal input"
    sh(f"{train} --name E23_smoke --seed 0 --epochs 2 --fraction 0.02 --temporal --data {E23 / 'data_e23_colab.yaml'}",
       log=RUNS_DRIVE / "E22_checks.log", show=r"E23|albumentations|^ +all |epochs completed")
    mark("checks_e23")
if E22C.exists() and not done("checks_e24c"):    # freeze then release (~3 min): expect "released 99 ..." and "changed ...: True"
    sh(f"{train} --name E24c_smoke --seed 0 --epochs 2 --fraction 0.02 {DEG} --data {E22C} --freeze 10 --unfreeze-at 1",
       log=RUNS_DRIVE / "E22_checks.log", show=r"E24c|^ +all |epochs completed")
    mark("checks_e24c")

# 3. Training runs, each followed by the E12/E14 evaluation protocol and a paired comparison with the base ----
if done("train_E18_base_s1"):
    evaluate(BASE_LAST, None)                    # the final-weights base, before anything is compared with it
for name, arch, seed, extra in RUNS:
    t0 = time.time()
    if "{E22C}" in extra:
        if not E22C.exists():
            print(f"[{name}] skipped: upload cuas_data_e22c.zip to MyDrive/cuas/ (python experiments/pack_for_colab.py "
                  "--e22c) and re-run this cell", flush=True)
            continue
        extra = extra.replace("{E22C}", str(E22C))
    if "{E39}" in extra:
        if not E39.exists():
            print(f"[{name}] skipped: upload cuas_data_e39.zip to MyDrive/cuas/ (python experiments/pack_for_colab.py "
                  "--e39) and re-run this cell", flush=True)
            continue
        extra = extra.replace("{E39}", str(E39))
    if "{E23" in extra:
        if not (E23 / "data_e23_colab.yaml").exists():
            print(f"[{name}] skipped: upload cuas_data_e23.zip to MyDrive/cuas/ (python experiments/pack_for_colab.py "
                  "--e23) and re-run this cell", flush=True)
            continue
        extra = extra.replace("{E23S}", str(E23S / "data_e23_colab.yaml")).replace("{E23}", str(E23 / "data_e23_colab.yaml"))
    state, ckpt = (("done", None) if done(f"train_{name}") else ("rfdetr", None) if arch == "rfdetr"
                   else checkpoint(name))
    if state == "done" and not done(f"train_{name}"):
        print(f"[{name}] training had already finished (stripped {ckpt.name}); going on to evaluation", flush=True)
        mark(f"train_{name}")
    if arch == "rfdetr" and not done(f"train_{name}"):     # E25: train_rfdetr.py resumes an interrupted run itself
        sh(f"{PY_RF} experiments/train_rfdetr.py --name {name} --seed {seed} --epochs {RFDETR_EPOCHS} {extra}",
           log=RUNS_DRIVE / f"{name}_train.log", show=r"E25|mAP|\d\.\d{4} |Error|error")
        mark(f"train_{name}")
        print(f"[{name}] trained in {(time.time() - t0) / 3600:.2f} h", flush=True)
    if not done(f"train_{name}"):
        cmd = (f"{sys.executable} experiments/train_local.py --resume {ckpt}" if state == "resume"
               else f"{train} --name {name} --seed {seed}" + (f" --arch {arch}" if arch else "") + f" {extra}")
        sh(cmd, log=RUNS_DRIVE / f"{name}_train.log",
           show=r"degraded|E18 add-ons|E23|E24c|Transferred|Resuming|^ +all |EarlyStopping|epochs completed")
        mark(f"train_{name}")
        print(f"[{name}] trained in {(time.time() - t0) / 3600:.2f} h", flush=True)
    # Final weights only (19 section 3.4), against the plain run of the same seed when it exists, else seed 1
    # Each arm against the recipe it changes one thing of: E22b/c change E22a; E24/E27 change E22c; E24b/c/d change E24
    parent = ("E22a" if name.startswith(("E22b", "E22c")) else "E24" if name.startswith(("E24b", "E24c", "E24d", "E39", "E25")) or name.endswith("_e80") and name.startswith("E24")
              else "E22c" if name.startswith(("E24", "E27")) else "E18_base")
    if name.startswith("E23"):                   # E23 against E23ctl (time); E23ctl against plain (colour)
        b = f"E23ctl_s{seed}" if name.startswith("E23_") else f"E18_base_s{seed}"
        evaluate_e23(f"{name}@last", f"{b}@last" if done(f"eval_{b}_last") else None,
                     E23 if name.startswith("E23_") else E23S)
    else:
        evaluate(f"{name}@last", f"{parent}_s{seed}@last" if done(f"eval_{parent}_s{seed}_last") else BASE_LAST,
                 py=PY_RF if arch == "rfdetr" else sys.executable)
    soups()
    print(f"[{name}] done. You can stop here and send the output so far; re-running continues with the next run.",
          flush=True)

# 4. E35 on DUT for every finished run: drone AP and recall at a false-alarm budget on the dataset never trained on,
# so recipes are compared there without a threshold (e15_dut.py scores one threshold per run). Caches each run's DUT
# detections as <run>/dets_dut_last.json (~2-3 min a run on an L4). Writes e35_metrics_dut.txt (16 #72).
DUT_RUNS = [f"{n}@last" for n, *_ in RUNS if done(f"train_{n}") and not n.startswith("E23")]
if DUT_RUNS and not done(f"e35_dut_all_{len(DUT_RUNS)}"):     # keyed by the number of runs: the table is rewritten
    sh(f"{sys.executable} experiments/e35_metrics.py --dut {' '.join(DUT_RUNS)}", log=RUNS_DRIVE / "e35_metrics_dut_log.txt",
       show=r"^E35|^R@|^run |dut_test|antiuav_test")            # when runs were added (the old marker froze it, 16 #90)
    mark(f"e35_dut_all_{len(DUT_RUNS)}")

print("\nALL DONE. Send back this cell's output. On Drive (MyDrive/cuas_runs/): the run folders and the loose")
print("eval_*_last.txt, cmp_*_last.txt, e15_*_last.txt/.json, e21_*_last.txt/.json files. Download them into")
print("C:/cuas_data/runs/ (same names) for the local write-up, then run: python tools/tidy_runs.py --apply")
