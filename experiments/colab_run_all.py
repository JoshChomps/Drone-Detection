# Colab, ONE cell: E14 (train + evaluate) and M2 detector runs over the Halmstad videos. Paste this whole
# file into a single cell of a Colab notebook with an L4 GPU (Runtime -> Change runtime type -> L4) and run it.
#
# Upload first, to MyDrive/cuas/ (build them locally with `python experiments/pack_for_colab.py`):
#   cuas_data_e14.zip   E14 training data, evaluation sets, E12 weights (~1.2 GB)
#   cuas_code.zip       cuas/ + experiments/
#
# Safe to re-run after a disconnect: finished stages are skipped (marker files on Drive) and training resumes
# from last.pt. Everything durable goes to MyDrive/cuas_runs/. Stages, roughly on an L4:
#   1 setup (~5 min)  2 smoke test (~3 min)  3 E14 training (~2-2.5 h)  4 E14 evaluation (~15 min)
#   5 M2: clone Halmstad + run E12 and E14 on all 285 videos (~15-25 min)
#   6 E15: E12 and E14 on DUT Anti-UAV test (2,200 images, never trained on; ~5 min)
# When it finishes, download MyDrive/cuas_runs/ (see the list printed at the end) and send back the summary.
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
RUN = "E14_hardneg_yolo26s_L4"
os.environ["CUAS_DATA"] = str(DATA)


def sh(cmd, log=None, show=r".", quiet=False):
    """Run a shell command, stream its output (split on \\r too, for progress bars), keep a full log on
    Drive, print only lines matching `show`. Stops the whole script if the command fails."""
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=REPO if REPO.exists() else None,
                         start_new_session=True)     # own process group, so a cell interrupt can kill all of it
    try:
        _stream(p, cmd, log, show, quiet)
    except KeyboardInterrupt:                       # Colab's stop button: don't leave training running headless
        try:
            os.killpg(p.pid, 9)
        except ProcessLookupError:
            pass
        raise


def _stream(p, cmd, log, show, quiet):
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
            if "it/s" in s or "s/it" in s:          # progress bar: print at most every 60 s
                if time.time() - last_progress > 60:
                    print("  " + s.strip()[-110:], flush=True)
                    last_progress = time.time()
            elif re.search(show, s):
                print("  " + s, flush=True)
    code = p.wait()
    if fh:
        fh.close()
    if code:
        print("  --- last output lines ---\n  " + "\n  ".join(tail), flush=True)
        raise SystemExit(f"FAILED ({code}): {cmd}" + (f"\nfull log: {log}" if log else ""))


def done(name):
    return (RUNS_DRIVE / f"DONE_{name}").exists()


def mark(name):
    (RUNS_DRIVE / f"DONE_{name}").write_text(time.ctime())


# 1. Setup ------------------------------------------------------------------------------------------------
from google.colab import drive  # noqa: E402
drive.mount("/content/drive")
RUNS_DRIVE.mkdir(parents=True, exist_ok=True)
for f in ("cuas_data_e14.zip", "cuas_code.zip"):
    if not (UP / f).exists():
        raise SystemExit(f"Upload {f} to MyDrive/cuas/ first")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
sh(f"{sys.executable} -m pip -q install ultralytics==8.4.162", quiet=True)
REPO.mkdir(exist_ok=True)
sh(f"unzip -q -o {UP / 'cuas_code.zip'} -d {REPO}")
if not (DATA / "e14" / "data_e14.yaml").exists():
    DATA.mkdir(exist_ok=True)
    sh(f"unzip -q -o {UP / 'cuas_data_e14.zip'} -d {DATA}")
if not (DATA / "runs").is_symlink():
    # E12's weights come in the data zip; move them to Drive, then make runs/ point at Drive.
    sh(f"mkdir -p {RUNS_DRIVE} && cp -rn {DATA}/runs/. {RUNS_DRIVE}/ && rm -rf {DATA}/runs && ln -s {RUNS_DRIVE} {DATA}/runs")
yaml = DATA / "e14" / "data_e14_colab.yaml"
yaml.write_text((DATA / "e14" / "data_e14.yaml").read_text().replace("C:/cuas_data", str(DATA)))
n_img = sum(1 for _ in DATA.rglob("*.jpg"))
print(f"setup ok: {n_img} images (expected 25,520: 15,480 train + 3,221 val + eval sets)")
train = f"{sys.executable} experiments/train_local.py"

# 2. Smoke test ---------------------------------------------------------------------------------------------
if not done("smoke"):
    sh(f"{train} --data {yaml} --name E14_colab_smoke --epochs 1 --fraction 0.05",
       log=RUNS_DRIVE / "E14_colab_smoke.log", show=r"^python |^ +all ")
    mark("smoke")

# 3. E14 training (resumes after a disconnect) ------------------------------------------------------------
if not done("train"):
    last = RUNS_DRIVE / RUN / "weights" / "last.pt"
    cmd = f"{train} --resume {last}" if last.exists() else f"{train} --data {yaml} --name {RUN}"
    sh(cmd, log=RUNS_DRIVE / "E14_L4_train.log", show=r"^ +all |EarlyStopping|epochs completed|Results saved")
    mark("train")

# 4. E14 evaluation (same protocol as E12) --------------------------------------------------------------------
if not done("eval"):
    sh(f"{sys.executable} experiments/e12_eval.py {RUN}", log=RUNS_DRIVE / f"eval_{RUN}.txt", show=r".")
    mark("eval")

# 5. M2: the real detector over all Halmstad videos, for tracking on detector output --------------------------
if not done("m2"):
    hal = Path("/content/halmstad")
    if not hal.exists():
        sh(f"git clone -q --depth 1 https://github.com/DroneDetectionThesis/Drone-detection-dataset {hal}")
    for tag, w in (("E12", RUNS_DRIVE / "E12_multidata_yolo26s" / "weights" / "best.pt"),
                   ("E14L4", RUNS_DRIVE / RUN / "weights" / "best.pt")):
        if (RUNS_DRIVE / "m2" / f"dets_{tag}.json.gz").exists():       # finished in an earlier session
            continue
        sh(f"{sys.executable} experiments/m2_detect_videos.py --weights {w} --src {hal}/Data/Video_V "
           f"--out {RUNS_DRIVE}/m2/dets_{tag}.json.gz", log=RUNS_DRIVE / f"m2_{tag}.log", show=r"videos")
    mark("m2")

# 6. E15: E12 and E14 on DUT Anti-UAV test, a dataset neither ever saw (each at its own val-chosen threshold) --
if not done("e15"):
    sh(f"{sys.executable} experiments/e15_dut.py E12_multidata_yolo26s {RUN}", log=RUNS_DRIVE / "e15_dut.txt",
       show=r".")
    mark("e15")

print("\nALL DONE. Download from MyDrive/cuas_runs/ into C:\\cuas_data\\runs\\ (same names):")
print(f"  {RUN}/  (at least weights/best.pt, e12_eval.json, dets_native.json, results.csv, args.yaml)")
print(f"  eval_{RUN}.txt, e15_dut.txt, e15_dut.json, E14_L4_train.log, E14_colab_smoke.log")
print("  m2/dets_E12.json.gz and m2/dets_E14L4.json.gz  ->  C:\\cuas_data\\m2\\")
print("Send back: this cell's output (smoke line, final 'all' rows, the evaluation printout).")
