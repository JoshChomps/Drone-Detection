# Training Run Log

A chronological record of **everything done to train and evaluate the detector**: every dataset build, training run, evaluation, crash, fix and re-run, on Google Colab and on the local laptop GPU. It records *what happened, in what order, with which settings*.

For *why* and *what it means*, see [15](15_detector_training_methods_and_findings.md) (methods, capabilities, limitations). For the experiment plan and the results table, see [14](14_build_log_and_experiments.md). **Add an entry here for every run** (template in §5).

---

## 1. Environments

| | Colab (E1, E2) | Local laptop (from E11) |
| :--- | :--- | :--- |
| GPU | NVIDIA Tesla T4, 15 GB | NVIDIA GeForce RTX 4060 Laptop, 8 GB (driver 616.56) |
| CPU for data loading | 2 dataloader workers (standard runtime) | Laptop CPU |
| Python / PyTorch | Python 3.13.15, torch 2.11.0 + CUDA 12.8 | Python 3.12, torch (cu130 wheels) in venv `C:\cuas_venv` |
| Ultralytics | 8.4.161 (smoke, E1); **8.4.162** (E2 resumed session) | Installed 2026-09-24 (version recorded in §2) |
| Code | `experiments/cuas_code.zip` uploaded per session (cell 2) | Repository `cuas/` directly |
| Data | `/content/` (wiped on disconnect) | `C:\cuas_data\` (outside the project folder, which is on D:) |
| Results | Google Drive `MyDrive/cuas_runs/` (survives disconnects) | `C:\cuas_data\runs\` |

---

## 2. Timeline

### 2026-09-23: Setup and data

**#1 Colab setup (cells 1-4).** T4 runtime. Installed `ultralytics` + `mcos-decoder`, uploaded the code zip, mounted Drive, cloned the Halmstad dataset (357 MB, CC0).

**#2 Sanity check (cell 5).**
* 285 visible videos, 89,233 frames, all 640×512.
* Box widths: drone median 33.8 px (10th percentile 23.0), bird 26.7, airplane 40.9, helicopter 72.4. **No boxes under 8 px.**
* The label-drawing check on V_DRONE_001 was correct.
* Found videos whose labels stop early (the first 10 listed were birds; V_BIRD_005: 1 label / 302 frames).

**#3 Code changes from #2.**
* The converter skips frames past a video's last label and takes no background frames from bird videos.
* New `cuas/zoomout.py` (simulated distance), because the dataset has no small drones.
* Code zip rebuilt; Colab session restarted to load it.

**#4 Dataset build (cell 6).** `python -m cuas.datasets.halmstad --every 10`:
* **6,829 train / 1,782 val tiles**; 12 videos had short labels (tails skipped).
* Zoom-out val mosaics k2/k3/k4/k6 = 445/198/111/49.

**#5 Smoke test (cell 8).** YOLO26s, 2 epochs, `fraction=0.1` (683 images), batch 16.
* 0.025 h. ~60 images/s training.
* Drone recall 0 at every size, which is expected: this only proves the pipeline end to end. Ultralytics' own validation agreed.
* Weights: `cuas_runs/smoke`.

### 2026-09-24: E1, E2 and diagnostics

**#6 E1 training (cell 9).** YOLO26s from COCO weights; 40 epochs max, patience 15, batch 16, imgsz 640, seed 0, default augmentation; `optimizer=auto` → **AdamW lr 0.00125, momentum 0.9**.
* 427 iterations/epoch, ~2 min 10 s + ~15 s validation per epoch; 5.4 GB GPU memory.
* Best at **epoch 12**, early-stopped at **27**; **1.12 h** total.
* Ultralytics val: drone P 0.85 / R 0.875 / mAP50 0.852; bird R 0.71.
* Weights: `cuas_runs/E1_halmstad_yolo26s/weights/best.pt`.

**#7 E1 evaluation (cell 10).** Native + zoom-out val, conf 0.1-0.55.
* **`w_min` 16 px.** Recall at 8-12 px 0.36, 4-8 px 0.03.
* Scores top out ~0.5; operating point conf 0.25.

**#8 E2 data (cell 11).** Zoom-out mosaics of the train split: 1,707/758/426/189 → **9,909 training images** via `data_e2.yaml`.

**#9 E2 training (cell 12).** Same settings as E1 plus the mosaics. 620 iterations/epoch, ~3 min/epoch; AdamW 0.00125 again (≈ 6,200 iterations, below the 10,000 MuSGD threshold).
* **The Colab session disconnected partway.** Resumed with `YOLO(.../E2_zoomtrain_yolo26s/weights/last.pt).train(resume=True)` after re-running cells 1-4, 6, 7, 11. It continued from **epoch 24**; the resumed session had Ultralytics 8.4.162.
* Ran to epoch 40, no early stop (0.96 h for the resumed part).
* Ultralytics val: drone P 0.863 / R 0.875 / mAP50 0.873; bird R 0.763; airplane R 0.713.

**#10 E2 evaluation (cell 13).**
* **`w_min` 8 px at conf 0.4** (4 px at 0.25): recall at 4-8 px 0.80, 8-12 px 0.95.
* Operating point conf 0.4.

**#11 Evaluation code iterations.**
* Added a background-false-alarms-by-width column.
* **Colab quirk:** uploading a new code zip doesn't reload modules already imported. Fixed with an `importlib.reload(cuas.eval)` cell (checks the source first).
* Added the **"extra box on a real drone"** category: E1's 326 "background" false alarms split into 230 on-drone + 96 background.

**#12 Diagnostics.**
* Cell 14, per-set breakdown: E2's false alarms were a flat ~0.012 per source frame at every scale, so the mosaic seams weren't the cause.
* Cell 15, pictures: first run failed (`cv2` not imported after a restart; the cell was made self-contained).
* **Finding: 12/12 native and 11/12 k3 "false alarms" were real, unlabelled drones.** Halmstad's "no label" ≠ "no drone".

**#13 Converter fix and rebuild.**
* `neg_every=0`: no unlabelled frames used as background.
* After a session reset, re-ran cells 4 and 6 with the fixed code. Val became 1,759 native + 439/195/109/48 mosaics = **2,550 images**.
* The existing E1/E2 models were **not** retrained (their 135 contaminated training "background" images remain a known flaw, to be fixed in the next training run).

**#14 Clean re-evaluation.** E1 @ 0.25 and E2 @ 0.4 on the rebuilt val:
* **E2: recall 0.89, 4 background false drones, ~99.5% of drone calls correct.**
* E1: recall 0.52, 31 background false drones.
* Cell 16 (empty-image check) was superseded and not run.

**#15 Anti-UAV300 acquisition (for E11).**
* The `gdown` download failed on Google Drive's "too many users" quota.
* Downloaded in the browser instead ("can't scan for viruses" prompt, normal for large files): `Anti-UAV-RGBT.zip`, 6.04 GB.
* Inspected locally: 318 sequences (train 160 / val 67 / test 91), visible video 1920×1080 at 20 fps.
* **Drones are large** (median width 109-122 px; 43 of ~280k boxes under 16 px). A tracking reticle is burned into the RGB video.
* E11 redefined: E11a (new camera/site at normal size) and E11b (simulated distance on unseen footage).

**#16 Local route set up (laptop GPU).**
* Decided to run E11 locally.
* Code added: tiled evaluation for frames larger than 640 px (`run_split(..., tile=640)`, so Full HD is never downscaled) and `cuas/datasets/antiuav.py`.
* Extracted the val/test visible files to `C:\cuas_data\antiuav_raw\`, built evaluation frames (every 40th frame with a drone + every 200th target-absent frame) into `C:\cuas_data\antiuav_val|test\`.
* Created venv `C:\cuas_venv` (CUDA PyTorch + Ultralytics). The E1/E2 `best.pt` files are copied from Drive to `C:\cuas_data\runs\`.

**#17 Local GPU environment verified.** `C:\cuas_venv`: **torch 2.14.0+cu130, CUDA available, NVIDIA GeForce RTX 4060 Laptop GPU, Ultralytics 8.4.162** (3.6 GB). E1/E2 `best.pt` copied from Drive into `C:\cuas_data\runs\<run>\weights\` (the browser named the second one `best (1).pt`; renamed; checksums differ).

**#18 Anti-UAV evaluation frames built (local, CPU).** `python -m cuas.datasets.antiuav --every 40 --absent_every 200`:
* **val: 67 sequences → 1,462 frames with a drone + 19 target-absent frames** (150 MB).
* test: built the same way.

**#19 E11 smoke test (local GPU, 20 val frames, E2 @ 0.4).**
* **0 of 20 drones found.** Every frame had 6-9 confident boxes, at the same positions frame after frame.
* The drawn frame showed why: **the Anti-UAV RGB video has a burned-in on-screen display** (Chinese status text "前端 可见光", azimuth "方位", elevation "俯仰", date/time, a tracking reticle, bottom-left text). The model labels the text **"airplane" at 0.54-0.81**.
* The real drone in that frame (bright, washed-out white quad on flat grey overcast sky) got **no drone prediction even at conf 0.05**. That could be a genuine domain gap: Halmstad drones are mostly dark silhouettes.
* Speed: 31 ms per 640 tile (251 ms per Full HD frame) at batch 8, FP32, Windows. Slower than expected for an RTX 4060; to investigate under E3.

**#20 Overlay removal.**
* The overlay sits at identical positions in all 297 sampled frames.
* `cuas/datasets/antiuav.py`: `overlay_mask` (a pixel is overlay if it shows a strong edge in ≥ 30% of frames) + `inpaint_split` (Telea inpainting in place).
* A first version (top-1% average-edge threshold) was **rejected**: its unit test showed it masks moving-drone trails when the overlay is small.
* On val the mask covers 2.7% of the frame and touches 153 of 1,462 labelled drones (10%, mostly drones crossing reticle lines).
* Justification: a deployed camera would have its OSD off, and the motion stage ignores static text.

**#21 Anti-UAV eval sets finalised.**
* test: **91 sequences → 2,009 frames with a drone + 29 target-absent frames** (216 MB).
* The overlay was inpainted from val and test with the **same mask** (`C:\cuas_data\antiuav_overlay_mask.png`, built from val, 2.73% of the frame).
* Mask touched: val 153/1,462 drones (10.5%), test 213/2,009 (10.6%).
* Extracted source video kept in `C:\cuas_data\antiuav_raw\` (2.6 GB; deletable). C: had 13 GB free.

**#22 E11 first pass, val (local GPU).** `experiments/e11_antiuav.py val`; E1 @ 0.25, E2 @ 0.4; native + zoom-out k4/k8/k12/k16 (92/23/10/5 mosaics); tiled at 640.
* **Drone recall was near zero everywhere:** E1 native 0.6%, E2 native 1.2%; shrunken ≤ 3%.
* Background false drones: E1 264, E2 225 (over 1,611 images). 370 s (E1) and 315 s (E2).

**#23 Is it a pipeline bug? Class-agnostic check.**
* Any class, IoU ≥ 0.3, score ≥ 0.05, native val (1,462 drones).
* **E1 put a correctly placed box on 1,098 (75%)**, but labelled them **airplane 773, helicopter 322, drone 3** (median score 0.63).
* E2 boxed 486 (33%): airplane 247, helicopter 197, drone 42.
* The boxes land on the drones, so tiling, labels and coordinates work. **The failure is classification.**
* Plausible cause: bird/airplane heads were inherited from COCO (broad "flying object" priors), while "drone" was learned only from Halmstad's drones (mostly dark silhouettes). Anti-UAV's bright white quads on grey sky fall under "airplane".

**#24 Local-vs-Colab reproduction check.**
* The user questioned whether the local pipeline was flawed. To settle it: cloned Halmstad locally (`C:\cuas_data\halmstad`, 636 MB) and rebuilt with the fixed converter. That gave **identical counts** to the Colab rebuild (6,694 train / 1,759 val; zoom 439/195/109/48).
* Re-evaluated E1 @ 0.25 and E2 @ 0.4 with `experiments/check_local_repro.py`: **every recall bin and every false-alarm count identical to Colab** (max difference 0.000).
* **The local pipeline is verified**; the Anti-UAV result is genuine. Rerun this script after any environment change.

**#25 Evaluation metric added: "missed drones boxed as another class"** (`missed_drone_boxed_as_other` in `cuas/eval.py`, plus a test). Counts at the operating threshold. Code zip rebuilt. Then re-ran E11 on **val and test** together so both come from identical code (#26).

**#26 E11 full run, val + test (local GPU).** `experiments/e11_antiuav.py val test` -> `C:\cuas_data\runs\E11\e11_summary_val_test.json`. Val reproduced #22 exactly. Timing: E1 189/207 s, E2 163/201 s (val/test).
* Test zoom-out: 127/31/14/7 mosaics.
* **Drone recall, test:** E1 native 0.0%, k4-k16 0.5/2.2/0.4/0.0%; E2 native 0.0%, k4-k16 2.8/1.1/3.8/3.2%. Val is similar (≤ 3%).
* **Missed drones boxed as another class, test:** E1 native 892 airplane + 427 helicopter of 2,009 (66%); E2 native 129 + 160 (14%).
* On shrunken test, E2 boxes 1,113/2,003 (k4), 1,145/1,956 (k8), 1,026/1,987 (k12), 825/1,766 (k16) as another class, mostly airplane. So **E2 still localises ~50-60% of unseen drones at 7-14 px but calls them airplane**. E1 at k8 boxes 695/1,956 (36%), at k16 2.
* Background false drones: E1 264 (val) / 370 (test), E2 225 / 205. Almost all on the zoom mosaics (native: 0-4). Not yet inspected.

### 2026-09-25: E12 (multi-dataset training, local)

**#27 Disk and data preparation.**
* With the user's approval, deleted the extracted val/test video `C:\cuas_data\antiuav_raw` (2.6 GB; the eval frames were already built; the zip is kept). C: then had 14 GB free.
* New code:
  * `cuas.tiling.tile_split`: positive tiles at true size, no background tiles, plus a test.
  * `experiments/train_local.py`: Colab-identical settings, Windows `__main__` guard, resume flag.
  * `experiments/build_e12.py`: the full data build.
* Tests: 22 pass.
* Build started: extract Anti-UAV **train** visible → frames (every 40th with a drone, no absent frames) → delete the extracted video → inpaint the overlay (same mask) → 640 tiles → zoom k2/4/8/16. Also Halmstad zoom train k2/3/4/6, Anti-UAV val tiles, and `C:\cuas_data\e12\data_e12.yaml`.

**#28 E12 data built.**
* Anti-UAV train: **160 sequences → 3,578 frames** with a drone (no absent frames). Overlay inpainted; the mask touched 431 drones (12%). → **3,578 positive tiles**; zoom k2/4/8/16 = 894/223/55/13 mosaics.
* Halmstad zoom train k2/3/4/6 = 1,673/743/418/185.
* **Train total: 14,476 images; instances: drone 31,111, airplane 7,508, helicopter 7,598, bird 4,774.**
* Ultralytics val: 3,221 images (Halmstad val 1,759 + Anti-UAV val tiles 1,462).
* Visual check: labels correct on a tile and a k16 mosaic (256 drones, 4-15 px). Two things Halmstad lacks:
  * Anti-UAV includes **drones in front of buildings**.
  * It includes **night sequences** (drone lights as bright dots on black). Night is out of the vision-only scope but kept in training. Watch for false alarms on lights.
* C: 9.3 GB free after the build.

**#29 E12 smoke test (local).** `train_local.py --epochs 2 --fraction 0.1 --name E12_smoke`, log `C:\cuas_data\runs\E12_smoke\reports\E12_smoke.log`.
* python 3.12.0, torch 2.14.0+cu130, Ultralytics 8.4.162, RTX 4060 Laptop; 4 dataloader workers; AdamW lr 0.00125.
* 2 epochs in **0.052 h**; 91 iterations/epoch at **3.1 it/s** (Colab T4 did 3.2 it/s); 4.7-5.1 GB GPU memory.
* After 2 epochs: drone P 0.70 / R 0.52 / mAP50 0.55 on the mixed val. Other classes 0, because the 10% subset came from the first training folder (Anti-UAV tiles, drones only).
* A harmless Ultralytics "slow image access" warning on val (probably Windows Defender scanning files).
* Estimate for the full run: 905 iterations/epoch → ~5.5 min/epoch → ≤ ~3.7 h for 40 epochs.

**#30 E12 full run started** (local): `train_local.py --data C:/cuas_data/e12/data_e12.yaml --name E12_multidata_yolo26s`, log `C:\cuas_data\runs\E12_multidata_yolo26s\reports\E12_multidata_yolo26s.log`, weights `C:\cuas_data\runs\E12_multidata_yolo26s\weights\`.
* Same schedule as E1/E2: 40 epochs max, patience 15, batch 16, imgsz 640, seed 0, AdamW auto.
* **Sleep protection:** the script calls Windows `SetThreadExecutionState` (blocks idle sleep while it runs, no settings changed) and the desktop app holds keep-awake for the session.
* A closed lid or manual Sleep would still stop it. If interrupted, resume with `--resume C:/cuas_data/runs/E12_multidata_yolo26s/weights/last.pt`.
* Epoch time ~5 min 5 s training + ~30-40 s validation.
* Val (Halmstad + Anti-UAV tiles) P / R / mAP50 / mAP50-95 per epoch:
  * 1: 0.53 / 0.67 / 0.68 / 0.31
  * 2: 0.81 / 0.49 / 0.65 / 0.30
  * 3: 0.78 / 0.78 / 0.80 / 0.39
  * 4: 0.80 / 0.74 / 0.82 / 0.41
  * 5: 0.73 / 0.77 / 0.84 / 0.43
  * 6: 0.85 / 0.84 / 0.89 / 0.45
  * **7: 0.89 / 0.82 / 0.89 / 0.46 (best so far)**
  * 8: 0.82 / 0.78 / 0.85 / 0.40
  * 9: 0.85 / 0.80 / 0.85 / 0.43
  * 10: 0.87 / 0.83 / 0.87 / 0.44

**#31 E12 paused after epoch 10 (2026-09-25 ~11:45)** because the user was moving the laptop. The process was stopped deliberately rather than letting the laptop sleep mid-epoch (GPU processes don't always survive sleep). `last.pt` was saved at the end of epoch 10 (11:44) and `best.pt` = epoch 7; the partial epoch 11 is lost (< 5 min). The GPU was confirmed free afterwards. Resume with `train_local.py --data C:/cuas_data/e12/data_e12.yaml --name E12_multidata_yolo26s --resume C:/cuas_data/runs/E12_multidata_yolo26s/weights/last.pt`.

**#32 E12 resumed (2026-09-25).**
* Ultralytics: "Resuming training … from epoch 11 to 40 total epochs". Same data (14,476 / 3,221), 3.4 it/s, sleep blocked again.
* Log `C:\cuas_data\runs\E12_multidata_yolo26s\reports\E12_multidata_yolo26s_resume1.log`. The resume keeps the original 40-epoch learning-rate schedule.
* Val P / R / mAP50 / mAP50-95 for epochs 11-36:
  * 11: 0.82 / 0.76 / 0.84 / 0.42
  * 12: 0.88 / 0.83 / 0.88 / 0.42
  * 13: 0.83 / 0.80 / 0.86 / 0.44
  * 14: 0.82 / 0.80 / 0.86 / 0.44
  * 15: 0.87 / 0.83 / 0.88 / 0.45
  * 16: 0.88 / 0.83 / 0.87 / 0.45
  * 17: 0.89 / 0.87 / 0.90 / 0.457
  * 18: 0.86 / 0.82 / 0.85 / 0.44
  * 19: 0.86 / 0.83 / 0.87 / 0.45
  * 20: 0.86 / 0.83 / 0.86 / 0.45
  * **21: 0.86 / 0.83 / 0.87 / 0.463 (best)**
  * 22: 0.87 / 0.84 / 0.88 / 0.46
  * 23: 0.87 / 0.81 / 0.87 / 0.46
  * 24: 0.86 / 0.83 / 0.88 / 0.45
  * 25: 0.82 / 0.81 / 0.83 / 0.42
  * 26: 0.84 / 0.83 / 0.85 / 0.44
  * 27: 0.85 / 0.84 / 0.86 / 0.45
  * 28: 0.85 / 0.83 / 0.87 / 0.45
  * 29: 0.84 / 0.83 / 0.85 / 0.44
  * 30: 0.85 / 0.84 / 0.85 / 0.44
  * 31 (mosaic off): 0.87 / 0.85 / 0.86 / 0.445
  * 32: 0.86 / 0.84 / 0.85 / 0.45
  * 33: 0.86 / 0.84 / 0.85 / 0.44
  * 34: 0.85 / 0.84 / 0.84 / 0.44
  * 35: 0.84 / 0.83 / 0.84 / 0.436
  * 36: 0.85 / 0.84 / 0.84 / 0.44
* **Early stopping after epoch 36** (15 epochs without beating epoch 21); the resumed part took 3.21 h; the process exited cleanly.
* **Plateau from ~epoch 6** (mAP50-95 0.42-0.46) and a slight decline after 30: no evidence that a longer schedule would help.
* **Final Ultralytics val of best.pt (epoch 21), mixed Halmstad + Anti-UAV val tiles** (optimistic, since the same set selected best.pt):
  * all: P 0.856 / R 0.832 / mAP50 0.872 / mAP50-95 0.463
  * **drone (2,165): 0.958 / 0.939 / 0.973 / 0.594**
  * bird (293): 0.827 / 0.799 / 0.842 / 0.447
  * **airplane (383): 0.744 / 0.675 / 0.771 / 0.414**
  * helicopter (380): 0.896 / 0.913 / 0.903 / 0.400
* Weights: `C:\cuas_data\runs\E12_multidata_yolo26s\weights\best.pt`.
* The user asked why training stops at 40 and about overtraining risk. The rationale is recorded in [15 §4](15_detector_training_methods_and_findings.md#4-model-and-training-configuration): 40 is a comparability ceiling; best.pt is chosen on val; the held-out test gives the final numbers; a longer schedule only if the best epoch lands at the very end.

**#33 E12 evaluation, with E2 on the same protocol (2026-09-25, local, ~15 min for both).**
* Command: `experiments/e12_eval.py <run>` for E12_multidata_yolo26s and E2_zoomtrain_yolo26s. Outputs in `C:\cuas_data\runs\eval_<run>.txt` and `C:\cuas_data\runs\<run>\e12_eval.json`.
* Protocol: detections are computed once at conf 0.05, with Full HD frames tiled at 640. The operating threshold is the F1 maximum on **Anti-UAV val** (native + k4/8/12/16 pooled). At that threshold the script scores Anti-UAV val, the held-out **Anti-UAV test (scored once)** and Halmstad val (native + k2/3/4/6).
* E12 threshold sweep on val (recall / precision / F1):
  * 0.10: 0.976 / 0.654 / 0.783
  * 0.25: 0.973 / 0.721 / 0.828
  * **0.40: 0.964 / 0.762 / 0.851 (chosen)**
  * 0.55: 0.889 / 0.809 / 0.847
  * 0.70: 0.512 / 0.905 / 0.654
* E2's F1 maximum was at 0.10 (F1 0.078).
* **Anti-UAV test, E12 @ 0.4:**
  * Recall 0.867, precision 0.779, `w_min` 8 px.
  * Recall by width: 0-4 0.61, 4-8 0.79, 8-12 0.86, 12-16 0.91, 16-24 0.90, 24-32 0.88, 32-64 0.95, 64+ 0.89.
  * Boxed as another class: 13 (was 5,064 for E2).
  * False drones: 819 on real drones, 0 on distractors, **1,568 background over 2,217 images**.
  * Per subset recall: native 0.899, k4 0.896, k8 0.857, k12 0.854, k16 0.826.
* E2 @ 0.1 on the same test: recall 0.049 (native 0.000); airplane 4,335, helicopter 662 and bird 67 of the boxed drones went to the wrong class.
* **Halmstad val, E12 @ 0.4:**
  * Recall 0.888, precision 0.984, 19 background FAs over 2,550 images.
  * Recall by width: 4-8 0.83, 8-12 0.95, 12-16 0.98, 16-24 0.93, 24-32 0.94, 32-64 **0.79**, 64+ 0.97.
  * `w_min` reports 64 px only because the 32-64 bin sits at 0.79, just under 0.8. The native-frame recall is 0.848.
* E2 @ 0.1 on Halmstad: recall 0.950, 82 background FAs, 889 on-drone boxes. That's a different threshold; the same-threshold comparison is in #34.

**#34 E12 diagnostics (2026-09-25, local, `experiments/e12_diag.py`, ~15 min).** Contact sheets are in `C:\cuas_data\runs\e12_diag\`.
* **Halmstad val at the same thresholds** (pooled native + k2-k6; recall / on-drone / background / distractor):

  | conf | E2 | E12 |
  | :--- | :--- | :--- |
  | 0.10 | 0.950 / 889 / 82 / 23 | 0.960 / 1,176 / 163 / 19 |
  | 0.25 | 0.933 / 306 / 14 / 8 | 0.952 / 423 / 60 / 7 |
  | 0.40 | 0.887 / 5 / 4 / 6 | 0.888 / 29 / 19 / 4 |

  * On **native** Halmstad frames at 0.4, E2 has recall 0.933 and E12 0.848, with 0 background FAs for both. That is a mild in-domain regression on normal-size dark drones.
  * The Halmstad background FAs are tiny sky specks (5-16 px) on mosaics.
* **Anti-UAV test native frames @ 0.4:**
  * 271 background FAs on 207 of 2,038 frames, across 50 videos.
  * The worst video is `20190925_200805_1_1` (night) with 23.
  * Scores: median 0.51 (90th percentile 0.66). Widths: median 97 px (10th-90th percentile 35-168). These are large objects, not specks.
* **What they are** (48 random plus 48 with score ≥ 0.6, inspected by eye):
  * **Night lights**, about 40%: street lamps, lit windows, motion-blurred light streaks. Anti-UAV has night sequences whose drones appear as bright blobs.
  * **Ground clutter**, about 35%: buildings, window rows, cranes, power pylons.
  * **Dark foliage or cloud clumps**, about 15%.
  * **Small grey blobs on dark sky**, about 10% (~21 px, `20190925_193610_*`). These might be real but unlabelled drones; not yet checked.
* **Cause:** E12 used **positive tiles only** from Anti-UAV (03 §2.1 rule 1), so it never saw Anti-UAV clutter labelled as "nothing". It learned "drone" from Anti-UAV's bright night blobs and drones against buildings, with no clutter counter-examples.
* **Relevance to deployment:** the planned camera is tilted up with the ground masked, and the current scope is daylight. That would remove much of this, but not lamps, cranes and pylons that stick up above the skyline.

**#35 Project moved to D: (2026-09-25, no training).** The repository moved to `<path-to>\.Drone Detection`. `C:\cuas_data\` (all runs through E12 and `e12_diag`), `C:\cuas_venv` and the Anti-UAV zip stayed on C:. Checks after the move: 18 + 22 tests pass, `check_docs.py` clean, venv reports torch 2.14.0+cu130 with CUDA available and Ultralytics 8.4.162.

### 2026-09-27: E14 (hard-negative mining), bootstrap intervals, E15 preparation

**#36 E14 mining, first pass (local, ~13 min).** New script `experiments/build_e14.py`.
* Method: run E12 (`best.pt`) over the 3,578 Anti-UAV **train** frames (the E12 positives' source frames; Full HD, tiled at 640) at conf ≥ 0.25 (below the 0.4 operating point, so borderline clutter is mined too). Any "drone" box with IoU 0 against the labelled drone is a false alarm. A 640-px tile centred on it becomes a training tile carrying the true labels (usually none).
* Safety rules: drop false alarms narrower than 24 px (possible unlabelled drones, 16 #34); skip tiles that would cut a labelled drone; at most 40 tiles per sequence; total capped at 15% of the final training set (2,555).
* Result: **966 false alarms on 822 of 3,578 frames**; 60 narrower than 24 px dropped; 906 kept (603 with score ≥ 0.4); median width 155 px. → 820 tiles (49 skipped for cutting a drone; 67 also contain the labelled drone).
* Log `C:\cuas_data\runs\build_e14.log`; detections cached in `C:\cuas_data\e14\mined_antiuav_train_frames.json` (first saved as `mined.json`, renamed in #38).

**#37 Visual check found a dataset artefact: a burned-in caption.**
* The contact sheets (`C:\cuas_data\e14\sheet_*.png`) showed that most mined "false alarms" were **not clutter**. They were a white Chinese caption, 民航飞机 ("civil aviation aircraft"), burned into the bottom-left of the frame (text at x 22-170, y 1006-1041) in night sequences. E12 boxes it as "drone" at scores up to ~0.74.
* **The overlay mask misses it completely** (0 of 14,000 mask pixels in that box). `overlay_mask` keeps pixels that are strong edges in ≥ 30% of frames across the dataset; this caption appears only in some sequences, so it never reaches 30%.
* Where it appears (normalised template match, threshold 0.6, every 4th frame; scores are bimodal, only 1 borderline in 0.3-0.6):
  * train: 159 of 894 sampled frames, **31 sequences**;
  * val: 67 of 370, **12 sequences**;
  * **test: 0 of 509**.
* Consequences:
  * **E12's test diagnosis stands:** the test set has no caption, so its 271 native background false alarms (16 #34) are real clutter.
  * **Val is contaminated:** the E12 operating threshold (0.4) was chosen by F1 on Anti-UAV val, which counts caption false alarms. This affects E12 and E14 equally, so the comparison stays fair, but val precision is understated. Recorded as a threat to validity (15 §7.3).
  * **Mining:** ~60% of the mined false alarms were the caption, which would spend the negative budget on an artefact that neither the test set nor a deployed camera has.
* Fix (mining only; the evaluation sets are left unchanged so E14 stays comparable with E12):
  * new `cuas.datasets.antiuav.caption_template` / `find_caption`, a template match in the bottom-left region;
  * `build_e14.py` drops false alarms that overlap a detected caption.
* Why it was missed before: the E12 diagnostics grouped test false alarms into "night lights, about 40%" etc. (16 #34). The test set has no caption, so it only showed up here, on train.

**#38 E14 mining, second pass; the pool is enlarged.**
* After excluding the caption: **499 on the caption dropped**, 60 narrower than 24 px dropped, **407 kept** (204 score ≥ 0.4; median width 95 px). → **328 tiles** (43 skipped for cutting a drone; 64 contain the drone). That is only **2.2% of training images**.
* Visual check of the kept sample: night street lamps and light bokeh, buildings and rooftops, cranes, power pylons, haze edges. No birds seen. **One 24-px grey blob on dark sky** (sequence `20190925_222534`, the same kind of blob as in #34 that may be a real unlabelled drone) passed at exactly the 24-px limit → **`MIN_W` raised to 28 px**.
* Decision: 328 negatives are few, and a weak E14 would cost another ~3.5 h run, while mining more frames costs ~40 min. So the mining pool was enlarged with more frames from the **same train videos**:
  * `cuas.datasets.antiuav.convert` got an `offset` argument (frames with (i - offset) % every == 0);
  * extra folder `C:\cuas_data\antiuav_train_mine`: every 20th frame from offset 13 (frames 13, 33, 53, …), which never coincides with the E12 frames (0, 40, 80, …). These frames are used for mining only, never as positives;
  * first plan was two folders at offsets 13 and 27 (two decode passes); replaced by one pass at every 20 from 13, same density, half the decode time.
  * The train video is re-extracted from the zip (2.93 GB), converted, overlay-inpainted with the same mask, then deleted.
* Command: `C:\cuas_venv\Scripts\python experiments/build_e14.py`, log `C:\cuas_data\runs\build_e14_v2.log`.
* Scripting slips (three, no damage): Python heredocs holding Windows paths (`C:\cuas_data`, `C:\Users\…`) failed on backslash escapes twice (the D:-move doc edit in #35 and the `build_e14.py` refactor), and a heredoc holding this entry failed on shell quoting. Each stopped before writing anything, because the replacements assert before the write. Edits now go through a scratch `.py` file written directly, with raw strings.

**#39 Tooling added while the E14 data builds (no GPU).**
* **Per-video bootstrap confidence intervals** (the planned fix for "~57 validation videos", 15 §7.3). New in `cuas/eval.py`:
  * `video_of(path)`: Halmstad `V_DRONE_001_00040_0_0` → `V_DRONE_001`; Anti-UAV `20190925_111757_1_10_00040` → `20190925_111757_1_10`. Counts checked: Halmstad val 57 videos, Anti-UAV val 67, test 91 (all as documented).
  * `image_counts`: per image, drones / found / false drones of any kind / on background.
  * `recall_precision`.
  * `bootstrap_ci(keys, rows, stat)`: cluster bootstrap that resamples whole videos (2,000 resamples, percentile 95%). Paired comparisons put both models' counts in one row.
  * 3 new tests (one checks that resampling videos, not frames, gives a wide interval when videos disagree).
* `experiments/e12_eval.py` now also prints 95% per-video intervals for native frames (recall, precision, background false alarms per 1,000 frames) and caches native detections in `<run>\dets_native.json`. Zoom mosaics mix several videos, so they get no interval.
* New `experiments/compare_runs.py A B`: paired per-video bootstrap of B − A on native frames, each run at its own val-chosen threshold.
* **E15 preparation (DUT Anti-UAV):**
  * Facts checked in the paper (arXiv 2205.10851): 10,000 detection images, train 5,200 / val 2,600 / test 2,200; resolutions 160×240 to 5616×3744; 35+ UAV types; day/night/dawn/dusk.
  * The annotation format is not stated in the paper. Secondary sources disagree (per-image VOC XML vs TXT/CSV). `cuas/datasets/dut.py` reads VOC XML and stops with a clear message if it finds no XML, so the format is checked on download. 1 new test.
  * New `experiments/e15_dut.py`: evaluates runs at their Anti-UAV-val threshold (nothing tuned on DUT). Images larger than 640 are tiled; **smaller ones are padded, not upscaled** (Ultralytics would otherwise enlarge them and change target pixel sizes). DUT has no video ids, so its intervals resample images (narrower than per-video would be).
* **Drone-vs-Bird:** request procedure verified at the primary source (github.com/wosdetc/challenge, 2026-09-27): email the request to the address on that page, then sign a data usage agreement; research use only. Annotations are one line per frame: `framenum num_objs x y w h class …`.
* Tests: 26 pass (was 22).

**#40 E14 data final; first training start aborted (GPU and memory shared with another project); queued.**
* **Mining result (log `C:\cuas_data\runs\build_e14_v2.log`):**
  * The extra frames: 160 sequences → **7,097 frames** (every 20th from offset 13), overlay inpainted (mask touched 826 drones), extracted video deleted afterwards.
  * Mining over **10,675 frames** (3,578 + 7,097) at conf ≥ 0.25: **2,976 false alarms on 2,567 frames**. Of these, **1,545 were the caption** (dropped), **169 were < 28 px** (dropped), and **1,262 were kept** (637 score ≥ 0.4; widths p10/50/90 45/97/240 px).
  * → 1,017 candidate tiles (151 skipped for cutting a labelled drone) from 95 sequences. The per-sequence cap (40) removed 1 and the 15% cap (2,555) did not bind. **1,016 tiles** chosen, 174 of which also contain the labelled drone.
  * Visual check (`sheet_kept_random.png`, `sheet_dropped_small.png`): kept = buildings, rooftops, cranes, pylons, street lamps and bokeh, haze edges. Dropped-small = mostly the ~21-px grey blobs of `20190925_222534` (possible real drones), correctly excluded.
* **Bug found: duplicate tiles.** Ultralytics scanned **15,480** training images and 830 empty ones, where 14,476 + 1,016 = 15,492 and 842 were expected.
  * The folder held 1,004 files: 11 tile names occurred twice, so 12 writes overwrote identical tiles.
  * Cause: `cuas.tiling.crops_around` returned the same origin twice when two false alarms near a frame edge clipped to the same tile. The "already inside an earlier tile" check uses a 32-px margin, so a box in that margin strip gets a new but identical origin.
  * Fix: `crops_around` skips an origin it already has, with a test. The first version of that test used two boxes 100 px apart that clip to *different* origins, so it failed. The corrected test uses two boxes in the bottom-right corner, and the old code was confirmed to return the duplicate on it.
  * Effect on E14: none on content. The 1,004 unique tiles are what training sees (6.5% of 15,480).
* **First training start (2026-09-27 ~19:00), aborted after ~3 min.**
  * Command: `train_local.py --data C:/cuas_data/e14/data_e14.yaml --name E14_hardneg_yolo26s`.
  * Symptoms: only 3 iterations in 3 minutes (E12 ran at ~3 it/s); 0 GB RAM free, 0.3 GB commit free; the unit tests could not even load OpenCV ("The paging file is too small").
  * Cause: another project's job was running at the same time: Singapore_REITS `announcement_llm.py score gemma4:12b` via Ollama `llama-server`, holding ~18 GB commit and ~5 GB of the 8 GB GPU at ~97% utilisation.
  * Action: stopped **only the E14 process tree** (the other job belongs to the user and was left alone). The log was kept as `C:\cuas_data\runs\E14_hardneg_yolo26s\reports\E14_hardneg_yolo26s_aborted1.log`. No weights were produced (it never finished an epoch).
* **Queued instead:** `experiments/run_e14_queue.sh` waits until the GPU uses < 1,500 MiB, then runs (1) `e12_eval.py E12_multidata_yolo26s` (per-video intervals and `dets_native.json`, output `eval_E12_multidata_yolo26s_v2.txt`) and (2) E14 training. Progress in `C:\cuas_data\runs\e14_queue.log`. Started 19:33, waiting on the GPU.
* **Practical lesson:** check `nvidia-smi` and free memory before a long local run. Other projects on the same laptop (local LLMs especially) can take the GPU and the paging file.

**#41 E12 re-evaluation (queued, local), 2026-09-28 01:33-01:46.** The GPU freed at 01:33 and the queue ran `e12_eval.py E12_multidata_yolo26s`, exit 0 (`eval_E12_multidata_yolo26s_v2.txt`, `dets_native.json` cached).
* Anti-UAV native @ 0.4, 95% CI over 91 videos: recall 0.899 [0.864, 0.932], precision 0.689 [0.659, 0.720], background false alarms 133 [97, 174] per 1,000 frames.
* Halmstad val native @ 0.4, CI over 57 videos: recall 0.848 [0.736, 0.938], precision 0.995 [0.986, 1.000].
* Halmstad pooled printed `w_min: 64 px` because the 32-64 px bin dips to 0.79 recall (bins below it are 0.83-0.98). This is an artefact of the w_min definition on one dip, not a regression. Check it when writing up.

**#42 E14 training (local RTX 4060), started 2026-09-28 01:46; laptop hibernated 2 min in.**
* Config as E12 (checked in `args.yaml`): 40 epochs, patience 15, batch 16, imgsz 640, seed 0, workers 4, cache off. 968 iterations/epoch (E12: 905; +7% matches the 1,004 negative tiles).
* **Pause:** Windows hibernated at 01:48:34 ("Hibernate from Sleep, Fixed Timeout") and resumed 08:52. `SetThreadExecutionState` blocks idle sleep, not a hibernate timer out of standby. Epoch 1's `time` in `results.csv` (26,698 s) is therefore mostly the pause. The training state is unaffected.
* Epoch 1 val P / R / mAP50 / mAP50-95: 0.65 / 0.70 / 0.72 / 0.33 (E12 epoch 1: 0.53 / 0.67 / 0.68 / 0.31).
* **After resuming, the laptop was on battery:** 1.5-1.8 it/s (E12 on AC: 3.1-3.4), GPU ~20 W, and only ~0.6 GB commit free.
* **Practical lesson:** for overnight runs, stay on AC and set sleep/hibernate to "never" on AC. Otherwise run on Colab.
* Back on AC (~09:20): 3.1-3.2 it/s, GPU ~39 W; epoch 3 took 363 s.
* Epochs 1-3 val P / R / mAP50 / mAP50-95:
  * 2: 0.75 / 0.73 / 0.78 / 0.36
  * 3: 0.68 / 0.69 / 0.74 / 0.34
* **Paused at the user's request, 2026-09-28 ~09:27.** The whole queue process tree was stopped ~1 min into epoch 4, so the queue's closing log lines never ran. `last.pt` holds the end of epoch 3 (09:26:06). Resume: `train_local.py --resume C:/cuas_data/runs/E14_hardneg_yolo26s/weights/last.pt`, then the E14 eval and compare manually.
* **Resumed 2026-09-28 09:39** ("Resuming training … from epoch 4 to 40 total epochs"), detached with `Start-Process`, output in `E14_hardneg_yolo26s_resume1.log`. Ultralytics progress goes to stdout here, not stderr.
  * The first resume attempt failed: `train_local.py` required `--data`/`--name` even with `--resume`, which contradicts its own usage line. Fixed so they are required only for a fresh run.
  * The laptop is on AC in Windows power-saving mode (the user's choice): ~2.0 it/s (1.9-2.6), ~8.5 min/epoch + ~40 s val. That is up to ~5.5 h for the remaining epochs, less with early stopping. Throttling changes speed only, not results.
  * Epoch 4 val P / R / mAP50 / mAP50-95: 0.77 / 0.78 / 0.84 / 0.42 (E12 epoch 4: 0.80 / 0.74 / 0.82 / 0.41).
  * Epochs 5-10 (~9 min each):
    * 5: 0.75 / 0.83 / 0.84 / 0.41
    * 6: 0.82 / 0.79 / 0.85 / 0.41
    * **7: 0.83 / 0.82 / 0.88 / 0.45 (best so far, E12 epoch 7: 0.89 / 0.82 / 0.89 / 0.46)**
    * 8: 0.83 / 0.76 / 0.85 / 0.43
    * 9: 0.81 / 0.80 / 0.84 / 0.42
    * 10: 0.79 / 0.73 / 0.81 / 0.40
* **Paused again at the user's request, 2026-09-28 10:39**, 1.5 min into epoch 11. `last.pt` = end of epoch 10 (10:37:59), `best.pt` = epoch 7. Resume with the same command.
* **Superseded (2026-09-28):** the user moved GPU work to Colab (L4). E14 will be retrained fresh there (#44). The local run stays on disk for reference only.

**#43 M2: tracker + track features on Halmstad oracle tracks (local CPU, 2026-09-28).** No GPU.
* New code: `cuas/track.py`, `cuas/track_features.py`, 6 tests (32 total). Script `experiments/m2_halmstad_tracks.py`.
* A test failure found a real artefact: a new track starts at zero velocity, so its first estimates read as hovering. Velocity features now skip 3 warm-up rows.
* `extract` (~3 min, `C:\cuas_venv`): 285 videos at 30 and 10 fps; ground-truth boxes with 5% centre/size jitter, 10% dropped; silhouettes from the real frames. **9,206 samples** (1/2/4-s windows and whole tracks) in `C:\cuas_data\m2\features.csv`.
* `classify` (scikit-learn 1.9.1 + pandas 3.0.6 installed into `C:\cuas_venv` for this): 5 folds grouped by video. Output in `C:\cuas_data\m2\classify_v1.txt`.
  * The first run crashed: in 1-s windows the periodicity columns are all NaN (they need ≥ 1 s of samples), and the classifier rejects an all-NaN column. Such columns are now dropped per window length.
  * Results in 14 §5: AUC 0.876 (1 s) → 0.923 (whole track) at 30 fps; 10 fps about the same overall but weaker shape/flap features; confound probe 0.72-0.76; ECE 0.08-0.10.
* Tracker check: median video one track. The worst 10% (mostly birds) fragment at label jumps of 7-20 box widths in one frame, i.e. the label switching between flock members.

**#44 E14 moved to Colab L4 (prepared 2026-09-28, not yet run).**
* `experiments/pack_for_colab.py` → `C:\cuas_data\colab\cuas_data_e14.zip` (1.11 GB, 46,642 files: E14 training data and all `e12_eval.py` evaluation sets) + `cuas_code.zip`.
* `e12_eval.py`, `compare_runs.py`, `e15_dut.py` and `train_local.py` read an optional `CUAS_DATA` variable (default `C:/cuas_data`).
* Cells in `experiments/colab_e14_cells.md`: setup (Ultralytics pinned to 8.4.162, runs symlinked to Drive), a 1-epoch smoke test, the full run as `E14_hardneg_yolo26s_L4`, resume, and evaluation. `compare_runs.py` then runs locally on the downloaded JSONs.
* Fresh, not resumed: the local checkpoint stores Windows data paths. **E12 was trained on the RTX 4060 and E14 on an L4**, so confirm a small difference with a second seed or an E12 rerun on the same GPU.
* **Superseded 2026-09-29 by `experiments/colab_run_all.py`** (#45): one cell with the same stages plus the M2 detector runs, and resume-after-disconnect. The data zip now also carries E12's `best.pt`.
* A script slip, no damage: a Python heredoc turned the `\r` and `\n` of `tr '\r' '\n'` into real control characters in the cells file. Found by grep and repaired from a scratch `.py` file with a raw-string regex (same lesson as #38).

**#45 M2 built through (local, code only; 2026-09-29).** No GPU.
* Checked on request:
  * The DUT test link on github.com/wangdongdut/DUT-Anti-UAV resolves to `test.zip` on Google Drive (sign-in page, normal for large files).
  * Kaggle "muhammadsaoodsarwar/drone-vs-bird" is a classification image set, not the WOSDETC video dataset (03 §2).
* New code:
  * `IMMFilter` + `Tracker(motion="imm")` and `ttc_expansion`;
  * `cuas/classify.py` (cross-fitted isotonic calibration, unknown output, named feature groups);
  * `cuas/rules.py` (doc 12 floor), `cuas/pipeline.py`, `cuas/datasets/dvb.py`;
  * `experiments/m2_detect_videos.py`, `experiments/colab_run_all.py`;
  * `m2_halmstad_tracks.py` rewritten for oracle or detector input, calibration, unknown rate, held-out val videos, and saving the pipeline classifier.
  * 6 new tests (38 total).
* **Found by tests:**
  * **TTC bias.** Fitting ln(width) over the last second gives the TTC at the window's middle, so it read 3.5 s against a true 3.0 s. The lag is now subtracted. The first TTC test's ±0.5 s tolerance had hidden it; tightened to ±0.1 s.
  * **Label-leak trap avoided in design.** Oracle detections carry the true class, so the detector-class features (`cls_drone_frac`, `score_mean`) would give the answer away. Oracle runs use motion/shape features only; detector features are used only with real detections.
  * **Slow classifier fits.** The classifier unit test took ~180 s for 5 small gradient-boosting fits, where well under 1 s is expected. Suspected OpenMP thread contention on Windows. Not yet diagnosed: the command safety check was failing for the rest of the session.
* **Not yet run after the rewrite:** the full test suite, `pack_for_colab.py` (the zips must be rebuilt: code changed, and E12 weights were added), and `m2_halmstad_tracks.py extract/classify` on the oracle boxes.
* The first shell edit of `classify.py` failed on the command safety check; the same change was made with the editor.

**#46 Housekeeping, DUT conversion, M2 verification (local CPU, 2026-09-29).**
* The shell worked again. **Tests:** 38 `cuas` + 18 tools pass; `check_docs.py` clean.
* **Slow classifier fits diagnosed:** OpenMP oversubscription. One 400-sample fit: 3.09 s (16 threads), 0.09 s (4), 0.06 s (1). `cuas/classify.py` and `m2_halmstad_tracks.py classify` now cap OpenMP at 1 thread (`threadpoolctl`). The suite went from ~190 s to ~15 s.
* **M2 oracle rerun** (`extract` ~3 min, 9,206 samples as before; logs `C:\cuas_data\m2\extract_oracle_v2.log`, `classify_oracle_v3.log`):
  * The first `classify` printed ECE → 0.000: **in-sample calibration** (isotonic fitted and scored on the same out-of-fold scores). Fixed by cross-fitting the calibrator (a second grouped split). A unit test now requires the cross-fitted ECE not to exceed the raw one by more than 0.02.
  * The second pass computed AUC on the cross-fitted probabilities and read ~0.01 lower (ties from per-fold calibrators). AUC is now computed on raw scores and reproduces #43 exactly.
  * The target track's share of labelled frames now reads a median 89% (was 100%). The denominator now counts all labelled frames, including the 10% deliberately dropped; this is not a tracker change.
  * Results in 14 §5. Pipeline classifier saved as `C:\cuas_data\m2\track_clf_oracle.pkl`.
* **DUT Anti-UAV test** (`%USERPROFILE%\Downloads\test.zip`, 271 MB): 2,200 JPGs + 2,200 VOC XMLs in `test/img` and `test/xml`, every object named `UAV` (2,245). `cuas.datasets.dut` worked unchanged → `C:\cuas_data\dut_test`. Widths 12/26/55/260/865 px (min/p10/median/p90/max); images 640-1920 px wide. File names are plain numbers (no video ids), so E15 intervals resample images.
* **Colab bundle rebuilt:** `cuas_data_e14.zip` 1.41 GB (51,042 files) now adds `dut_test` and E12's `e12_eval.json` (threshold for E15). `colab_run_all.py` gained stage 6 (E15).
* **Checked on request:**
  * arXiv 2601.08319 (YOLOBirDrone / BirDrone) is still images from Chandigarh, mostly medium/large targets, partly released, no licence file. Added to `papers_database.json` and 01 §2.8 (33 verified papers), plus a row in 03.
  * It does not replace Drone-vs-Bird for M2, which needs video.

**#47 E14 + M2 detector runs + E15 launched on Colab (user, L4, 2026-09-30).**
* The user uploaded the two zips and ran `experiments/colab_run_all.py` in one cell. It is in progress; results expected in `MyDrive/cuas_runs/`.
* The user emailed data requests for both Drone-vs-Bird (WOSDETC) and BirDrone.
* YOLOBirDrone's architecture was reviewed on request (01 §2.8, 11 row 8). It is not adopted: it ties plain YOLOv8 in single runs, and no code is released.
* **Colab progress pasted by the user:**
  * Setup and smoke run OK (1 epoch at 5%: mAP50 ~0, as expected).
  * E14 runs at ~2.6 min/epoch on the L4 (968 batches at ~7 it/s + ~20 s validation). Validation mAP50 by epoch: 0.746, 0.745, 0.658, 0.779 (epoch 3 dipped; epoch 4 recovered). 15,480 train / 3,221 val images, as built.
  * The setup line's "expected ~46,000" was a wrong constant in `colab_run_all.py`; the zip holds 25,520 images. Fixed; the data was right.

**#48 E18 built: YOLOBirDrone-style add-ons (local CPU + GPU venv, 2026-09-30).**
* The user asked to implement AELAN, MPDA and RMPDA. Built as `cuas/arch.py`, our reading of an underspecified paper (14 §2). Also: `ArchTrainer`, `train_local.py --arch/--seed`, `tests/test_arch.py` (10 tests, pass), `experiments/e18_arch_bench.py` and `experiments/colab_e18.py`.
* `e15_dut.py` gained `--out`, so E18 does not overwrite E14's `e15_dut.json`. `pack_for_colab.py --code` rebuilds only the code zip, which now includes `tests/test_arch.py`.
* **ONNX:** torchvision has no exporter for `deform_conv2d`, so I wrote the mapping to ONNX `DeformConv` (opset 19). ONNX Runtime CPU matches PyTorch. Installed `onnx`, `onnxruntime` and `onnxscript` in `C:\cuas_venv`.
* **Benchmark, first attempt discarded.** The GPU was at 6.3 GB / 81% (a resident Ollama `llama-server` from another project): base YOLO26s read 23.6 ms, and the ratios were erratic (RMPDA timed faster than base). `nvidia-smi` was checked, but the run went ahead anyway. Rerun with the variants interleaved over 3 rounds, best round kept. Ratios in 14 §5: full set 1.78× GPU / 1.57× CPU; ECA 1.06× / 1.04×.
* **End-to-end CPU test** (scratch, 16 tiles, imgsz 320): fresh → simulated disconnect after epoch 1 → resume in a new process → finish → reload → predict. It passed, but the first pass took **zero optimizer steps** (nominal batch 64 = 16 accumulations of batch 4, and only 12 iterations ran), so no weight moved. With `nbs=4` the add-ons and the pretrained layers trained.
* No local GPU training (the GPU is shared); the accuracy runs are E18 on Colab.


**#49 Drone-vs-Bird: organisers' reply; annotations verified (local CPU, 2026-09-30).**
* The user reports that the organisers provide the training set only; testing data and annotations are not provided.
* The training annotations are public on github.com/wosdetc/challenge (77 files; repo last pushed 2022-12). Downloaded to `C:\cuas_data\dvb\annotations`; all 77 parse with `cuas.datasets.dvb`.
* Numbers: 106,485 frames, 119,259 boxes, frames start at 0, 9,086 empty frame lines, 12,061 frames with more than one drone.
* Class `drone` only. Widths 4 / 11 / 25 / 71 / 1,403 px (min / p10 / median / p90 / max).
* To check when the videos arrive: every video has an annotation file (a newer edition may send more), and the frame rates.


**#50 E14 on Colab L4: trained and evaluated; M2 stage crashed (2026-09-30).**
* **Training:** `E14_hardneg_yolo26s_L4`, 28 epochs (EarlyStopping; best epoch 13), 1.224 h, ~2.6 min/epoch, 6.1 GB GPU. Best val (Ultralytics): P 0.870, R 0.849, mAP50 0.896, mAP50-95 0.460 (E12: mAP50 0.899).
* **`e12_eval.py`:**
  * The val sweep chose conf **0.55** (E12: 0.40).
  * Anti-UAV test native: recall 0.762 [0.702, 0.817], precision 0.877 [0.845, 0.907], background 14.7 [6.4, 24.5] per 1000 frames.
  * Halmstad val native: recall 0.138 [0.048, 0.251], precision 0.980.
  * Pooled Anti-UAV val `w_min` 4 px; test `w_min` None (recall under 0.8 in the 64+ px bin).
* **Analysis:**
  * `compare_runs.py --sweep` on E12's cached detections: E12 at 0.70 on test gives 0.757 / 0.878 / 6.4. That is the same operating point as E14 at 0.55, so the gain at own thresholds is mostly threshold movement.
  * Halmstad drones score 0.40-0.55 for both models (E12: 0.848 at 0.40, 0.310 at 0.55).
  * Details in 15 §6.5. Still to run locally: E14's full sweep and the `--match` paired comparison, once `dets_native.json` is downloaded.
* **M2 stage failed** after running E12 over all 285 videos: `json.dump` could not serialise numpy float32. The Colab cell showed only the traceback's line numbers, because the stage printed only lines matching "videos". Fixed in the detector and the script, plus a fail-fast check after the first video; `sh()` now prints the last 20 lines on any failure. Stages 5 (M2) and 6 (E15) still to run: re-run `colab_run_all.py` with the new code zip (finished stages are skipped).


**#51 Colab re-run: M2 detections + E15; local analysis (2026-09-30/10-01).**
* Colab (user) re-ran `colab_run_all.py` with the fixed code zip; finished stages were skipped. The cell text still printed the old "expected ~46,000" message (the fix is in the file, not the pasted cell; harmless).
* **M2 detections:** E12 and E14L4 over all 285 Halmstad videos (89,233 frames), 544 s and 539 s. Downloaded to `C:\cuas_data\m2\dets_E12.json.gz` and `dets_E14L4.json.gz`.
* **E15** on Colab (`e15_dut.txt`): E12 recall 0.141, precision 0.540; E14 0.076 / 0.612. Re-run locally on the 4060 (GPU idle) with the new background split (`e15_dut_local.txt`/`.json`), reproducing the recall exactly (precision 0.538 vs 0.540). Probe contact sheet: ground truth correct; drones called helicopter/airplane; many DUT scenes have ground/building backgrounds.
* E14 folder downloaded to `C:\cuas_data\runs\E14_hardneg_yolo26s_L4\`. `compare_runs.py --sweep / --match / own` → `C:\cuas_data\runs\cmp_E12_E14L4.txt`. Results in 15 §6.5.
* The first `m2_halmstad_tracks.py extract --dets` attempt used the global Python and failed (`mcos_decoder` is only in the venv). Re-run with `C:\cuas_venv\Scripts\python`.
* **M2 real-detection extract** (~5 min each): E12 9,268 samples, target tracked in 271/285 videos; E14 9,166 samples, 267/285.
* **Classify:** track AUC 0.918 / 0.933 on held-out videos at 1 / 2 s (E12); detector votes 1.000 (in-domain); confound probe 0.74-0.78.
* `classify` now also saves a track-only pipeline classifier (`track_clf_dets_E12_trackonly.pkl`: 12% unknown, 92.6% accurate on decided windows), because E15 showed the class votes do not transfer. Results in 14 §5.


**#52 E13 prepared (local, 2026-10-01).**
* `train_local.py --classes` (Ultralytics' training class filter). Verified on Halmstad tiles: `YOLODataset(classes=[0])` drops an airplane tile's label and keeps a drone tile's. A 1-epoch GPU smoke run on the 16-tile set ran.
* `colab_e18.py` now runs **E13_droneonly_s0 first**, then the E18 runs. E15 runs straight after each run's evaluation (`e15_<run>.txt/json`) instead of once at the end. Code zip rebuilt.
* Design note: the DUT confusion is mostly *helicopter* (347 of 547), which is not a COCO class. It comes from Halmstad's own helicopter videos, so `cls_remap=False` alone would not fix it; drone-only training removes the competing classes.


**#53 `colab_e18.py` re-run guard (local, 2026-10-01).**
* The user's first `colab_e18.py` run died (cause not yet seen). Re-running is the intended recovery (DONE_E18_* markers, resume from `last.pt`), with one hole: if the runtime died after training finished but before its marker was written, `last.pt` is already stripped by Ultralytics' `final_eval` (`epoch = -1`) and `--resume` fails with "nothing to resume".
* Fix: `finished(last)` reads the checkpoint's epoch; a stripped one marks training done and goes on to evaluation. Checked on E14's `last.pt` (epoch -1). Code zip rebuilt.
* Later the same day the user paused a run and asked to re-run. Colab's stop button only interrupts the kernel, so the training subprocess can keep running headless and collide with the resumed one. `sh()` now starts each command in its own process group and kills the group on interrupt; setup also `pkill`s leftover `train_local`/`e12_eval`/`e15_dut`/`compare_runs`/`unittest` processes from older cells (pattern checked not to match the kernel or itself).


### 2026-10-01: Design review fixes, system evaluation, degradation sweep

Why and what it means: [19](19_design_review_parameters_and_novel_directions.md).

**#54 Design review: code fixes (local, 2026-10-01).** Global Python for the tests, venv for anything with `mcos_decoder` or the GPU.
* **Review:** all of `cuas/`, `experiments/`, `tests/`, `tools/` read; 10 findings, each re-measured before fixing, then 6 more found while fixing (19 §2).
* **Fixes:**
  * `.gitignore`: `/data/`, `/datasets/`, `/runs/` anchored to the root. The bare `datasets/` had kept `cuas/datasets/` (the four converters) out of git; none was on `origin/main`. **They are untracked files until committed.**
  * `cuas/track.py`: confirmed tracks are matched before tentative ones; `history_s` / `keep_finished` bound memory for live use; `Track.t0`.
  * `cuas/detector.py`: `nms(class_aware=False)` for the tracker. `cuas/pipeline.py`: uses it; bounded history and silhouette store; `drone_prob` is `None` until classified; `coasting` flag.
  * `cuas/rules.py`: TTC of 0 counts as imminent; the classifier's label wins over p; a closing unknown gets `slew_ptz_for_identification`.
  * `cuas/track_features.py` `ttc_expansion`: a straight-line fit of **1/width** (exact for constant closing speed; replaces ln(width) plus a lag correction), slope significance z = 4, at least 0.5 s of boxes, 2-s window (was 1 s, no tests).
  * `cuas/motion.py`: the registration reference follows the scene (`ref_alpha` 0.02, refreshed every 10 frames; restart above `max_shift` 16 px).
  * `cuas/tiling.py`: a box starting off-frame no longer crashes `training_tiles`; `tile_split` writes no tile for a mostly hidden target.
  * `experiments/train_local.py`: optimizer pinned (AdamW, lr 0.00125, momentum 0.9, `warmup_bias_lr` 0), equal to what `auto` chose for every run so far; `--optimizer`, `--lr0`, `--freeze`.
  * `experiments/colab_e18.py`: `checkpoint()` replaces `finished()` (a truncated `last.pt` is set aside, resume from `best.pt`); `E18_base_s2` queued after `E18_full_s0`. `colab_run_all.py` got the same interrupt handling.
* **Tests:** `tests/test_cuas.py` 38 → 47 (57 with `test_arch.py`'s 10, skipped outside the venv), ~60 s (the whole-program test renders and runs a 3.5-s scene). The seven regression tests written first were run against the committed tree (`git archive HEAD`) and all failed there.
* **Measured before/after:**
  * Tracker on E12 detections, all 285 videos, 30 fps: confirmed tracks 570 → 516, non-target tracks 6.1 → 5.1 per minute, target-track observations +1.4%. The target track is found in 269 videos (was 271): V_BIRD_003 and V_BIRD_006, whose labels jump between flock members, no longer pass the "matches the label in half its frames" rule.
  * `m2_halmstad_tracks.py classify --tag dets_E12` re-run: track AUC on held-out videos 0.907 (1 s) / 0.936 (2 s), was 0.918 / 0.933; inside the intervals. Pipeline classifiers re-saved; the previous files are in `C:\cuas_data\m2\pre54\`.
  * TTC, simulated 5% white width jitter at 30 fps: non-closing tracks reading under 10 s within 3 s: 75% (old) → under 0.5%. A true 10 s is seen 29% of the time with a 1-s window and always with 2 s.
  * Motion cueing on a sky that cross-fades to new clouds: hit rate 0.44 → 1.0, false blobs per frame 56 → 0.3.
* **New code:** `cuas/sim.py` (scene simulator, PX4 log reader, stand-in detector), `cuas/replay.py` rewritten as the program, `cuas/__main__.py` (`python -m cuas`), `experiments/e20_system_eval.py`, `experiments/e21_degradation.py`.
* Code zip rebuilt (`python experiments/pack_for_colab.py --code`). The Colab queue is unaffected until the new zip is uploaded; with it, training is identical (the pinned optimizer equals `auto`).

**#55 Colab L4: E13 and `E18_base_s1` finished; `E18_full_s0` training (user, 2026-10-01).**
* `E13_droneonly_s0`: finished (its stages were skipped on the re-run). **Its output has not been sent yet:** `eval_E13_droneonly_s0.txt`, `cmp_E13_droneonly_s0.txt`, `e15_E13_droneonly_s0.txt` on Drive.
* **`E18_base_s1`** (the E14 recipe, seed 1): ran all 40 epochs. Threshold chosen on val: 0.55 (as E14).
  * Pooled: Anti-UAV val recall 0.914 / precision 0.827, `w_min` 8 px; **test** 0.788 / 0.895, `w_min` 12 px; Halmstad val 0.686 / 0.998.
  * Native, 95% per-video intervals: Anti-UAV test recall 0.843 [0.795, 0.888], precision 0.792, 12.8 background FAs per 1000 frames; Halmstad val recall 0.599 [0.435, 0.748].
  * Against E14 (seed 0) at matched recall (seed 1 at 0.71): test background FAs −13.2 per 1000 [−22.3, −5.5]; Halmstad recall +0.054 [+0.003, +0.116]; val precision +0.023 [+0.003, +0.040]. Same threshold, different recall: test at 0.70 is 0.468 (E14) vs 0.787; Halmstad at 0.55 is 0.138 vs 0.599.
  * E15 (DUT): recall 0.149 [0.134, 0.164], precision 0.699; sky backgrounds 0.18 as drone / 0.33 as another class / 0.49 missed. E14 had 0.076.
  * **Reading (19 §3.1):** two seeds of one recipe differ "clearly" under the per-video bootstrap, so that interval does not cover training variance. At test recall ~0.90, background FAs per 1000 frames: E12 best 133, E12 last 62, E14 seed 0 159, this run 64. E14's hard-negative gain (#51) is therefore not established. A third seed (`E18_base_s2`) is queued.
* `E18_full_s0` (aelan+mpda+rmpda): "Transferred 708/862 items"; 4.3 it/s (968 iterations, ~3.75 min per epoch against ~2.6 for the base). Val mAP50 over its first seven epochs: 0.711, 0.724, 0.581, 0.809, 0.860, 0.868, 0.840. E14 read 0.746 at epoch 1 and 0.888 at epoch 6: no visible difference so far.

**#56 Local: E21, E20, checkpoint comparison, program on real video (2026-10-01).** RTX 4060 (free at the start; `nvidia-smi` checked), venv.
* **E21** (`experiments/e21_degradation.py E12_multidata_yolo26s`, ~15 min): E12 at 0.4 on degraded Halmstad val native / k4 and Anti-UAV test k8. Recall clean 0.848 / 0.931 / 0.857; noise σ 4: 0.663 / 0.619 / 0.815; noise σ 8: 0.248 / 0.239 / 0.674; blur σ 1.5: 0.629 / 0.349 / 0.768; JPEG 25: 0.805 / 0.280 / 0.651; blur 1.0 + noise 6 + JPEG 40: **0.373 / 0.076 / 0.462**. Full table in 19 §3.2. `C:\cuas_data\runs\E12_multidata_yolo26s\reports\e21_E12_multidata_yolo26s.txt` / `.json`. This also answers most of E5 (compression).
* **E20** (`experiments/e20_system_eval.py --tag dets_E12`, ~5 min CPU): the pipeline over the 57 Halmstad val videos from cached detections, track classifiers fitted on 979 windows of 213 train videos. No false drone declaration by the detector's class or by detector+track on 5.9 minutes of non-drone video (upper 95% bound 30 per hour); the track-only classifier makes 7 (71 per hour), 1 with a 1-s hold. 23 of 23 drone videos declared, median 1.03 s after the drone appears. `C:\cuas_data\m2\e20_system_dets_E12.txt` / `.json` / `.csv`.
  * A first version of the unknown rule (alert under 3 s) raised 4 alerts on the non-drone videos; a count over all 285 videos found 0.8-1.9 tracks per minute reading TTC under 3 s on distant aircraft and birds. The rule now only asks for identification (19 §2 #16). Final run: 0 alerts on non-drone videos and 13 alerting tracks on drone videos (detector votes); 3 and 13 with the track-only classifier.
* **Checkpoint comparison** (scratch script; native frames): E12 `last.pt` (epoch 36) against `best.pt` (epoch 21). At matched test recall ~0.895: 62 against 133 background FAs per 1000; Halmstad at 0.40: recall 0.943 against 0.848. On Anti-UAV val the order reverses. E14 `last.pt` (epoch 28) against `best.pt` (epoch 13): 46 against 159 at test recall ~0.90, and val agrees. Table in 19 §3.4; raw output in `C:\cuas_data\runs\best_vs_last.txt`.
* **The program on real video** (`C:\cuas_venv\Scripts\python -m cuas <video> --weights ...E12...best.pt --clf C:/cuas_data/m2/track_clf_dets_E12.pkl --out ...`): V_DRONE_037 declared at 1.07 s; V_BIRD_034 and V_AIRPLANE_017 no declaration, no alert. ~52 ms per frame detector (GPU shared with the checkpoint comparison), ~8 ms tracking and classification. Outputs in `C:\cuas_data\demo\`.
* **The program on a simulated scene** (`python -m cuas.sim --out C:/cuas_data/sim/approach.avi`, then `python -m cuas ... --weights blob`): three targets, three tracks, one escalation (the closing drone). The first attempts exposed the flap and tentative-crop findings (19 §2 #12, #14) and that a 41-px square structuring element fires on cloud corners (an ellipse does not).
* Issue: the desktop app was quit mid-session; the two background jobs kept running as orphans and one script was started twice. No results were lost.

**#57 Colab L4: `E18_full_s0` finished; epochs rule corrected; final-weights scoring (2026-10-01, late).**
* **`E18_full_s0`** (aelan+mpda+rmpda, seed 0): resumed at epoch 37, ran all 40 epochs. 4.2-4.3 it/s; 0.069 h per epoch (4 epochs in 0.276 h) against 0.051 h for `E18_base_s1`'s final epoch, about 1.35×; `E18_attn_s0` runs at 5.4 it/s. Validation mAP50 at epochs 37-39: 0.850, 0.849, 0.852; `best.pt` 0.892.
  * Threshold chosen on val: 0.55. Pooled: Anti-UAV val recall 0.899 / precision 0.828, `w_min` 8 px; **test** 0.758 / 0.888, `w_min` 12 px; Halmstad val 0.618 / 0.996.
  * Native, 95% per-video intervals: Anti-UAV test recall 0.791 [0.733, 0.847], precision 0.841 [0.805, 0.876], 11.3 background FAs per 1000 frames; Halmstad val recall 0.398 [0.238, 0.571].
  * Against E14 at matched recall (full model at 0.62): val background FAs **+43.9 per 1000 [+5.8, +91.1]** (80 against 15), val precision −0.028; test background FAs −8.3 [−16.9, −0.5] (13 against 30), test recall −0.014 (not clear); Halmstad recall +0.088 [+0.012, +0.168].
  * E15 (DUT): recall 0.106 [0.094, 0.119], precision 0.842; sky backgrounds 0.14 as drone / 0.35 as another class / 0.51 missed; 357 drones called helicopter, 141 airplane.
  * **Reading (19 §3.5):** between the two plain seeds on test recall (0.762 / 0.843), Halmstad (0.138 / 0.599), DUT (0.076 / 0.149) and matched-recall test false alarms (30 / 3); worse than both on Anti-UAV val. No evidence of a gain.
* The user's cell (pasted before the queue change) went on to `E18_attn_s0` (2 epochs in when reported).
* **Epochs.** The user asked whether 25 epochs is enough. It was the wrong number: it came from validation-mAP best epochs (12-21), while the measured comparison favours later weights (epoch 28 over 13, 36 over 21, #56). Rule changed to the full 40 epochs with no early stopping; whether 80 is better is now a queued run.
* **Code:** `cuas.eval.run_files`; `e12_eval.py`, `compare_runs.py` and `e15_dut.py` accept `<run>@last` (final weights; results in `e12_eval_last.json`, `dets_native_last.json`). `colab_e18.py`: `evaluate()` scores every run's `best.pt` (against E14) and its final weights (against `E18_base_s1@last`); queue is now E13, base s1, full (all done), **`E18_base_s2`** (`--patience 100`), **`E18_base_s1_e80`** (`--epochs 80 --patience 100`), then the three ablations. Code zip rebuilt. Test added (`test_run_spec`); 48 tests.
* Local, RTX 4060: `e12_eval.py E12_multidata_yolo26s@last`, `E14_hardneg_yolo26s_L4@last`, then `compare_runs.py --match` between them → `C:\cuas_data\runs\eval_*_last.txt`, `cmp_E12_E14L4_last.txt`.

**#58 Recipe compared with published recipes (local, 2026-10-01, late).** No training.
* Question from the user: has every detail of the training been compared with the most successful papers, repositories and models? It had not; 19 §9 now does it.
* **Read directly:** `yolo26s.pt`'s own `train_args` (`torch.load`): pretrained on Objects365 (`yolo26s-objv1-150.pt`), then COCO for 70 epochs, patience 100, batch 128, MuSGD lr 0.00038, scale 0.9, mosaic 0.99, mixup 0.05, copy-paste 0.40, loss gains 9.83 / 0.65. Our runs use the package defaults, which differ on almost every line.
* **Read from full texts:** the 2025 Drone-vs-Bird winner (YOLOv11m, SGD 0.01, batch 16, patience 15, mosaic 0.1, scale 0.3, degraded copy-paste, four-segment multi-scale inference worth +0.19 mAP50, ±6-frame consistency, trained on DUT Anti-UAV among others); YOLOMG (3-frame difference input +0.45 AP on ARD100, small-object layer +0.02); D-FINE README (Objects365 start recommended, backbone lr 1/20).
* **Run here:** the installed `TaskAlignedAssigner` on boxes of 4-128 px with perfect predictions. Positives per box, one-to-many head: 5.3 for sides ≤ 16 px, 10 from 24 px. With the P2 model (`yolo26s-p2.yaml`, 9.77 M parameters, 27.8 GFLOPs): 5.3 for ≤ 8 px, 10 from 12 px.
* **Consequences (19 §9.3):** E22 gains an augmentation arm (mosaic 0.1 / scale 0.3, degraded copy-paste); **E27** (Objects365 start) added; E23 (temporal input) moved ahead of E24; E7 (P2) moved down; DUT's train split to be added to training with its test split still held out.
* References added and verified at arXiv: SAHI, Scale Match, Kisantal et al., TOOD (52 verified papers).

**#59 Wider literature sweep; E28 multi-scale inference; final-weights evaluation (local, 2026-10-01, night).**
* **Sweep (19 §10):** about 30 further sources checked at arXiv abstract pages or by DOI through Europe PMC (the publisher's pages returned 403): long-range drone datasets, the small-bird detection and tracking challenges, tiny-object label assignment and DETR variants, robustness and single-domain generalisation, missing-annotation training, confidence calibration, open-vocabulary detectors, event cameras. 77 verified papers in the database; 01 §2.10, §8.6 and §9; 03 gained LRDDv2/v3, SOD4SB and SMOT4SB.
* **LRDDv2 / LRDDv3** (real small drones with range labels) looked like the missing dataset, but both project pages say access is **on hold** and name a licence with a US export-regulation condition. Recorded as a request to send, not as available data.
* **Verdict:** the architecture matches current ground-to-air practice (the closest published model is a YOLO26-style detector plus a P2 head). The faults are around it: clean-only training, sparse labels treated as complete, no object-level augmentation, IoU matching at a few pixels, one threshold for all sizes, a thin bird class. Two suspected faults were measured and are small: one inference scale (E28) and unlabelled objects in training tiles (the count below).
* **E28** (`experiments/e28_multiscale_dut.py E12_multidata_yolo26s`, local GPU, ~12 min; `C:\cuas_data\runs\E12_multidata_yolo26s\reports\e28_E12_multidata_yolo26s.txt/.json`): DUT recall 0.141 with native tiles, 0.165 with a whole-frame pass added, 0.170 with a half-scale pass as well; 64-128 px 0.10 → 0.19, 128-256 px 0.06 → 0.12, 256+ px 0.01 → 0.02; background false drones 184 → 235 → 277. A coarse look is not the fix for DUT: most misses are still the class (526 missed drones boxed as helicopter, airplane or bird). Not added to the program.
* **Final weights, E12 against E14** (`e12_eval.py <run>@last` for both, then `compare_runs.py --match E12_multidata_yolo26s@last E14_hardneg_yolo26s_L4@last`; local GPU, ~25 min; `eval_*_last.txt`, `cmp_E12_E14L4_last.txt`): val-chosen thresholds 0.55 (E12) and 0.40 (E14). At matched validation recall (E14 at 0.37): Anti-UAV test background FAs −35.8 per 1000 frames [−64.1, −15.5], recall −0.017 [−0.046, +0.010]; Halmstad val recall +0.226 [+0.133, +0.335]. The same two training runs as #50, so no new seed: the hard-negative verdict stays "not established" (19 §3.1).
* **Unlabelled objects in the training tiles** (scratch script, E12 `best.pt` over 6,694 Halmstad and 3,578 Anti-UAV training tiles, ~6 min; contact sheets `C:\cuas_data\runs\unlabelled_halmstad.png`, `unlabelled_antiuav.png`): confident (≥ 0.4) "drone" boxes touching no label: 13 (Halmstad, 6 videos) and 12 (Anti-UAV, 11 sequences); mostly specks and night lights. A lower bound (the model trained on these tiles). Too few to act on.
* New experiments: E28 (done), E30 (per-size score calibration), centre-distance matching under 12 px. E29 (reviewed pseudo-labels) was drafted and is not scheduled.

**#60 Colab L4: the three E18 ablations finished (output received 2026-10-02).** The user's cell was the old queue (pasted before #57's reorder), so it ran `E18_attn_s0`, `E18_aelan_s0` and `E18_eca_s0` with patience 15, and did not run `E18_base_s2` or `E18_base_s1_e80`. E13's lines in this output say only "done"; its files are on Drive.
* **`E18_attn_s0`** (mpda+rmpda, seed 0): 708/848 items transferred; 5.4 it/s; **early stop at epoch 29** (best 14); 1.61 h. Threshold on val 0.40.
  * Pooled: Anti-UAV val 0.942 / 0.807, `w_min` 4 px; test 0.818 / 0.805, `w_min` 8 px; Halmstad val 0.894 / 0.966.
  * Native: test recall 0.846 [0.800, 0.889], precision 0.714, **97.2 background FAs per 1000** [66.0, 130.7]; Halmstad 0.930.
  * Against E14 at matched recall (0.65): val precision +0.028 [+0.015, +0.041]; test recall −0.011 and background FAs −2.9 (neither clear; 24 boxes against 30); **Halmstad recall −0.121 [−0.226, −0.040]**.
  * DUT at 0.40: recall 0.108 [0.095, 0.121], precision 0.568, 132 background false drones; 761 drones boxed as another class.
* **`E18_aelan_s0`** (aelan, seed 0): 708/722 items; 5.2 it/s; **early stop at epoch 39** (best 24); 2.27 h. Threshold on val 0.40.
  * Pooled: Anti-UAV val 0.949 / 0.799, `w_min` 4 px; test 0.816 / 0.795, `w_min` 12 px; Halmstad val 0.909 / 0.979.
  * Native: test recall 0.848 [0.803, 0.891], precision 0.770, 21.6 background FAs per 1000 [12.3, 31.9]; Halmstad 0.900.
  * Against E14 at matched recall (0.65): val background FAs −8.8 per 1000 [−19.8, −0.7] (2 boxes against 15); test background FAs −9.8 [−19.2, −1.5] (10 against 30), test recall +0.002; Halmstad −0.009 (not clear).
  * DUT at 0.40: recall 0.174 [0.160, 0.190], precision 0.523, 271 background false drones; sky 0.22 as drone / 0.37 as another class / 0.42 missed.
* **`E18_eca_s0`** (eca, the cheap control, seed 0): 708/712 items; 6.1 it/s; all 40 epochs; 1.99 h. Threshold on val 0.55.
  * Pooled: Anti-UAV val 0.931 / 0.826, `w_min` 4 px; test 0.800 / 0.861, `w_min` 12 px; Halmstad val 0.705 / 0.998.
  * Native: test recall 0.849 [0.802, 0.894], precision 0.792, 26.0 background FAs per 1000 [14.4, 39.3]; Halmstad 0.575 [0.400, 0.735].
  * Against E14 at matched recall (0.68): **test recall +0.032 [+0.002, +0.060]**, test background FAs −7.9 [−16.1, 0.0] (14 against 30); val precision +0.023; **Halmstad recall +0.081 [+0.010, +0.161]**.
  * DUT at 0.55: recall 0.089 [0.078, 0.101], precision 0.542, 117 background false drones.
* **Reading (19 §3.6):** the near-null control is called "better" than plain seed 0 as much as any add-on, and is level with plain seed 1. Plain seed 0 is the weakest of the six runs, so it is the wrong reference. Against plain seed 1 no add-on wins on any held-out measure; attention only is worse (97 background FAs per 1000 at test recall ~0.85 against 13-34 for the rest). AELAN's DUT 0.174 is at a lower threshold with three times the false drones of plain seed 1. **E18 closed.**
* **Still to run:** the new `colab_e18.py` (upload the rebuilt `cuas_code.zip` first). It skips everything finished, scores each finished run's final weights (~20 min each), then trains `E18_base_s2` and `E18_base_s1_e80`. **Still to download from Drive into `C:\cuas_data\runs\`:** `eval_*.txt`, `cmp_*.txt`, `e15_*.txt/.json` and the run folders, E13's first.
* **Code:** comments in `colab_e18.py` updated (the ablations are done); code zip rebuilt.

**#61 E13 (drone-only training): in-domain results received (2026-10-02).** Files copied to `C:\cuas_data\runs\`: `eval_E13_droneonly_s0.txt`, `cmp_E13_droneonly_s0.txt`, `E13_droneonly_s0_train.log` (the last segment only: resumed at epoch 33, ran to 40; validation mAP50 0.971, drone class only). `e15_E13_droneonly_s0.txt` (DUT) arrived the same day and is copied there too.
* Threshold chosen on val: 0.40. Pooled: Anti-UAV val 0.950 / 0.826, `w_min` 4 px; test 0.803 / 0.842, `w_min` 12 px; Halmstad val 0.902 / 0.981, `w_min` 8 px.
* Native, 95% per-video intervals at 0.40: Anti-UAV test recall 0.881 [0.843, 0.917], precision 0.749, 26.5 background FAs per 1000 frames [16.2, 38.4]; Halmstad val recall 0.930 [0.847, 0.984], precision 0.995.
* Native at 0.55: test 0.852 / 0.792 / 12.3; Halmstad 0.595. Plain seed 1 at 0.55: 0.843 / 0.792 / 12.8; Halmstad 0.599.
* Against E14 at matched recall (E13 at 0.71): val background FAs −8.8 per 1000 [−19.8, −0.7] (2 boxes against 15); test background FAs −13.7 [−23.7, −5.3] (2 against 30), test recall +0.015 (not clear); Halmstad recall −0.101 [−0.205, −0.026]. Plain seed 1 against E14 gave 3 test boxes and recall 0.779.
* No false "drone" box on a labelled bird, airplane or helicopter in Halmstad val at 0.40 (9 on background in 2,550 images).
* **DUT (E15) at 0.40:** recall **0.041** [0.033, 0.049], precision 0.442, 99 background false drones over 2,200 images (45 per 1000). By width: 0.00 under 24 px, 0.03 at 24-32, 0.06 at 32-64, 0.03 above 64. Sky backgrounds: 0.03 as drone, **0.97 missed**; clutter: 0.05 / 0.95.
* **Reading:** in-domain, the drone-only model is indistinguishable from plain seed 1; dropping the three other classes cost nothing there and did not turn birds or aircraft into drones. **On DUT it is the worst run of all** (0.041 against 0.076-0.174). The four-class models box 35-52% of DUT sky drones as helicopter or airplane; the drone-only model does not box them at all. So the hypothesis behind E13 is refuted: those drones were not "seen and misnamed" by a drone detector, they were found by the *other* class heads, and removing those heads removes the detection. **Keep four classes.** The useful signal on unseen drones is "some airborne object" (any-class recall 0.49-0.66 on sky), which the tracker already takes as a candidate; the drone class itself needs more varied drone data (19 §3.7).

**#62 E22 built and smoke-tested; E31 (seed soup) built; Colab queue rewritten (local, 2026-10-02, evening).**
* **Code:** `cuas/degrade.py` (`Degrade`, `install`); `train_local.py --degrade / --mosaic / --scale` (a `cuas_degrade.txt` marker in the run folder re-enables degradation on `--resume`); `e21_degradation.py` takes `<run>@last` and adds four held-out conditions (motion blur 5 px, signal-dependent noise gain 0.4, WebP quality 30, all three); `experiments/soup.py`; test `test_training_degradation` (49 tests).
* **Smoke run** (RTX 4060): `train_local.py --data C:/cuas_data/e14/data_e14.yaml --name E22_smoke --epochs 1 --fraction 0.02 --degrade --mosaic 0.1 --scale 0.3 --patience 100`. Trained (0.039 h); the log prints the `Degrade` settings; `train_batch0.jpg` shows blurred and JPEG-blocked tiles; AdamW lr 0.00125 as pinned. `--resume` on that run printed the `Degrade` line again (marker read).
* **Why no library:** Ultralytics accepts custom Albumentations transforms, but Albumentations is not installed here and its noise arguments differ between versions. The transform is 60 lines of numpy and OpenCV.
* **Soup, local:** `soup.py E31_soup_E12_E14 E12_multidata_yolo26s E14_hardneg_yolo26s_L4`, then `e12_eval.py E31_soup_E12_E14@last` and `compare_runs.py --match/--sweep E14_hardneg_yolo26s_L4@last E31_soup_E12_E14@last` (the first evaluation attempt was stopped at the 30-minute job limit while the GPU was shared; the second took 17 min). Result: the final weights of E12 and E14 (two different recipes from the same pretrained weights; the only finished final weights on this machine) were averaged and scored (`eval_E31_soup_E12_E14_last.txt`). The average is a working detector (native Anti-UAV test recall 0.807, precision 0.830 at 0.40; it did not collapse), but **it is worse than either member at matched recall**: at test recall ~0.87 it has 81 background false alarms per 1000 frames against 23 for E14's final weights (+58 [+25, +98]) and 62 for E12's at recall 0.89. Its scores are also lower (Halmstad recall 0.64 at 0.40 against 0.94 and 0.97). So averaging is not free here. This pair differs in training data and in stopping epoch (36 and 28), so it is not the seed test; that is `E31_soup_base_s1s2` on Colab, and the other soups stay switched off until it reports.
* **`colab_e18.py` rewritten:** `E18_base_s1` (scoring only), `E18_base_s2`, `E22a_s1`, `E22a_s2`, `E18_base_s0`, `E22a_s0`, `E22b_s1`, `E18_base_s1_e80`, `E22b_s2`, `E22b_s0`; one soup of the two plain seeds as soon as both exist (the three-seed and E22 soups are listed but switched off until that one reports). Final weights only; each run compared at matched recall with the plain run of its own seed; E15 and E21 after each. A 2-epoch `E22_smoke` runs first on Colab. About 25 GPU-hours in all, in priority order. Code zip rebuilt.
* **GPU memory (asked 2026-10-02):** training uses ~7 GB of the L4's 23 GB at batch 16. The batch stays 16: it is part of the recipe `E18_base_s1` was trained with, and every run in this queue is compared with that run. The queue now passes `--workers` (up to 8, capped by the runtime's cores), since data loading is the likely limit for a model this size and `--degrade` adds CPU work per image; it prints the core count. Whether the GPU or the loader is the limit has **not been measured**.
* **Reference added:** model soups (arXiv 2203.05482, checked at the abstract page); 78 verified papers.
* **Not done:** E22c (pasted degraded drone crops on varied backgrounds) and E32 (PX4 paths with pasted real crops) are planned, not built.

**#63 Colab L4: the rewritten queue started (2026-10-02, night; first output).** 12 CPU cores, so 8 loader workers.
* **`E22_smoke`** (2 epochs, 2% of the data, `--degrade --mosaic 0.1 --scale 0.3`): trained on Colab; the `Degrade` line printed. 0.015 h.
* **`E18_base_s1@last`** (plain recipe, seed 1, final weights; the reference for this queue). Threshold on val 0.55.
  * Pooled: Anti-UAV val 0.922 / 0.817, `w_min` 8 px; test 0.806 / 0.869, `w_min` 12 px; Halmstad val 0.756 / 0.996.
  * Native: test recall 0.859 [0.815, 0.900], precision 0.784, 17.2 background FAs per 1000 frames [7.7, 28.0]; Halmstad val recall 0.716 [0.579, 0.840], precision 1.000.
  * Against its own `best.pt` at 0.55 (test 0.843 / 0.792 / 12.8; Halmstad 0.599): close on the test, higher on Halmstad.
  * DUT: recall 0.138 [0.123, 0.152], precision 0.695, 81 background false drones (`best.pt`: 0.149, 92). Sky: 0.18 as drone / 0.34 as another class / 0.47 missed.
* E21 on those weights was running when the output was sent.

**#64 Colab L4: plain seed 2, the E21 baselines, and the seed soup (output received 2026-10-03).**
* **E21 on `E18_base_s1@last`** (0.55): Halmstad native 0.716 clean, 0.232 under blur 1.0 + noise 6 + JPEG 40, 0.115 under the held-out combination; ~8.5 px: 0.816 / 0.150 / 0.019; Anti-UAV test ~14 px: 0.813 / 0.465 / 0.511. Single held-out kinds at ~8.5 px: motion blur 0.657, shot noise 0.291, WebP 0.497.
* **`E18_base_s2`** (plain recipe, seed 2, `--patience 100`, 8 loader workers): 40 epochs in 1.778 h, 6.7-7.0 it/s, 6.1 GB (the earlier plain run took about 0.051 h per epoch with 4 workers; this one 0.044). Final validation mAP50 0.863.
  * Threshold on val 0.55. Pooled: Anti-UAV val 0.935 / 0.810, `w_min` 4 px; test 0.812 / 0.851, `w_min` 12 px; Halmstad val 0.767 / 0.996.
  * Native: test recall 0.843 [0.793, 0.892], precision 0.780, 15.2 background FAs per 1000 [8.1, 23.1]; Halmstad 0.772 [0.637, 0.890].
  * Against seed 1 at matched recall (seed 2 at 0.46): test recall −0.003 and background FAs +6.9 (neither clear), test precision −0.034 [−0.047, −0.022]; Halmstad recall +0.189 [+0.100, +0.290].
  * DUT: recall 0.157 [0.142, 0.173], precision 0.687, 90 background false drones; sky 0.22 / 0.31 / 0.47.
  * E21 (0.55): Halmstad native 0.772 / 0.425 / 0.215; ~8.5 px 0.804 / 0.111 / 0.006; Anti-UAV ~14 px 0.826 / 0.465 / 0.532.
* **`E31_soup_base_s1s2`** (`soup.py`; largest single-weight difference between the members 30.7): **drone recall 0.000 at every threshold on every set**; 37-282 drones boxed as airplane; DUT 0.000. Cause, read from the installed Ultralytics source (`_remap_cls_by_names`) and the training logs ("Remapped 2/4 cls head rows"): the drone and helicopter rows of the class layer start from a random value that depends on the seed, so the two members did not share a start for those rows. The test is invalid as run; E31 is shelved (19 §3.8).
* **Reading (19 §3.1):** under the fixed protocol the two seeds agree within 0.02 in test recall at every threshold. The earlier seed spread was mostly early stopping and `best.pt`.
* **Correction:** 19 §4.1 and §3.7 said three class rows, including helicopter, start from COCO. It is two: bird and airplane.
* `E22a_s1` had started (the `Degrade` line printed) when the output was sent.

**#65 Colab L4: `E22a_s1` (first E22 run) and `E22a_s2` trained (output received 2026-10-03).** The cell had been stopped by the user during `E22a_s1` and re-run.
* **`E22a_s1`** (`--degrade`, seed 1): resumed from epoch 24 to 40 (the `Degrade` line printed on resume, so the marker file worked); 6.7-6.9 it/s, the same as a plain run; final validation mAP50 0.870.
  * Threshold on val: **0.40** (plain seed 1: 0.55). Pooled: Anti-UAV val 0.949 / 0.795, `w_min` 4 px; test 0.827 / 0.790, `w_min` 12 px; Halmstad val 0.868 / 0.976, `w_min` 8 px.
  * Native at 0.40: test recall 0.899 [0.867, 0.931], precision 0.749, 21.6 background FAs per 1000 [12.7, 31.7]; Halmstad 0.895 [0.818, 0.962].
  * Against `E18_base_s1@last` at matched validation recall (both at 0.55): test recall **+0.023 [+0.009, +0.039]**, test background FAs −6.4 (not clear; 22 boxes against 35); val background FAs −14.9 (not clear); **Halmstad recall −0.159 [−0.246, −0.078]**.
  * DUT at 0.40: recall 0.103 [0.091, 0.117], precision 0.406, **246 background false drones** (114 of them under 16 px); sky 0.13 / 0.42 / 0.45.
  * E21 at 0.40: Halmstad native 0.895 clean, 0.811 trained-kind combination, 0.558 held-out combination; ~8.5 px 0.866 / 0.313 / 0.147; Anti-UAV ~14 px 0.825 / 0.771 / 0.711. Noise σ 16 at native size: 0.848 (plain: 0.125).
* **`E22a_s2`** (seed 2): 40 epochs in 1.793 h; final validation mAP50 0.872. Its scoring had just started.
* **Reading (15 §6.14), one seed:** large gains under degradation at ~14 px and ~34 px, **including the held-out kinds**; at ~8.5 px recall under the combined degradations doubles but stays low (0.31, and 0.15 held out). Clean Anti-UAV test is not harmed. Costs: lower Halmstad recall at a given threshold, and on DUT fewer drones with three times the false drones, many of them tiny. The E21 columns are at different thresholds (0.40 against 0.55), so part of the E21 gain is the threshold; the false-alarm column says most of it is not (at ~8.5 px, 174 against 193 per 1000 images at twice the recall).

**#66 Local: run folders downloaded; first commit of the review work; three CPU checks (2026-10-03).**
* **Downloaded** (user): `E18_base_s1`, `E18_base_s2`, `E22a_s1`, `E22a_s2` (weights, `dets_native_last.json`, `e12_eval_last.json`) into `C:\cuas_data\runs\`. The loose `eval_/cmp_/e15_/e21_*_last` text files were not in the download; the comparisons can be rebuilt from the cached detections, the DUT and E21 output for `E22a_s2` cannot without a GPU.
* **Commit `2de9134`** on `main` (not pushed): the design-review fixes, the program, doc 19, `cuas/datasets/`, E20-E22, E28, E31.
* **`E22a_s2@last` against `E18_base_s2@last`** (`compare_runs.py --match/--sweep`, CPU, cached detections; both at 0.55): Anti-UAV test recall +0.017 [+0.002, +0.032], precision +0.024 [+0.013, +0.035], background FAs −2.5 (not clear); Anti-UAV val background FAs +18.2 (not clear; 32 boxes against 5); Halmstad recall −0.260 [−0.384, −0.150] at 0.55 (0.937 against 0.950 at 0.40). Threshold chosen on val 0.40.
* **E30** (`e30_size_calibration.py`, CPU, ~1 min): refuted. Median score of correct 32-64 px drone boxes: Anti-UAV 0.76-0.80, Halmstad 0.54-0.63. Per-size isotonic calibration fitted on Anti-UAV val: Halmstad recall at the val-chosen threshold 0.11-0.18 against 0.15-0.26 raw. Halmstad's own best threshold 0.32-0.34: recall 0.987-0.994.
* **E33** (`e33_center_match.py`, CPU inference on the five smallest sets, cached as `dets_small_last.json`, ~4 min a run): centre matching under 12 px raises recall at 0-4 px by 0.12-0.22 and by at most 0.01 above; `w_min` 12 / 12→8 / 8 / 8 px. Code: `cuas.eval.match_quality`, `evaluate(center_below=...)`, test `test_centre_matching_for_tiny_targets` (50 tests).
* **E34** (`e34_any_class.py`; DUT by CPU inference, cached as `dets_dut_last.json`): on the unseen dataset (DUT) boxes of any class reach 2.2-2.6 times as many drones as "drone" boxes (0.353 against 0.138 at 0.55) for 4 times the stray candidates, and still reach more at an equal stray-candidate rate; on the trained datasets it gains no drone and costs 2.5-5 times the candidates. The pipeline already tracks every class; keep it, with a strict track decision.
* **`E22a_s2@last`, DUT and E21** (two files sent by the user, copied to `C:\cuas_data\runs\`): DUT at 0.40 recall 0.158 [0.143, 0.174], precision 0.537, 211 background false drones; sky 0.19 / 0.36 / 0.45. E21 at 0.40: Halmstad native 0.937 clean, 0.855 trained-kind combination, 0.607 held-out combination; ~8.5 px 0.869 / 0.320 / 0.071; Anti-UAV ~14 px 0.808 / 0.755 / 0.677.
* **DUT at matched thresholds, seed 1** (from `dets_dut_last.json`): E22a drone recall 0.068 / 0.103 / 0.139 at 0.55 / 0.40 / 0.25 against 0.138 / 0.189 / 0.223 for the plain run; background false drones per 1000 images 30 / 113 / 337 against 37 / 73 / 146.
* The local GPU was in use by another program (6.5 GB of 8 held), so everything here ran on the CPU.

**#67 Colab L4: third seeds and the first E22b run (files received 2026-10-04).** The queue had finished `E18_base_s0`, `E22a_s0` and `E22b_s1`; `E18_base_s1_e80` was at epoch 51 of 80.
* **Files:** run folders `E18_base_s0`, `E22a_s0`, `E22b_s1` (weights, `dets_native_last.json`, `e12_eval_last.json`) moved into `C:\cuas_data\runs\`. They had been extracted into the project's `runs/` folder, which also held byte-identical copies of the four folders from #66; those copies were removed (data stays out of the project folder). DUT and E21 text files for `E18_base_s0` and `E22a_s0`, DUT for `E22b_s1`. **`e21_E22b_s1_last_log.txt` is still missing** (a second copy of `E22a_s1`'s was sent in its place).
* **`E18_base_s0@last`** (plain, seed 0, 40 epochs): threshold 0.55. Pooled Anti-UAV test 0.803 / 0.853, `w_min` 12 px. Native test at 0.55: 0.834 / 0.782 / 11.8. Halmstad 0.725 at 0.55, 0.943 at 0.40. DUT 0.118 [0.105, 0.132], precision 0.697, 66 background false drones. E21: Halmstad native 0.725 / 0.447 / 0.135 (clean / trained kinds / held out); ~8.5 px 0.827 / 0.099 / 0.010; ~14 px 0.839 / 0.434 / 0.556.
* **`E22a_s0@last`**: threshold 0.40. Pooled test 0.803 / 0.818, `w_min` 12 px. Native test at 0.55: 0.861 / 0.802 / 14.2. Halmstad 0.569 at 0.55, 0.945 at 0.40. DUT at 0.40: 0.117 [0.104, 0.132], precision 0.594, 89 background false drones. E21 at 0.40: Halmstad native 0.945 / 0.879 / 0.351; ~8.5 px 0.871 / 0.316 / 0.084; ~14 px 0.785 / 0.731 / 0.665.
  * Against `E18_base_s0@last` at matched validation recall (both 0.55; `compare_runs.py`, local CPU): test recall +0.018 [+0.007, +0.032], precision +0.029 [+0.021, +0.037], background FAs +1.0 (not clear); Halmstad recall −0.233 [−0.360, −0.123].
* **`E22b_s1@last`** (`--degrade --mosaic 0.1 --scale 0.3`): threshold 0.40. Pooled test 0.820 / 0.818, `w_min` 12 px. Native test at 0.55: 0.884 / 0.767 / 25.5. Halmstad 0.667 at 0.55, 0.875 at 0.40. DUT at 0.40: 0.170 [0.155, 0.186], precision 0.511, 233 background false drones.
  * Against `E18_base_s1@last` at matched validation recall: test recall +0.026 [+0.010, +0.043], precision −0.020 [−0.031, −0.009], background FAs +9.3 [+1.9, +17.3]. Against `E22a_s1@last` in the sweep: the same recall at each threshold with 2.4 times the test false alarms and 4.6 times the val false alarms.
* **Reading (15 §6.14):** E22a's gain holds in three seeds of three and is many times the seed spread; adopted. E22b so far adds false alarms in-domain.

**#68 Evaluation-methods review; E35 (local CPU, 2026-10-04).**
* **Intention.** The user asked whether the validation scheme and the scores (MAE, RMSE, recall, F1 and others) are the right ones. The useful answer needed the scores we had *not* been computing, so they were computed. A result that would change what we do: any conclusion that flips when the threshold is removed.
* **What was done.** `experiments/e35_metrics.py` (new, no inference; ~1 min for 7 runs on cached `dets_native_last.json` and `dets_small_last.json`): drone AP, recall at 10 and 30 background false alarms per 1000 frames, ECE and Brier score of the raw score, and width / centre error of correct boxes by size. Runs: `E18_base_s1/s2/s0`, `E22a_s1/s2/s0`, `E22b_s1`, all `@last`. `e21_E22b_s1_last_log.txt` received and filed.
* **Measurements** (`C:\cuas_data\runs\e35_metrics.txt`): Anti-UAV test AP plain 0.856 / 0.858 / 0.845, E22a 0.894 / 0.882 / 0.891, E22b 0.899; recall at 10 per 1000: 0.836 / 0.828 / 0.830, 0.880 / 0.850 / 0.839, 0.844. Halmstad val AP 0.993-0.997 for all seven. ECE 0.09-0.13 (Anti-UAV), 0.25-0.28 (Halmstad). Width error, plain seed 1: +25% bias and 1.5 px MAE under 8 px; +5% and 1.6 px at 8-16 px; MAE 11-14% above. Centre error 0.6 px under 8 px. E22b seed 1 E21 at 0.40: ~8.5 px 0.854 / 0.310 / 0.131.
* **What worked.** Everything needed was already cached, so seven runs were rescored in a minute. The threshold-free scores agree with the adopted conclusion (E22a is better on clean data too) and sharpen it.
* **What did not.** The box-error table mixes sets: the three runs without cached small-target detections (`E18_base_s0`, `E22a_s0`, `E22b_s1`) have no rows under 16 px and a 16-32 px row made of native Halmstad frames only, so their rows are not comparable with the other four. Reported for the four complete runs. Two doc-update scripts also stopped halfway on text that did not match the file (a bold marker that was not there; a shell heredoc that turned an escaped newline into a real one), so each had to be finished by a second script; the same heredoc slip had already happened on 2026-10-01 (#56). Scripts that edit docs are now written to a file with the editor, never through a shell heredoc.
* **Mistakes found.** (1) Thresholds chosen by maximum F1 had confounded E22a's E21 and DUT columns (0.40 against 0.55); the fix was a one-minute computation that should have been in the protocol from E14. (2) "Halmstad recall fell" was written up twice as a cost of E22a; AP shows no loss. (3) Box error had never been measured although range and time to collision depend on it. (4) The test sets have been scored after every run, against the project's own "score it once" principle.
* **Lesson and decision (19 §11.4).** Headline scores become AP and recall at a fixed false-alarm budget; F1 only picks the operating point for the per-width table; one dataset stays unscored until a recipe is final; Halmstad native frames are a sanity check only; width bias is quoted with any range figure.

**#69 E22c built: real drones cut out and pasted across datasets; E35 added to the Colab queue (local CPU, 2026-10-04).**
* **Intention.** Nothing so far has moved the unseen dataset: DUT recall is 0.12-0.16 for the plain recipe, the add-ons, the drone-only model and degraded training alike. One explanation that can be tested with the data we have: the drone class is tied to the scenes it was filmed in (a Halmstad drone only ever appears in Halmstad sky, an Anti-UAV drone in an Anti-UAV scene). E22c breaks that tie with real pixels: each dataset's drones are pasted into the other's training images. **A result that would change what we do:** DUT AP and recall at a fixed false-alarm budget clearly above E22a's three seeds. **What it cannot show:** it adds no new drone models, so if DUT fails because its drones look different, this will not help, and the answer is then more varied drone data (the DUT train split, Drone-vs-Bird).
* **What was done.**
  * `cuas/paste.py`: `cut(frame, box)` makes a soft matte of a drone filmed against something plain; `paste(tile, patch, x, y, scale)` composites it, un-mixing edge pixels from the old background. Test `test_cut_and_paste_a_drone` (51 tests).
  * `experiments/build_e22c.py` (CPU, ~3 min): Anti-UAV drones into Halmstad frames (all four classes' videos), Halmstad drones into Anti-UAV tiles and the mined clutter tiles; 1,500 tiles each way; random position, random size 0.35-1.0 of the original, never enlarged; at most 12 patches per source video. Output `C:\cuas_data\e22c\paste_tiles` (3,000 tiles, 105 MB), `data_e22c.yaml` (E14's training list plus the tiles), `sheet.png`.
  * `pack_for_colab.py --e22c` → `C:\cuas_data\colab\cuas_data_e22c.zip` (0.099 GB). `colab_e18.py`: unzips it if present; a 2-epoch `E22c_smoke`; runs `E22c_s1/s2/s0` (`--degrade` + the E22c data) placed after the 80-epoch run and before the two remaining E22b seeds; E22b and E22c runs are compared with E22a of the same seed; **E35 runs after every run** (`e35_<run>.txt`).
* **Measurements.** Mattes: 1,469 of 2,666 Anti-UAV drone boxes (≥ 14 px) from 143 sequences, 834 of 1,770 Halmstad boxes from 83 videos; the rest had clutter behind them or did not stand out. 604 Anti-UAV patches were filmed at night and left out of daylight scenes. Pasted drone width: 16 / 39 / 133 px at the 10th / 50th / 90th percentile.
* **What worked.** The matte and paste functions passed their test at the first run. The un-mixing removes the halo of old sky around a pasted drone.
* **What did not, and how it was noticed** (each by looking at the contact sheet, not from any number):
  1. **First build: night drones in daylight.** Anti-UAV's night sequences show a drone as coloured lights on black. Pasted into a Halmstad day sky they became glowing blobs with a black fringe, about a quarter of the Anti-UAV pastes. Fix: `paste` refuses a spot whose brightness differs from the drone's old background by more than 60 grey levels.
  2. **First build: pale halos.** Plain alpha compositing carried a rim of the old, brighter sky. Fix: un-mix the edge pixels (`out = patch + (1 − alpha) × (new − old background)`).
  3. **Second build: two night patches still got through**, pasted over dark forest and field, which pass the brightness rule. Fix: the builder drops patches whose old background is darker than 60 before pasting into Halmstad.
  4. A test assertion failed after fix 2: matte values of 0.001 far from the drone were adding the source's noise to the new background. Fix: ignore matte values under 0.02.
* **Known limits of this first version.** Halmstad drones keep a bluish tint from their sky when pasted on Anti-UAV's grey scenes, which could be learned as a cue (the saturation jitter and `--degrade` weaken it). The brightness rule keeps most pastes over sky-like backgrounds, so "drone over dark terrain" is rare. The backgrounds are still our two datasets; own target-free footage is the real source of new terrain (19 §3.9). Evaluation is on real footage only, so a learned paste artefact can cost training time but cannot inflate a result.
* **Not run locally:** a training smoke test with the E22c data (the local GPU is held by another program). The data file was checked instead (12 training folders exist, 3,000 label files), and the Colab queue runs a 2-epoch smoke first.
* **Lesson and decision.** Look at generated training data before training on it: three of the four faults were visible in seconds and none would have shown in a loss curve. Upload `cuas_data_e22c.zip` and the new `cuas_code.zip`, re-paste `colab_e18.py`.

**#70 Colab L4: the old queue finished: `E18_base_s1_e80`, `E22b_s2`, `E22b_s0` (output received 2026-10-04, late).**
* **Intention.** Two questions left in the queue. (1) Is 40 epochs enough? Decides the schedule of every later run. A result that would change it: clearly better held-out numbers at 80. (2) Do the challenge winner's augmentation settings (mosaic 0.1, scale ± 0.3) add to degraded training? Three seeds, because one seed had shown a possible DUT gain.
* **What was done.** The user's cell was still the queue from #62, so it ran these three and stopped ("ALL DONE"); the E22c runs and E35 added in #69 need the new zip and file. `E18_base_s1_e80`: resumed at epoch 51, 80 epochs in all (30 epochs in 1.329 h; 6.7-7.0 it/s). `E22b_s2`, `E22b_s0`: 40 epochs each in 1.79 h.
* **Measurements.**
  * `E18_base_s1_e80@last`: threshold 0.55. Pooled Anti-UAV test 0.808 / 0.841, `w_min` 12 px. Native test 0.864 [0.821, 0.904] / 0.793 / 7.9 background FAs per 1000 [2.9, 14.0]. Halmstad 0.787 at 0.55, 0.982 at 0.40. Against the 40-epoch run at matched validation recall (0.60): test background FAs −11.8 [−21.3, −3.0], precision +0.025 [+0.014, +0.036], recall −0.003; val background FAs −16.2 [−40.3, −2.1]. DUT 0.140 [0.126, 0.155], precision 0.744, 54 background false drones. E21: ~8.5 px 0.836 / 0.137 / 0.016 (clean / trained kinds / held out); Halmstad native 0.787 / 0.289 / 0.087, noise σ 4 and 8: 0.366, 0.166.
  * `E22b_s2@last`: threshold **0.55**. Native test 0.866 [0.821, 0.907] / 0.765 / 24.0. Halmstad 0.679 at 0.55, 0.828 at 0.40. Against plain seed 2 at matched recall: test recall +0.018 [+0.005, +0.033], background FAs +6.9 (not clear), Halmstad recall −0.105 [−0.204, −0.023]. DUT at 0.55: 0.124 [0.111, 0.139], precision 0.655, 91 background false drones. E21 at 0.55: ~8.5 px 0.639 / 0.231 / 0.080.
  * `E22b_s0@last`: threshold 0.40. Native test at 0.40: 0.898 [0.864, 0.930] / 0.722 / 53.0; at 0.55: 0.868 / 0.774 / 25.0. Halmstad 0.898 at 0.40. Against plain seed 0 at matched recall: test recall +0.036 [+0.023, +0.052], precision −0.012 [−0.024, −0.001], background FAs +14.2 [+7.3, +22.0]. DUT at 0.40: 0.128 [0.114, 0.142], precision 0.583, 120 background false drones. E21 at 0.40: ~8.5 px 0.863 / 0.324 / 0.143; Halmstad native 0.898 / 0.842 / 0.677.
* **What worked.** Both questions are answered. The 80-epoch run gives a clean result on the thing it changes (false alarms on trained datasets).
* **What did not.** Neither change touches DUT or robustness. E22b's first-seed DUT figure (0.170) did not repeat (0.124, 0.128). E22b seed 2 chose threshold 0.55 while the other two chose 0.40, so its E21 table cannot be set beside theirs: the threshold confound of 19 §11 again, in runs started before E35 existed.
* **Mistakes.** The E22b comparison in this queue was made against the plain run of each seed, not against E22a of the same seed, which is the comparison that isolates mosaic and scale (fixed in the queue file in #69, after these runs had started). The numbers above were therefore read from the sweeps against E22a's logged values.
* **Lesson and decision.** 40 epochs stays for experiments (equal schedules, half the cost); 80 is an option for a final model. E22b is dropped. Next on Colab: upload `cuas_data_e22c.zip` and the new `cuas_code.zip`, re-paste `colab_e18.py` (E22c seeds; E35 for every finished run). To compute AP for these three runs locally their folders are needed (`E18_base_s1_e80`, `E22b_s2`, `E22b_s0`).

**#71 A demo video of the current state (local CPU, 2026-10-04).**
* **Intention.** The user asked for a video that shows the system working, to show progress. It had to be honest: footage the detector never trained on, the program's own overlay, and a statement of what it is not.
* **What was done.** `experiments/make_demo.py` (venv Python, CPU because the GPU was held by another program; about 6 minutes). Part 1 runs the whole program (`cuas.replay.run`: motion cueing, detector, tracker, track decision) on three held-out Halmstad validation videos, 9 s each, with `E22a_s1` final weights and the track classifier `track_clf_dets_E12.pkl`. Part 2 puts the drone video through blur 1.0 + noise 6 + JPEG 40 (E21's combined condition) and shows the plain model (`E18_base_s1`) beside `E22a_s1`, both at threshold 0.40, with a running count of frames in which each drew a "drone" box. Output: `C:\cuas_data\demo\progress_demo.mp4` (46 s, 1280 × 556) and `progress_demo_stills.png`.
* **Measurements.** `V_DRONE_037`: one track, declared a drone 1.1 s after it appears. `V_BIRD_034`: one track, labelled not-drone, no declaration. `V_AIRPLANE_017`: two tracks (one not-drone, one unknown), no declaration. Degraded drone video: the plain model boxed the drone in 144 of 270 frames, E22a in 255 of 270. Detector time on the CPU: 0.2-0.3 s per frame.
* **What worked.** The program ran end to end on all three with no change to `cuas/`.
* **What did not.** No H.264 encoder on this machine (no ffmpeg), so the file is MPEG-4 Part 2 (`mp4v`); it plays in VLC and most desktop players but may not play inside a browser. The script uses ffmpeg when it is installed.
* **What it is not.** Three videos from one dataset the model also trained on (different videos): it shows the pipeline and the E22a effect, not generalisation. The frame counts in Part 2 are "a drone box was drawn", not checked against labels frame by frame; the boxes seen in the stills are on the drone. It runs slower than real time here (CPU).

**#72 Colab L4: E22c trained (three seeds); first scored seed `E22c_s0`; E35 for the last two E22b seeds (output received 2026-10-05).**
* **Intention.** E22c (#69) asks whether the drone class is tied to the scenes it was filmed in. A result that would change what we do: DUT AP and recall at a fixed false-alarm budget clearly above E22a of the same seeds. A null would point to new drone data (DUT train split, Drone-vs-Bird) rather than new backgrounds.
* **What was done.** The user uploaded `cuas_data_e22c.zip` and the rebuilt code and re-ran `colab_e18.py`. `E22c_s1` and `E22c_s2` had trained and been scored in earlier sessions of the cell (their output was not in this paste); this session resumed `E22c_s0` at epoch 24 (17 epochs in 0.907 h, 6.5-6.8 it/s; 18,480 training images, 830 of them backgrounds), then scored it, then ran E35 for `E22b_s2` and `E22b_s0`. All at `@last`, threshold chosen on Anti-UAV val 0.40 (as for `E22a_s0`, so the per-threshold columns below are comparable).
* **Measurements, `E22c_s0` against `E22a_s0`** (both 0.40; source: this cell's output and `C:\cuas_data\runs\E22a_s0\reports\e15_E22a_s0_last.txt`, `e21_E22a_s0_last_log.txt`, `e35_metrics.json`):

  | | E22a_s0 | E22c_s0 |
  | :--- | ---: | ---: |
  | Anti-UAV test AP / R@10 / R@30 | 0.891 / 0.839 / 0.887 | 0.886 / 0.845 / 0.874 |
  | Anti-UAV test native at 0.40: recall / precision / background FAs per 1000 | 0.887 / 0.755 / 29.9 | 0.889 / 0.716 / 76.1 |
  | The same at 0.55 | 0.861 / 0.802 / 14.2 | 0.859 / 0.781 / 18.2 |
  | Halmstad val AP | 0.995 | 0.994 |
  | **DUT at 0.40: drone recall / precision / background false drones per 1000 images** | 0.117 / 0.594 / 40 | **0.219** [0.202, 0.236] / 0.574 / 125 |
  | DUT sky drones: as drone / as another class / missed | 0.14 / 0.41 / 0.45 | 0.26 / 0.33 / 0.41 |
  | DUT clutter drones: as drone / as another class / missed | 0.09 / 0.07 / 0.84 | 0.16 / 0.06 / 0.78 |
  | E21 Halmstad native: clean / trained-kind combination / held-out combination | 0.945 / 0.879 / 0.351 | 0.947 / 0.863 / 0.657 |
  | E21 ~8.5 px (Halmstad k4) | 0.871 / 0.316 / 0.084 | 0.903 / 0.341 / 0.093 |
  | E21 ~14 px (Anti-UAV test k8) | 0.785 / 0.731 / 0.665 | 0.810 / 0.751 / 0.701 |

  * Paired at matched validation recall (E22c at 0.37): test recall +0.005 (not clear), precision −0.061 [−0.082, −0.040], **background FAs +74.6 per 1000 [+42.2, +110.7]**. The sweep puts the extra false alarms at low scores: 252 against 76 per 1000 at 0.25, 18 against 14 at 0.55, equal at 0.70.
  * E35 box error (native sets only; no cached small-target detections) matches `E22a_s0` within 0.3 px in every bin.
  * **E22b, all three seeds by E35:** Anti-UAV test AP 0.899 / 0.884 / 0.893 (mean 0.892; E22a 0.889), R@10 0.844 / 0.829 / 0.839 (mean 0.837; E22a 0.856). Confirms #70: not adopted.
* **Reference points on DUT, found while reading this** (local CPU, `e35_metrics.py` on the two runs with cached DUT detections): **drone AP on DUT, plain seed 1 0.190, E22a seed 1 0.090**. So E22a's DUT cost (15 §6.14) is confirmed without a threshold, and is large. Plain seed 1 at matched thresholds reaches 0.189 at 73 and 0.223 at 146 false drones per 1000 DUT images (#66).
* **What worked.** The paste data loaded and trained at the plain speed (the 2-epoch smoke passed in an earlier session). In-domain scores are unchanged (AP within E22a's seed spread of 0.013) and robustness is no worse; the held-out degradation at native size rose 0.35 → 0.66 (E22a's other seeds: 0.56, 0.61).
* **What did not, or is not yet known.** **DUT drone recall nearly doubled (0.117 → 0.219), almost all from drones named "drone" instead of airplane/helicopter**; any-class recall on sky drones moved little (0.55 → 0.59). But the false drones tripled, and the point (0.219 at 125 per 1000) lies on plain seed 1's curve (between 0.189 at 73 and 0.223 at 146). Read at a single threshold, E22c seed 0 may only have undone E22a's DUT cost, not gone beyond the plain recipe. The extra false alarms on Anti-UAV test are low-scoring (gone by 0.55); the tracker's second stage at 0.1 would see them as candidates.
* **Mistakes.** None new in the runs. The E21 Anti-UAV k8 column reads thousands of false drones per 1000 images in every run (E22a_s0 4,194 clean, plain s0 9,645): each k8 mosaic packs 64 cluttered test frames, so it is per mosaic, not per frame. Not new, but never stated; read that column per 64 frames.
* **Lesson and decision.** A DUT result read at one threshold is exactly the confound of 19 §11. **`e35_metrics.py --dut` now adds DUT to the threshold-free scores** (infers DUT once per run, cached as `dets_dut<sfx>.json`), and `colab_e18.py` gained a final step that runs it over every finished run (`e35_metrics_dut.txt`). E22c is judged on that table over three seeds, against E22a and the plain recipe of the same seeds. Needed from Drive: the `E22c_s1/s2` output files and all new run folders.

**#73 E22c seeds 1 and 2: files read; the three-seed result (local, 2026-10-05).**
* **Intention.** Finish #72's question over three seeds: does E22c move DUT, and at what in-domain cost?
* **What was done.** The user downloaded the `E22c_s1` and `E22c_s2` outputs from Drive; they arrived as two `drive-download-…` folders inside `C:\cuas_data\runs\`. Checked against the nine files `evaluate()` writes per run: all present for both seeds, plus each run's `E22c_s<n>_train.log` and `e12_eval_last.json` (identified by its `run` field). Moved to `runs\` and `runs\E22c_s<n>\`; the empty download folders removed. Training: 40 epochs in 2.10 h (s1) and 2.13 h (s2), about 18% longer than E22a for 18% more training images. Not downloaded, and not needed for this entry: seed 0's loose files (its numbers are in #72) and the six new run folders.
* **Measurements** (all nine models at threshold 0.40; table in 15 §6.17):
  * Anti-UAV test AP: E22c 0.886 / 0.885 / 0.886 (seeds 1 / 2 / 0) against E22a 0.894 / 0.882 / 0.891; R@10 0.825 / 0.857 / 0.845 against 0.880 / 0.850 / 0.839; R@30 0.886 / 0.899 / 0.874.
  * Paired against E22a of the same seed at matched validation recall (E22c at 0.41 / 0.40 / 0.37): test recall −0.001 / **+0.019 [+0.008, +0.033]** / +0.005; precision −0.020 / −0.022 / −0.061 (each clear); **background FAs per 1000 +51.0 [+26.4, +77.1] / +35.8 [+9.1, +63.7] / +74.6 [+42.2, +110.7]**. Sweeps: at 0.55 the test FAs are 16.2 / 19.1 / 18.2 against 10.8 / 18.2 / 14.2.
  * **DUT at 0.40: drone recall 0.188 [0.172, 0.204] / 0.202 [0.186, 0.219] / 0.219** against E22a 0.103 / 0.158 / 0.117; precision 0.483 / 0.521 / 0.574 against 0.406 / 0.537 / 0.594; background false drones per 1000 images 159 / 139 / 125 against 112 / 96 / 40. Sky drones as drone / other class / missed: 0.24 / 0.36 / 0.40 and 0.26 / 0.34 / 0.40 (seed 0: 0.26 / 0.33 / 0.41). Drones boxed as another class (seeds 1, 2): airplane 253, 261; helicopter 252, 239.
  * E21 at 0.40, seeds 1 / 2: Halmstad native 0.937 / 0.925 clean, 0.858 / 0.853 trained-kind combination, 0.542 / 0.629 held out; ~8.5 px 0.913 / 0.861, 0.337 / 0.314, 0.097 / 0.124; ~14 px 0.827 / 0.845, 0.766 / 0.793, 0.692 / 0.736.
* **What worked.** The DUT gain repeats in every seed and is several times E22a's seed spread at the same threshold and similar precision. In-domain AP and robustness hold.
* **What did not.** The in-domain false-alarm cost repeats too (clear in all three pairings), mostly at low scores. Against the plain recipe on DUT, E22c is level at one threshold: E22c s1 (0.188 at 159 per 1000) is slightly *below* plain seed 1's curve (0.189 at 73, 0.223 at 146). So "E22c fixes DUT" is not shown; "E22c undoes E22a's DUT cost while keeping its robustness" is.
* **Mistakes.** None in the runs. The download landed as nested folders, not loose files; checked file by file before reading anything.
* **Lesson and decision.** E22c is the better of the two degraded-training recipes on the unseen dataset. Whether it replaces E22a as the base depends on DUT AP for plain, E22a and E22c over three seeds each (the new last step of `colab_e18.py`). If E22c's DUT AP only matches plain, the background-swap idea is spent and the next lever for DUT is new drone appearance (DUT train split or Drone-vs-Bird, after choosing which set stays unseen).

**#74 E26: the track decision by accumulated evidence (local CPU, 2026-10-05).**
* **Intention.** The user asked how to cut false drones. Per-frame detector false alarms mostly score low and do not persist; what the system answers for is false drone *declarations* per hour (E20). The program declared a track a drone on the first window the classifier called "drone", giving every track many chances. A result that would change what we do: more drone videos declared, or sooner, at the same false-declaration rate. The user also set the project priority (accuracy first, then speed; no over- or underfitting), so the operating point is chosen on training videos only.
* **What was done.**
  * `cuas/decide.py` (`SequentialDecision`): Wald's sequential test. Each classifier reading adds `logit(p) − logit(prior)`, weighted by the share of its 2-s window that is new since the last reading (at 30 fps a track is read 6 times a second, so unweighted the same seconds would count up to 12 times); readings from tracks under 1 s or with features out of range add nothing; the sum is clamped to [reject, accept + 2] so a track can change its mind. `TrackPipeline(decision=...)` uses it; each state also carries `window_label`, `p_fresh`, `p_usable`. `TrackClassifier` stores its training drone share (`prior`); for older pickles `drone_share()` recovers it exactly from the calibrator (an isotonic fit preserves the mean of its targets): 0.426 for both saved E12 classifiers. Tests `test_sequential_decision`, `test_pipeline_with_sequential_decision` and a `drone_share` check (53 tests).
  * `experiments/e26_sequential.py` (venv Python, CPU, ~20 min): every Halmstad video with cached E12 detections; train videos in 5 folds stratified by class, each fold classified by a track classifier fitted on the other folds' windows; val classified by one fitted on all train windows (as E20). The pipeline runs once per video and classifier; both rules are replayed on the recorded states over a grid (hold 0-3 s; accept 0.5-8). Per rule, the operating point is chosen on train: most drone videos declared with at most 10 false declarations per hour (a provisional budget: doc 13 sets none), ties to the shorter delay. Val scored once.
* **Measurements** (`C:\cuas_data\m2\e26_sequential_dets_E12.txt`): the hold-rule replay reproduces E20 exactly (six checks). Track-only classifier, cross-fitted train (23.2 min of non-drone video, 91 drone videos): hold 3 s 2 false (5.2/h), 68/91 declared, median delay 4.07 s; **evidence 5.0: 2 false (5.2/h), 78/91, 3.23 s**. Hold 2 s 10.3/h, 78/91, 3.03 s against evidence 4.0 10.3/h, 80/91, 2.70 s. Hold 1 s 38.7/h, 81/91, 2.03 s against evidence 1.5 38.7/h, 81/91, 1.37 s. Hold 0 (the program's old rule): 144.6/h, 83/91, 1.03 s. Val (5.9 min, 23 drone videos) at the chosen settings: hold 3 s 1 false, 16/23, 4.12 s; evidence 5.0 1 false, 18/23, 3.13 s. Detector+track classifier: val 0 false for every setting of both rules; on train, 0 false at evidence 5.0 (88/91, 2.37 s) against hold 3 s (88/91, 4.03 s).
  * The program on three held-out videos (`E22a_s1` weights, `track_clf_dets_E12.pkl`, CPU): V_DRONE_037 declared 2.33 s after its track starts (old rule 1.0 s); V_BIRD_034 and V_AIRPLANE_017 not declared under either rule.
* **What worked.** The sequential rule is as good or better at every matched rate, on 4× the non-drone footage E20 had, and val agrees in direction. The cross-fitted protocol made the choice without looking at val.
* **What did not.** The single val false declaration survives every setting of both rules: one non-drone track the classifier is confident about. A decision rule cannot fix a confident classifier error; better features, the detector's class, or E23 must. The train detections come from a detector that trained on those videos (cleaner boxes than on new footage). 23 minutes of non-drone video still gives wide intervals (5.2/h has an upper bound of 16/h). The test of clutter tracks as classifier negatives, also listed under E26, needs footage with clutter tracks and was not done.
* **Mistakes.** In the first draft of the test, a pipeline loop unpacked one state per frame; the tracker reports a track only once it is confirmed, so early frames return none. The test was wrong, not the code.
* **Lesson and decision.** Adopted: `python -m cuas` declares at `--evidence 5.0` (0 restores the old rule). The cost is about 1.3 s more delay than the old rule; the gain is a different operating curve, not just a stricter threshold. The deployed value must be chosen on site footage by false declarations per hour; doc 13 needs a per-hour requirement (to agree with the user and the professor).

**#75 Colab L4: DUT scored without a threshold for all twelve runs; E22c adopted (output received 2026-10-05).**
* **Intention.** Settle E22c (#73): is its DUT gain real once no threshold is involved, and does it go beyond the plain recipe? A result that would change what we do: E22c's DUT AP clearly above both E22a and plain makes it the base and says the drone class was tied to its scenes.
* **What was done.** The user uploaded the rebuilt code and re-ran `colab_e18.py` on a fresh runtime (the data zips unpacked again; every training and evaluation step skipped as done). Its new last step, `e35_metrics.py --dut` over the 13 finished runs, ran DUT inference once per run (cached as `<run>/dets_dut_last.json` on Drive). The cell printed only the AP lines; the box-error rows were filtered from the display by the step's `show` pattern, not missing from the file.
* **Measurements** (cell output; `e35_metrics_dut.txt` on Drive). DUT AP, seeds 1 / 2 / 0: plain 0.190 / 0.209 / 0.180, E22a 0.090 / 0.154 / 0.121, E22b 0.169 / 0.177 / 0.132, **E22c 0.177 / 0.188 / 0.220**; `E18_base_s1_e80` 0.189. DUT recall at 10 / 30 false drones per 1000 images: plain 0.041 / 0.117, 0.096 / 0.137, 0.065 / 0.118; E22a 0.047 / 0.068, 0.044 / 0.103, 0.056 / 0.106; E22b 0.071 / 0.122, 0.053 / 0.107, 0.083 / 0.105; E22c 0.057 / 0.102, 0.041 / 0.093, 0.062 / 0.101; 80 epochs 0.088 / 0.141. Anti-UAV test AP as in #68, #73. DUT ECE 0.03-0.10.
  * Paired by seed: E22c − E22a DUT AP +0.087 / +0.034 / +0.099; E22c − plain DUT AP −0.013 / −0.021 / +0.040, Anti-UAV AP +0.030 / +0.027 / +0.041, DUT recall at 30 per 1000 −0.015 / −0.044 / −0.017; E22a − plain DUT AP −0.100 / −0.055 / −0.059.
* **What worked.** One cell gave every run a threshold-free DUT score. E22c's gain over E22a holds without a threshold, in every seed.
* **What did not.** E22c is not better than plain on DUT: AP level, and below plain in every seed when false drones are held to 30 per 1000 images. The background swap undid E22a's DUT cost but added nothing beyond the plain recipe. At strict budgets every recipe finds about one DUT drone in ten: unseen-dataset transfer remains the main open failure.
* **Mistakes.** None in the run. The E35 display filter hid the box-error rows (cosmetic).
* **Lesson and decision.** **E22c becomes the base recipe** (accuracy first: it is no worse than E22a anywhere and better on DUT AP; against plain it trades a little strict-budget DUT recall for +0.033 Anti-UAV AP and the robustness to blur, noise and compression the deployed camera will need). Background variety from our two datasets is exhausted as a lever. Queued next, both on the E22c recipe, three seeds, compared with E22c of the same seed: **E27** (Objects365-pretrained start; `yolo26s-objv1-150.pt` verified in ultralytics/assets v8.4.0, 20.9 MB; the cell downloads it by URL and a 2-epoch smoke prints which class rows start from it) and **E24** (`--freeze 10`). E35 now scores DUT for every run. Beyond that, new drone appearance (DUT train split or Drone-vs-Bird), which needs a decision on which dataset stays unseen. 80 epochs cut DUT false drones as well (one seed): a candidate for the final model.

**#76 E36: what the false drones are; the daytime Anti-UAV score (local CPU, 2026-10-05).**
* **Intention.** The user wants the false drones cut. Before choosing a lever (threshold, sky mask, training data, track stage), find out what they are. A result that would change what we do: false drones mostly on things the deployed camera will see (sky objects, skyline) means training or tracking work; mostly on scenes or artefacts it will not see means fix the measurement first.
* **What was done.**
  * Files: the user downloaded `e35_metrics_dut.*` and ten detection caches from Drive, with no run names (four `dets_dut_last*.json` and three zips). Each was identified by recomputing its E35 scores and matching them to `e35_metrics_dut.json`: every file matched exactly one run with zero error (the next-closest run was 0.009-0.049 away), and each zip's native and DUT files matched the same run. Copied to `runs\<run>\`; `E22c_s0` has no `e12_eval_last.json` (threshold 0.40, from #72).
  * `experiments/e36_false_alarm_map.py` (CPU, ~5 min): every background false "drone" box at score ≥ 0.10 for plain / E22a / E22c × three seeds on Anti-UAV test (native) and DUT, by width, E15's sky/clutter ring texture and score, at 0.40 and at each run's threshold for 30 per 1000 images; contact sheets of the 24 most confident per recipe. It checks that the cached ground truth lines up with the image list.
  * **Audit by eye:** numbered crop sheets of all false alarms of seed 1 per recipe at its 30-per-1000 threshold (52-65 crops each, six sheets), each crop labelled text/logo, real object in the sky, lights, vegetation, structure or other. Labels and paths in `C:\cuas_data\runs\e36_audit_seed1.json`, sheets in `C:\cuas_data\runs\e36_audit\`.
  * **Daytime Anti-UAV set:** per-video median brightness of the 91 test videos is bimodal (18 at 2-9, 73 at 90-147 of 255). `e35_metrics.py` now scores `antiuav_day` (videos ≥ 50; cached in `runs\antiuav_test_brightness.json`) and skips box error when a run's threshold file is missing. Nine runs rescored (`runs\e35_metrics_day.txt`; the seven-run `e35_metrics.txt` left as it was).
* **Measurements.**
  * Counts (three seeds summed): Anti-UAV test at 0.40, plain 192 / E22a 184 / **E22c 471** background false drones; E22c's extra are 32-64 px (309 against 38-48) and score 0.40-0.55 (362 against 96-102). At an equal count the thresholds are 0.37-0.45 (plain), 0.35-0.46 (E22a), 0.47-0.49 (E22c). DUT at 0.40: 550 / 548 / 931; E22a's are small (0-16 px: 198 against 57 for plain).
  * Audit shares (seed 1, equal count): DUT text/logos 22% / 5% / 26%, objects in the sky 6% / 6% / 5%, vegetation 36% / 9% / 39%, structures 34% / 58% / 21%. Anti-UAV objects in the sky 21% / 22% / **48%**, night lights 40% / 27% / 21%, structures 24% / 30% / 6% (plain / E22a / E22c).
  * Daytime Anti-UAV (means over three seeds): AP plain 0.850, E22a 0.887, E22c 0.887; recall at 10 per 1000 0.879 / 0.893 / **0.895**; at 30 per 1000 0.895 / 0.905 / 0.907. All videos, at 10 per 1000: 0.831 / 0.856 / 0.842.
  * DUT recall at 30 per 1000 for plain seed 1 is 0.112 from its local CPU detections against 0.117 from Colab's GPU detections (AP 0.190 both): CPU and GPU inference differ slightly at the margin.
* **What worked.** Matching files by their scores made the unnamed downloads safe to use. The sheets answered in minutes what the counts could not: the texture rule calls text on a black band "sky", so the counts alone overstated sky false alarms on DUT.
* **What did not.** Hand labels by one rater on one seed (about ±12 points per share). Some "objects in the sky" are uncertain: the recurring white object on Anti-UAV may be the target in frames labelled empty, and the light streaks are probably aircraft. DUT false-alarm rates still include overlay text; masking it needs per-image overlay boxes, not made yet.
* **Mistakes.** (1) Every Anti-UAV score so far, including the E22c "false-alarm cost" in #72, #73, mixed night videos into a project whose scope is daytime (13 §6); found by looking at the crops. (2) A doc-edit script was first written through a shell heredoc and failed on its own text, the slip #68 had already ruled out; rewritten as a file.
* **Lesson and decision.** Score the in-scope condition: E35 reports `antiuav_day` from now on, and it should lead in-domain comparisons. By day E22c matches E22a and beats plain at a strict false-alarm budget, which confirms the adoption (#75). Look at false alarms before acting on their count. For the deployment camera the false drones that matter are objects in the sky and the skyline: those go to the track stage (E26), the class head and own footage. Next for false alarms: an overlay mask for DUT's false-alarm rate, and daytime target-free footage.

**#77 Field-session plan for filming our own drones (doc, 2026-10-05).**
* **Intention.** Every gap left after E22c, E26 and E36 needs footage no public dataset has: range and speed truth for time to collision, real drones at 4-16 px for `w_min`, hours of empty sky for false declarations per hour, and a test set nobody has tuned on (19 §11.1). The user flies with a drone club, so the drones and their flight logs are available. The user asked for the shot list and to keep it as a long-term plan; daytime first, night later (11 row 12).
* **What was done.** [20](20_field_sessions_own_footage.md): why (gaps table), a before-session checklist (rules, camera and drone details to send ahead, tape-measured span, camera GPS position, two landmarks for heading, GPS clock, logging on), camera setup, shot list S0-S7 for a daytime session, a distance table, a clip-log template, the processing steps, data rules and session types A-D, and a session log (§9) to fill after each session.
* **Measurements.** None yet. Distances from *R* = *f*ₚₓ · *S* / *w*: a phone at 4K and ~70° (*f*ₚₓ ≈ 2,740) puts a 0.25-m drone at 8 px at 86 m and at 4 px at 171 m; a Pi HQ with a 16 mm lens (*f*ₚₓ ≈ 10,320) puts it at 323 m and 645 m.
* **What worked.** Writing the processing steps first (sync, pose, projection) fixed what has to be filmed: a visible clock and lift-off for sync, corner hovers for the camera's pose, a measured span for size.
* **What did not.** The ladder uses an assumed phone field of view (70°) and assumed spans; it needs the user's camera and drone models. No flight-log reader exists except PX4's.
* **Mistakes.** None.
* **Lesson and decision.** Own footage is the project's planned unseen set: the first sessions are marked test before anything is scored and are never used to choose a recipe or threshold. Projected boxes from a log are checked by eye before use (#69, #76). Next: the user sends the camera model and video mode and each drone's model, flight controller and span; the exact ladder and the log reader follow.

**#78 Colab L4: E27 (Objects365 start) and E24 (`--freeze 10`) trained, three seeds each (output received 2026-10-06).**
* **Intention.** The two queued arms on the E22c recipe, each against E22c of the same seed: does a broader pretraining start (E27) or a frozen backbone (E24) keep the general features that find drones in an unseen dataset (DUT)?
* **What was done.** `colab_e18.py` trained `E27_s1/s2/s0` and `E24_s1/s2/s0` (40 epochs, final weights) and ran the per-run protocol (e12_eval, paired comparison with E22c, E15, E21, E35 with DUT). Only the last run's E35 was in the pasted output; the rest are on Drive. The final all-runs DUT step did not re-run (its marker was set on 2026-10-05). The E35 table on Colab has no `antiuav_day` row, so the uploaded `cuas_code.zip` predates that change; the daytime score is computed locally.
* **Measurements (one seed so far).** `E24_s0@last` against `E22c_s0@last`: **DUT AP 0.310 against 0.220**, R@10 0.102 / 0.062, R@30 0.163 / 0.101; Anti-UAV test AP 0.858 against 0.886, R@10 0.857 / 0.845, R@30 0.880 / 0.874; Halmstad val AP 0.989 / 0.994. Box width +24% at 16-32 px (E22c +21%).
* **What worked / did not / mistakes.** Pending the files of the other five runs.
* **Lesson and decision.** None until three seeds: a freeze that keeps the COCO backbone may trade in-domain AP for transfer, which is the trade this project wants if the daytime in-domain score holds.

**#79 Demo video re-rendered with the evidence-based decision (local CPU, 2026-10-06).**
* **Intention.** The user asked for a short video showing the system working. The #71 demo predates E26, so its drone was declared by the old first-window rule.
* **What was done.** `experiments/make_demo.py` re-run unchanged in content (E22a_s1 weights, since the E22c weights are still on Drive; `track_clf_dets_E12.pkl`; the program's default `--evidence 5.0`). ffmpeg is not on this laptop's PATH, so the first render stayed in OpenCV's mp4v codec, which browsers and phones often refuse; OpenCV's Media Foundation H.264 writer made 120 MB and ignored its quality setting. Fix: `pip install imageio-ffmpeg` in `C:\cuas_venv` (a bundled ffmpeg), which the script now finds when ffmpeg is not on PATH, called through `subprocess` with an argument list.
* **Measurements.** 42.5 s (the closing summary card removed at the user's request; it ends on the degraded-video comparison), 1280×556, H.264, 2.9 MB (`C:\cuas_data\demo\progress_demo.mp4`). Drone declared 2.4 s after its track starts (#71: 1.1 s with the old rule); bird and airplane not declared; degraded video, drone boxed in 144 (plain) against 255 (E22a) of 270 frames, as in #71.
* **What worked.** Sample frames checked by eye before sending.
* **What did not.** The demo still shows E22a, not the adopted E22c; it should be re-made when the E22c (or E24) weights are local.
* **Mistakes.** The first ffmpeg call went through `os.system` with a quoted program path, which `cmd.exe` strips when the line holds further quotes, so it failed silently and the mp4v file was kept.
* **Lesson and decision.** Call external programs with an argument list, not a shell string, on Windows.

**#80 `C:\cuas_data\runs` tidied; E24 and E27 files in (local, 2026-10-06).**
* **Intention.** The user downloaded the six E24/E27 run folders and their loose files as twelve zips into `runs\` and asked for the folder to be tidied: 118 entries, most of them per-run report files at the top level.
* **What was done.** `tools/tidy_runs.py` (dry run by default, `--apply` to act; moves only, never deletes): unpacks Drive zips (a run folder's zip into `runs\`; a loose-file zip's `e12_eval_last.json` into the folder of the one run its other file names name), keeps an existing file and reports a differing copy, parks each zip in `C:\cuas_data\_downloads\`; then moves every report named after a local run folder (`eval_`, `cmp_`, `e15_`, `e21_`, `e28_`, `e35_` files, training logs) into `<run>\reports\`. Code paths checked first: scripts read only run folders and `antiuav_test_brightness.json`, which stay where they were. Doc mentions of the moved files rewritten (16: 7 lines, 19: 2), the location table in §3 updated.
* **Measurements.** Six run folders and 72 loose files unpacked, no conflicts (the loose `e12_eval_last.json` of each zip equals the copy in its run folder); 116 reports moved; 36 files stay at the top (cross-run summaries, and the E13 files, which have no local run folder). Zips parked: ~262 MB.
* **What worked.** A dry run first, then the same script with `--apply`.
* **What did not.** `E22_checks.log` (the E27 smoke check) was not among the downloads; the training logs may show which classes the Objects365 start carried over.
* **Mistakes.** A first in-line attempt to rewrite the doc paths escaped the backslash class wrongly and changed nothing; rewritten as a file (the #68 lesson again).
* **Lesson and decision.** Scripts and the Colab queue keep writing to the top of `runs\`; run `tidy_runs.py --apply` after each download.

**#81 E24 and E27 scored, three seeds each; E37 (DUT by background); E24 adopted provisionally (local CPU, 2026-10-06).**
* **Intention.** Decide the two queued arms on the E22c recipe (#78): does a broader pretraining start (E27) or a frozen backbone (E24) find more drones in the unseen dataset without losing the in-domain score? A result that would change what we do: a DUT gain beyond the seed spread in every seed.
* **What was done.** `e35_metrics.py` gained `--out NAME` (so a table no longer overwrites `e35_metrics.txt`); run over plain, E22a, E22c, E24 and E27, three seeds each, from cached detections (`runs\e35_e24_e27.txt`, 5 min). E24's gain at its own threshold (E15) was mostly on clutter backgrounds, so `experiments/e37_dut_sky.py` splits the threshold-free DUT scores into images whose every drone is on sky and the rest (`runs\e37_dut_sky.txt`). E21 tables and training logs read. LP-FT checked at its arXiv page (2202.10054, ICLR 2022 oral; already in 01 and 19 §5).
* **Measurements** (means over seeds 1 / 2 / 0; full tables in 15 §6.19):
  * E24 against E22c: DUT AP 0.311 / 0.195 (per seed +0.144 / +0.113 / +0.090), R@30 0.165 / 0.099; daytime Anti-UAV AP 0.846 / 0.887, R@10 0.871 / 0.895 (per seed −0.038 / −0.020 / −0.013); Halmstad AP 0.985 / 0.995. Training 1.67 h a run on an L4 (E22c 2.1 h).
  * E27 against E22c: DUT AP 0.222 / 0.195 (+0.079 / −0.003 / +0.006); daytime R@10 0.891 / 0.895. "Remapped 2/4 cls head rows" (bird, airplane), as from COCO. 2.1 h a run.
  * E37, sky-only DUT (1,263 images): AP plain 0.255, E22a 0.149, E22c 0.242, **E24 0.330**, E27 0.282; R@30 0.190 / 0.127 / 0.125 / **0.200** / 0.161. Clutter (937 images): AP 0.116 / 0.093 / 0.136 / **0.286** / 0.149.
  * Local CPU detections give plain s1 DUT R@30 0.112 (Colab GPU 0.117, #76), so this table's plain mean is 0.122, not 0.124.
* **What worked.** The paired seeds: E24's DUT gain is positive in all three, and so is its in-domain loss, which makes it a clear trade, not noise. Splitting DUT by background showed the gain is not only on scenes the deployed camera never sees.
* **What did not.** E27 does not help reliably. E21 cannot compare E24 with E22c because they chose different thresholds (0.55 against 0.40). The in-domain loss puts E24 below plain at the strictest daytime budget.
* **Mistakes.** E22c was adopted (#75) on pooled DUT AP; split by background, on sky at strict budgets it is worse than plain in every seed. Pooled scores over a mixed dataset hid an in-scope loss, the same kind of slip as the night videos in #76.
* **Lesson and decision.** Score the in-scope part of every dataset (day, sky), not the pool. **E24 becomes the base recipe provisionally**, because the project's priority is accuracy on unseen footage. Next arms (19 §8 row 5, not built): E24b `--freeze 5`, E24c freeze then unfreeze (LP-FT), E24d plain + freeze 10; three seeds each, ~15 L4-hours. Confirm the choice on a set nobody has tuned on (own footage, doc 20).

**#82 E24b/c/d queued; freeze-then-release added to `train_local.py` (local, 2026-10-06).**
* **Intention.** The user approved the three follow-ups to E24 (#81): can the in-domain loss of a frozen backbone be won back without losing its DUT gain?
* **What was done.** `train_local.py --unfreeze-at N` (with `--freeze`): an `on_train_epoch_start` callback sets every frozen parameter trainable from epoch N and clears the trainer's frozen-name list, so the BatchNorm layers also go back to training mode; a marker file (`cuas_unfreeze_at.txt`) makes `--resume` do the same. Ultralytics 8.4.162 puts every parameter, frozen or not, into the optimizer (`build_optimizer` walks all modules), so no rebuild is needed. A second callback prints once whether the first backbone layer changed during the first released epoch. `--device` added for CPU smoke tests. `colab_e18.py` gained `E24b_s1/s2/s0` (`--freeze 5`), `E24c_*` (`--freeze 10 --unfreeze-at 30`) and `E24d_*` (plain data, `--patience 100 --freeze 10`), ordered seed 1 of all three first; each is compared with E24 of the same seed; a 2-epoch smoke check (`checks_e24c`, release at epoch 1) runs before them. `cuas_code.zip` rebuilt (it now also carries `antiuav_day` scoring, missing from the zip Colab used for E24/E27).
* **Measurements.** CPU smoke test, 160 pasted tiles, 2 epochs, release at epoch 1: "released 99 frozen parameter tensors at epoch 1", "first backbone layer changed during epoch 1: True". In-process check with `nbs=4` (one optimizer step per batch): epoch 0, 6 steps, no backbone gradient, backbone unchanged, head changed; epoch 1, 6 steps, all with backbone gradients, backbone changed. Unit tests 63 run, 10 skipped, OK. Cost on an L4: ~1.7 h a run, ~15 h for the nine plus scoring.
* **What worked.** Reading the trainer source first: the optimizer already held the frozen weights, which made the release a few lines.
* **What did not.** The first smoke test (24 images, batch 4) showed no change anywhere, because Ultralytics accumulates gradients towards a nominal batch of 64 and the run took a single optimizer step; a head weight chosen for the check was on a branch outside the loss. Neither was a fault in the code under test. The resume path with a release marker is untested (no interrupted run to resume).
* **Mistakes.** Two misleading checks before a valid one (above): a smoke test must be large enough for the mechanism to act.
* **Lesson and decision.** The user re-pastes `colab_e18.py` after uploading the new `cuas_code.zip`. The decision rule for the three arms: keep the arm with the best daytime Anti-UAV R@10 among those whose DUT sky AP (E37) stays within the seed spread of E24's.

### 2026-10-08: E23 (temporal input), E25 (RF-DETR) and E39 (DUT train split) built

Entries #83 and the first half of #84 were written the same day from the code, the build logs and `build_e23.json`: the session that did the work ended before it was logged.

**#83 E23: temporal input built (local, 2026-10-08).**
* **Intention.** 19 §5 rank 2 and §9: YOLOMG's motion input is the largest single effect in any source read, and under 8 px one frame may not hold enough to tell a drone from noise. Does a detector that sees three frames find more small drones, at no more false alarms, than the same detector on one frame? A result that would change what we do: E23 above its control under 8 px in every seed.
* **What was done.** `cuas/temporal.py`: three grey frames t−2k, t−k, t become the three image channels (so the pretrained first layer still applies), the past two registered to frame t by phase correlation on clipped gradient magnitude (the registration `cuas/motion.py` already uses), past frames only. `experiments/build_e23.py --k 3` rebuilds every training and evaluation image of the E14 recipe from its source video (Halmstad `Video_V`, the Anti-UAV zip, inpainted with the same overlay mask), with the same names, crops and labels, and rebuilds the zoom-out mosaics from the stacked sources with the same seed. Stacks are written as 4:4:4 JPEG (`zoomout.make_split` gained `jpg_params`), because 4:2:0 halves the resolution of channel differences, which is where the motion is. `train_local.py --temporal` turns off hue and saturation jitter and Ultralytics' albumentations step (grey conversion, CLAHE), all of which mix channels; a marker file makes `--resume` do the same. The control **E23ctl** is frame t in all three channels (`build_e23.py --static`, derived on Colab so only one set is uploaded). Two unit tests: registration undoes a known camera shift; a stack keeps frame t untouched and shows a moving 3-px target and nothing else.
* **Measurements** (`C:\cuas_data\build_e23_k3.log`, `e23_k3\build_e23.json`). 18,016 images from 598 videos. Frame t of each stack against the existing single-frame image: mean absolute difference median 0.10 grey levels, 99th percentile 0.56, maximum 0.62 (so the frame indexing and crops are the same). Registration shift: median 0.11 px, 95th percentile 3.7 px, maximum 926 px (a cut, handled in #84).
* **What worked.** Checking channel 2 against the existing images found no mismatch: the stacks pair image for image with the single-frame sets.
* **What did not.** The recipe is the plain one on the E14 data, not the base recipe: `--degrade` has a JPEG step that mixes channels, and a frozen backbone could not learn to read motion. So E23 is compared with its own control, and the control with the plain run of the same seed (what dropping colour costs). DUT cannot score it (still images), so E23 has no unseen-dataset test.
* **Mistakes.** The session ended with nothing logged and `cuas_code.zip` not rebuilt (#87).
* **Lesson and decision.** Six runs queued (#87).

**#84 E23: cuts reset to static; one video rebuilt; packed (local, 2026-10-08).**
* **Intention.** Make the stacks safe to train on and complete.
* **What was done.** (a) `build_e23.reset_cuts`: a stack whose past frames differ from frame t over the whole image (median change above 8 grey levels: scene cuts and zoom changes in Halmstad, exposure jumps in Anti-UAV) carries no usable motion, so it becomes a static stack; a live pipeline restarts its frame buffer on such a jump. (b) The first build missed one Anti-UAV video (`20190925_140917_1_5`, 40 training tiles): OpenCV's decoder timed out ("Stream timeout triggered after 901 s"). Found on reading the build log in the next session; that video rebuilt alone with `build_e23.build_video`, the cut reset re-run, `e12/antiuav_zoom_train` deleted and rebuilt. (c) `pack_for_colab.py --e23`.
* **Measurements.** 197 of 18,016 stacks reset to static (195 in the first pass, 2 in the rebuilt video). After the rebuild every split has the same image count as its single-frame set and every mosaic set the same count (Anti-UAV train k2 894, k4 223; they were 888 and 222). The 40 rebuilt tiles: frame-t difference at most 0.34. `cuas_data_e23.zip`: 1.494 GB, 46,642 files. Local smoke (`--temporal`, 2 epochs, 2% of the data, RTX 4060): trains and validates, 3.5 min, `hsv_h=0.0, hsv_s=0.0` in the trainer's arguments.
* **What worked.** The build's own count line ("888 mosaics (single-frame set: 894)") made the gap visible.
* **What did not.** With 40 tiles missing the training mosaics had different layouts from the single-frame set (the shuffle depends on the count), which would have weakened the E23ctl-against-plain pairing in training. Fixed before upload.
* **Mistakes.** The first zip was packed with the gap in it; a build that reports missing frames must stop, not continue. A sheet of 16 stacks cropped around the drone was looked at afterwards: a moving drone shows as three coloured copies on a grey (aligned) background, a slow one as coloured fringes, which is what the design intends.
* **Lesson and decision.** Read the whole build log before packing. Upload `cuas_data_e23.zip`.

**#85 E25: RF-DETR Small behind the detector interface; local smoke runs; Colab faults fixed before the run (local RTX 4060, 2026-10-08).**
* **Intention.** 19 §5 rank 5. E24 showed that keeping pretrained features is what transfers to DUT. Does a backbone pretrained far more broadly than COCO (DINOv2) find more unseen-dataset drones than YOLO26s with a frozen COCO backbone? A result that would change what we do: DUT AP (pooled and sky) above E24 in every seed without an in-domain loss. RF-DETR sizes N-L are Apache-2.0, which also matters for publishing (13 §5).
* **What was done.** `experiments/train_rfdetr.py` merges an Ultralytics data yaml's folders into one Roboflow-layout YOLO folder per split and trains `RFDETRSmall` at 640; `cuas.detector.RFDETRDetector` pads each tile to 640 × 640 (never scales) and `cuas.detector.load` picks it when the run folder holds `cuas_rfdetr.json`, so `e12_eval.py`, `e15_dut.py`, `e21_degradation.py`, `e28`, `e33` and `e35` score an RF-DETR run unchanged. Installed `rfdetr` 1.11.2 for a local test. A side environment `C:\cuas_venv_rfdetr` (a venv with a `.pth` file pointing at `C:\cuas_venv`'s packages, plus RF-DETR's training extras installed with `--no-deps`) keeps the GPU venv's numpy and OpenCV as they were. Runs: 1 epoch on 2% of E22c's data, scored with `e12_eval.py`; then 3 epochs on 10% to check the class order.
* **Measurements.**
  * 1 epoch, 369 images: trains (2.6 min); `e12_eval.py` runs to the end in 23.9 min. The model is too undertrained to score: every drone boxed as airplane or helicopter.
  * 3 epochs, 1,848 images, 35.9 min (the 8 GB card was full: 6-11 min an epoch): RF-DETR's own validation mAP@50 0.82. On 269 held-out validation images with one class each, the top box is "drone" on 180 of 180 drone images, airplane on 27 of 32, helicopter on 28 of 32, bird on 7 of 25. So the class order is kept. 73 ms per 640 tile on the 4060, unoptimised.
* **What worked.** Running it locally before queueing it: the faults below would each have cost a Colab round trip.
* **What did not.**
  * `pip install rfdetr` is not enough to train: it needs `rfdetr[train]` (PyTorch Lightning and others), which would also change numpy and add a headless OpenCV. **Fix:** on Colab RF-DETR gets its own environment on top of the system packages (`virtualenv --system-site-packages`, `/content/rfdetr_env`); E25's training and scoring use that interpreter, the YOLO runs never see it.
  * The merged dataset was a folder of symlinks inside the run folder, which on Colab is on Google Drive, where symlinks do not exist. **Fix:** it is built under `<CUAS_DATA>/rfdetr_ds/` (local disk) and removed after training.
  * RF-DETR leaves ~0.9 GB of checkpoints per run. **Fix:** deleted once the final EMA weights are copied to `weights/last.pt` (128 MB); `--keep-checkpoints` keeps them.
  * The head has one more output than there are classes ("background", id 4), and a few boxes carry it. **Fix:** the wrapper drops them (unit test `RFDETRWrapper`).
* **Mistakes.** The queue had `RFDETR_EPOCHS = 20` with a note that the smoke check would set it, but nothing did, and 20 epochs of ~21,400 images may take far longer than a YOLO run. Now `train_rfdetr.py` records its training time and `colab_e18.py` fits the epoch count to a budget once (at most 20, at least 6, 4 h a run by default, `RFDETR_BUDGET_H`), stored in `E25_epochs.txt` on Drive so every seed gets the same schedule. The first reading of the smoke evaluation ("class ids are shifted") was wrong; the 10% run settled it.
* **Lesson and decision.** Code that has never run does not go into the Colab queue. E25 is a model-family comparison (its own augmentation and schedule; no `cuas.degrade`), not a one-change arm, and it is expected to be worse on small targets (patch 16: an 8-px drone is half a token). No resume: an interrupted E25 run restarts. Scoring will be several times slower than a YOLO run's. The Colab environment step itself is untested (it cannot be run locally).

**#86 E39: DUT train and val downloaded, converted and tiled (local CPU, 2026-10-08).**
* **Intention.** 19 §3.9 and §9: the 2025 Drone-vs-Bird winner trained on DUT; how much of the DUT failure is simply missing data? **Cost:** once a model has trained on DUT's train split, DUT test measures "seen dataset, unseen images", like the Anti-UAV test. So E39 is last in the queue, after every arm that is chosen on DUT test, and the unseen yardstick afterwards has to be own footage (doc 20) or Drone-vs-Bird.
* **What was done.** `train.zip` (745 MB) and `val.zip` (372 MB) fetched with `gdown` from the Google Drive links in the DUT repository's README (`wangdongdut/DUT-Anti-UAV`, branch `master`) into `%USERPROFILE%\Downloads`. `experiments/build_e39.py`: VOC XML to YOLO (`cuas/datasets/dut.py`), one 640-px tile per drone at true pixel size, smaller images padded with grey, no background tiles (DUT labels only drones, so empty parts may hold unlabelled birds); `data_e39.yaml` = E22c's training list plus the tiles, validation unchanged. `pack_for_colab.py --e39`.
* **Measurements.** Train: 5,200 images, 5,243 drones, widths 8-3,570 px (10th / 50th / 90th percentile 24 / 43 / 169). Val: 2,600 images, 2,621 drones (same percentiles within 1 px), converted to `C:\cuas_data\dut_val` and not used yet. Tiles: 5,211 with 5,259 drones; in the tiles 1% of drones are under 16 px, 71% are 16-64 px, 27% are larger. `cuas_data_e39.zip`: 0.425 GB.
* **What worked.** Every image had its XML file.
* **What did not.** Nothing failed. DUT adds scenes and drone models, not small targets.
* **Mistakes.** None found. A contact sheet of 12 random tiles was looked at (the #69 lesson): every box sits on a drone at true size; most scenes have buildings or trees behind or beside the drone, so much of what E39 adds is outside the deployed camera's sky-only view (score it with the E37 sky split).
* **Lesson and decision.** E39 = E24's recipe (`--degrade`, E22c data, `--freeze 10`) plus these tiles, three seeds, compared with E24 of the same seed. DUT val stays untouched as a possible threshold set.

**#87 Queue extended; code zip rebuilt (local, 2026-10-08).**
* **What was done.** `colab_e18.py`: after seed 1 of E24b/c/d come `E23ctl_s1`, `E23_s1` and `E25_s1`; then seeds 2 and 0 of E24b/c/d, of E23 and its control, and of E25; `E39_s1/s2/s0` last. A run whose data zip is missing from Drive is skipped with a note. E23 runs are scored on the stacked copies of the evaluation sets (`evaluate_e23`: `e12_eval`, the paired comparison, E35; no DUT, no E21). Smoke checks for E23 and E25 run first. `cuas_code.zip` rebuilt (it dated from 2026-10-06 and had none of this).
* **Measurements.** Unit tests: 67 run, 10 skipped, OK; `test_arch.py` 10 OK in the GPU venv after the `rfdetr` and `gdown` installs; tools tests 18 OK.
* **To run.** Upload to `MyDrive/cuas/`: `cuas_code.zip` (replace), `cuas_data_e23.zip`, `cuas_data_e39.zip`; re-paste `experiments/colab_e18.py`. About 21 runs, 45-55 L4-hours with scoring if all of it runs; it is in priority order and can be stopped after any run.
* **Not known at the time of writing.** Whether E24b/c/d have already run on Colab: no output newer than 2026-10-06 is in `C:\cuas_data\runs\`.

**#88 Runtime cap removed; RF-DETR resumes; E24 at 80 epochs queued (local, 2026-10-08).**
* **Intention.** The user: accuracy comes first and long runtimes are acceptable; there is no own footage and no Drone-vs-Bird video yet. So nothing in the queue should be shortened to save hours, and DUT test stays the only unseen yardstick for now.
* **What was done.**
  * **E25:** the 4-hour budget from #85 is gone; RF-DETR trains its full 20 epochs. Because a run may now outlast a Colab session, `train_rfdetr.py` resumes: a run folder with RF-DETR's `last.ckpt` and no `cuas_rfdetr.json` is continued from its last finished epoch (`model.train(resume=...)`); an unreadable `last.ckpt` is set aside and the run starts again.
  * **E24 at 80 epochs** (`E24_s1_e80`, then seeds 2 and 0): the candidate final model. On the plain recipe 80 epochs halved the in-domain false alarms at the same recall and raised DUT recall at 30 false drones per 1000 images from 0.117 to 0.141 (#70, #75); it was kept out only for its cost. Compared with E24 of the same seed. Comparisons between arms stay at 40 epochs.
  * **E39 stays last.** With no other unseen set, a model trained on DUT's train split cannot be confirmed on unseen data, so E39 is a measurement of what DUT data is worth, not a candidate base recipe, until own footage or Drone-vs-Bird exists. Its DUT test score is a "seen dataset" number and must be labelled so.
* **Measurements.** Resume test on the 4060 (184 images): 1 epoch, marker removed, same command with `--epochs 2`: "resuming ... (epochs finished: 0)", Lightning "Restored all states", only epoch 2 of 2 ran, final weights written, checkpoints removed.
* **What worked.** RF-DETR's trainer already takes a `resume` checkpoint; the wrapper only had to find it.
* **What did not.** Not tested: a resume after a real kill in the middle of an epoch, and the unreadable-checkpoint branch.
* **Mistakes.** #85 capped a run by time under a rule the project had already dropped (project rule, 2026-10-05: do not reject an option for compute cost).
* **Lesson and decision.** Queue order: seed 1 of E24b/c/d, `E23ctl_s1`, `E23_s1`, `E25_s1`, `E24_s1_e80`; then seeds 2 and 0 of each; E39 last. About 24 runs; of the order of 70-100 L4-hours, RF-DETR's share unknown until its first run. `cuas_code.zip` rebuilt.

**#89 E24b/c/d trained on Colab, all three seeds; the 2026-10-08 queue has not started (Colab L4, output received 2026-10-10).**
* **Intention.** Close the open point of #87: had E24b/c/d run, and with which version of the queue.
* **What was done.** The user sent the cell output of the latest `colab_e18.py` run. Read against the script and `C:\cuas_data\runs\`.
* **Measurements.** None yet. The output lists 28 runs as done, ending `E24b/c/d_s1`, `_s2`, `_s0`, then "ALL DONE". No scores are printed in it; they are in the loose files on Drive.
* **What worked.** All nine E24b/c/d runs finished (train, evaluation, DUT, E21, by the DONE markers).
* **What did not.** The cell was the 2026-10-06 version of the queue: setup unzips only `cuas_data_e14.zip` and `cuas_data_e22c.zip`, there is no RF-DETR environment step, and no line (done or "skipped: upload ...") for E23, E23ctl, E25, E24 at 80 epochs or E39. So none of the 2026-10-08 runs has started. Whether the uploaded `cuas_code.zip` was the 2026-10-08 one cannot be read from the output.
* **Mistakes.** None found.
* **Lesson and decision.** (1) Download the nine `E24b/c/d_s*` folders and the new loose files from `MyDrive/cuas_runs/` into `C:\cuas_data\runs\`, run `python tools/tidy_runs.py --apply`, then score them locally (E35 with `antiuav_day` and DUT, E37 sky DUT) and decide by the #82 rule: best daytime R@10 whose sky-DUT AP stays within E24's seed spread. (2) Upload the 2026-10-08 `cuas_code.zip`, `cuas_data_e23.zip` and `cuas_data_e39.zip` to `MyDrive/cuas/` and paste the current `experiments/colab_e18.py`; finished stages are skipped.

**#90 E24b/c/d scored: none replaces E24; E24 stays the base recipe (local CPU, 2026-10-10).**
* **Intention.** Decide the freeze arms by the #82 rule: keep the arm with the best daytime Anti-UAV R@10 whose sky-DUT AP stays within E24's seed spread. Also the user's question: do we need more epochs?
* **What was done.**
  * The nine run folders and the loose files the user downloaded were moved into `C:\cuas_data\runs\` (the Drive zips had been unpacked into two `drive-download-*` folders, which `tidy_runs.py` does not read); 33 loose files that were copies of E24's existing reports were compared byte for byte (all identical) and parked with their folder in `C:\cuas_data\_downloads\`; `tidy_runs.py --apply` moved 13 reports.
  * Each run checked against its `args.yaml`, `results.csv` and markers: 40 epochs, patience 100, the right seed, `freeze` 5 / 10 / 10, E22c data and `cuas_degrade.txt` for b and c, E14 data and no degradation for d, `cuas_unfreeze_at.txt` = 30 for c. E24c's logs (seeds 0 and 2, both resumed after a disconnect, at epochs 21 and 16) show "released 99 frozen parameter tensors at epoch 30" and "first backbone layer changed during epoch 30: True", so the release survived the resume.
  * `e35_metrics.py --out e24bcd` and `e37_dut_sky.py` over plain, E22c, E24, E24b, E24c, E24d, three seeds each, from the cached detections (`runs\e35_e24bcd.txt`, `runs\e37_dut_sky.txt`; the 2026-10-06 E37 table is kept as `e37_dut_sky_2026-10-06.txt`). `e37_dut_sky.py` got the three new groups.
* **Measurements (three-seed means; [15 §6.19](15_detector_training_methods_and_findings.md#619-e24-and-e27-a-frozen-backbone-and-an-objects365-start-dut-by-background-2026-10-06)).**

  | Recipe | Day AP | Day R@10 | DUT AP (seeds 0, 1, 2) | DUT R@30 | Sky-DUT AP | Sky-DUT R@30 |
  | :--- | ---: | ---: | :--- | ---: | ---: | ---: |
  | Plain | 0.850 | 0.879 | 0.193 (0.180, 0.190, 0.209) | 0.122 | 0.255 | 0.190 |
  | E22c | 0.886 | 0.895 | 0.195 (0.220, 0.177, 0.188) | 0.099 | 0.242 | 0.125 |
  | **E24** (freeze 10) | 0.846 | 0.871 | **0.311** (0.310, 0.321, 0.301) | **0.165** | **0.330** | **0.200** |
  | E24b (freeze 5) | 0.859 | 0.883 | 0.166 (0.210, 0.146, 0.142) | 0.076 | 0.192 | 0.079 |
  | E24c (freeze 10, release at 30) | 0.875 | 0.887 | 0.227 (0.218, 0.240, 0.224) | 0.138 | 0.273 | 0.190 |
  | E24d (plain + freeze 10) | 0.856 | 0.864 | 0.169 (0.173, 0.147, 0.188) | 0.090 | 0.196 | 0.112 |

* **What worked.** The rule gives a clear answer. E24's sky-DUT AP is 0.308-0.352 over its seeds; E24b (0.192) and E24d (0.196) are far below it in every seed, E24c (0.273) is below it in two seeds of three. No arm qualifies, so E24 stays.
* **What the arms show.**
  * **E24c:** ten epochs of training the released backbone at a low learning rate give back 0.03 of daytime AP and lose 0.08 of DUT AP. The in-domain score and the unseen-dataset score pull the backbone in opposite directions, and it moves quickly.
  * **E24b:** freezing layers 0-4 only keeps nothing of the gain. What matters is that layers 5-9 (the deep half of the backbone) stay as pretrained.
  * **E24d:** freezing by itself does nothing (0.169 against plain 0.193). The gain needs the frozen backbone together with degradation and pasted drones.
* **What did not.** The download held only the `cmp_` reports and four training logs for E24b/c/d; their `eval_`, `e15_`, `e21_` and `e35_` loose files and five training logs are still on Drive. E35 and E37 were recomputed from the cached detections, so the decision does not need them, but **the E21 (degradation) tables of E24b/c/d have not been read**.
* **Mistakes.** The queue's last step (the DUT table over all runs) was guarded by one marker, `e35_dut_all`, so after its first run it never ran again and `e35_metrics_dut.txt` on Drive stops at the 2026-10-05 runs. The marker is now keyed by the number of runs.
* **Epochs (the user's question).** Not more than the queue already has.
  * In-domain training is finished by about epoch 25: validation loss is flat from there (E24 seeds: 2.25-2.28 at epochs 25, 30, 35 and 40) while the training loss keeps falling (2.67 → 2.28), and validation mAP peaks between epochs 9 and 33 in every run looked at.
  * Longer training of weights that are free to move buys in-domain score only: 80 epochs of the plain recipe halved the in-domain false alarms and did nothing for DUT AP (#70), and E24c shows ten epochs of a free backbone already cost DUT.
  * The limit is the data, not the schedule: about 18,000 training images from a few hundred videos of two datasets. The levers are new scenes (E39, own footage), time (E23) and a stronger pretrained backbone (E25).
  * **Over- or underfitting (the user asked, 2026-10-10; `results.png` and `results.csv` of plain, E24, E24c, E24d, seed 1).** Not underfitting: in-domain scores are high (Halmstad AP 0.99, Anti-UAV AP 0.85-0.89). Mild overfitting inside the trained datasets: validation loss is flat, not rising, while training loss falls (plain, epoch 30 → 40: box 1.48 → 1.29, class 0.65 → 0.51; validation 1.62 / 0.60 → 1.61 / 0.61), and validation recall or mAP50 slips a little in the last ten epochs (plain recall 0.859 → 0.820; E24 mAP50 0.867 → 0.846). The large gap is to the unseen dataset (DUT AP 0.19-0.31), which these curves cannot show because validation comes from the same two datasets. The frozen backbone is a regulariser (final training loss 1.51 / 0.79 against plain 1.29 / 0.51 at the same validation loss) and is the one change that raised DUT, which points to fitting the trained scenes too closely, not to too little training. Training loss is measured on augmented images, so its level is not comparable with validation loss; the step at epoch 31 is mosaic switching off.
  * `E24_s*_e80` stays queued as the test of the one open case (a frozen backbone, where only the neck and head train longer). Expect fewer in-domain false alarms; DUT must not fall. Nothing beyond 80.
* **Lesson and decision.** E24 is the base recipe; the freeze question is closed at this model size. `cuas_code.zip` rebuilt 2026-10-10. Next: the user uploads it with `cuas_data_e23.zip` and `cuas_data_e39.zip` and pastes `experiments/colab_e18.py`.

**#91 The 2026-10-08 queue started on Colab; every setup check passed (Colab L4, 2026-10-10 evening).**
* **Intention.** Start E23ctl / E23 / E25 / E24 at 80 epochs / E39 with the code zip of #90, and confirm the one step never run on Colab (the RF-DETR environment).
* **What was done.** The user uploaded `cuas_code.zip`, `cuas_data_e23.zip`, `cuas_data_e39.zip` and ran `colab_e18.py`.
* **Measurements (cell output).** E23 control set derived on Colab: 23,320 static images. RF-DETR environment: `rfdetr[train]` installed in `/content/rfdetr_env`, torch 2.11.0+cu130, CUDA available. `E25_smoke` (1 epoch, 924 images): trained in 1.2 min, scored by `e12_eval.py` in 9.3 min (Anti-UAV test native recall 0.945 at conf 0.4, so the class order is right; the other numbers of a one-epoch model mean nothing). `E23_smoke`: "E23 temporal input: hue/saturation augmentation and albumentations off", 2 epochs completed. The 22 earlier runs were skipped as done. `E23ctl_s1` started: seed 1, 40 epochs, patience 100, `hsv_h=0.0, hsv_s=0.0`, data `e23_k0`, 708/708 pretrained items transferred, 15,480 Anti-UAV training tiles found.
* **What worked.** Everything up to the first training run.
* **What did not.** Nothing so far. Not yet seen: a full RF-DETR run, and its resume after a disconnect.
* **Mistakes.** None found.
* **Lesson and decision.** A full E25 epoch is about 24 min by extrapolation from the smoke run (about 8 h for 20 epochs). Next output to read: `[E23ctl_s1] done` and `[E23_s1] done` (the first time-against-control comparison).

**#92 `E23_s1` trained and scored: temporal input beats its control on the Anti-UAV test, one seed; `E25_s1` in its last epoch (Colab L4, 2026-10-11).**
* **Intention.** First reading of E23 (three registered grey frames) against E23ctl (the same frame three times), seed 1. A gain under 8 px would say time helps where pixels run out; no gain would close the arm.
* **What was done.** `E23ctl_s1` finished (its own tables were not in the pasted output). `E23_s1` resumed from epoch 7 after a disconnect and ran to 40 (34 epochs in 1.51 h, 968 iterations an epoch at about 6.8 it/s); then `e12_eval.py E23_s1@last`, `compare_runs.py --match` and `--sweep` against `E23ctl_s1@last`, `e35_metrics.py`. `E25_s1` (RF-DETR Small, 20 epochs, 18,480 train / 3,221 val images) started at 22:07 UTC, about 13.6 min an epoch.
* **Measurements (cell output; files on Drive only).**
  * Matched recall (control at 0.40, E23 at 0.47), E23 minus control, 95% per-video bootstrap: Anti-UAV test recall +0.032 [+0.013, +0.053], precision +0.020 [+0.003, +0.039], background false alarms −20 per 1000 frames [−49, +3]; Anti-UAV val no difference; Halmstad recall −0.026 [−0.067, +0.004].
  * Sweep, Anti-UAV test, recall / background false alarms per 1000 frames, control → E23: 0.40: 0.870 / 55 → 0.911 / 46; 0.55: 0.849 / 33 → 0.887 / 26; 0.70: 0.808 / 21 → 0.829 / 9. Halmstad at 0.55: recall 0.795 → 0.883. Anti-UAV val: recall level, E23 has more false alarms at every threshold (0.55: 7.4 → 13.5).
  * E35, `E23_s1@last`: Anti-UAV test AP 0.874, R@10 0.845, R@30 0.893; `antiuav_day` AP 0.869, R@10 0.901, R@30 0.925; Halmstad AP 0.995. Pooled test recall by width at 0.55: 0-4 px 0.40, 4-8 px 0.68, 8-12 px 0.77, 12-16 px 0.89; `w_min` 12 px on test, 8 px on val.
  * `E25_s1`: 19 of 20 validation tables printed; "Best EMA metric improved" was last printed at epoch 4 (0.4617; epochs 0, 1, 2: 0.4255, 0.4505, 0.4547).
* **What worked.** Resume of a temporal run from `last.pt`. The time arm is better than its control on the held-out test at every threshold, in recall and in false alarms, by more than the ~0.02 two plain seeds differ.
* **What did not.** The question E23 was built for (under 8 px, against the control) cannot be answered from this output: the control's size table was not pasted and neither run folder is local. The RF-DETR validation score has not improved for 14 epochs.
* **Mistakes.** None found. Caution: the E35 numbers of E23 are not comparable one-to-one with the colour runs' (grey input, plain recipe on E14 data); the reference for colour is `E23ctl_s1` against `E18_base_s1`.
* **Lesson and decision.** One seed; no decision. Seeds 2 and 0 are queued. Needed locally: `E23ctl_s1` and `E23_s1` folders (`e12_eval_last.json`, `dets_*.json`) for the per-width paired comparison. For E25, read the final scores before judging the stall: a flat validation mAP after epoch 4 can be the schedule or a true plateau.

**#93 Field test v1 started: the frozen `cuas_v1` bundle on whole real videos (local RTX 4060, 2026-10-22).**
* **Intention.** The user asked to "graduate" the project: test on real, varied footage (trees, night, everything) instead of only training arms, and widened the scope the same day (night and tree backgrounds in scope, 13 §2.4/§6). Measure what the whole program (`python -m cuas` code path) does per clip and per condition: drone declared or not, delay, false declarations per hour.
* **What was done.** Froze `C:\cuas_data\models\cuas_v1\` (`model.json`): `E24_s1@last` (seed 1 = the reference seed, not chosen on test), threshold 0.55 (best F1 on Anti-UAV VAL), track classifier `track_clf_dets_E12_trackonly.pkl`, evidence 5.0. New `experiments/field_test.py` (sets `dut_tracking`, `antiuav_test` (91 visible test videos extracted to `D:\cuas_data\datasets\antiuav_test_video\`, overlay inpainted), `multiview`, or `--folder` of own unlabelled clips; `--sheet` contact sheets; `--report`); `cuas/replay.run` gained `fps=`, `preprocess=`, `on_frame=`. Backgrounds tagged by eye from the contact sheets in `experiments/field_conditions.json` (DUT: 10 sky, 2 cloud, 3 trees, 5 buildings); night set by brightness. Full run of all three sets started in the background (`D:\cuas_data\field_test\run_v1.log`; results `D:\cuas_data\field_test\cuas_v1\`).
* **Measurements so far (8 of 20 DUT clips).** The tracker covers the drone in 85-86% of drone frames (video01, video07), and a box of some class at 0.55 matches it in 63-78%, but a "drone" box only 7-25%; 2 of the 8 clips are declared (video01 after 18.9 s, video08 after 6.1 s); 0 false declarations. Short clips (under ~3 s) cannot be declared by design (evidence needs ~2.3 s of track). video07 (large drone, 1.4 min) was never declared: 5 of its on-target tracks ended labelled not_drone, and the drone's track broke into many pieces (40 tracks in the clip).
* **What worked.** The runner, labels and scoring line up (checked on contact sheets: DUT boxes, multi-view points and Anti-UAV boxes all sit on the drone).
* **What did not.** The first full launch ran the laptop out of memory on clip 1 (6.6 GB): `UltralyticsDetector` sent every crop of a frame as one batch, and each new batch size grew host memory. Fixed by fixed chunks of 8 crops (`cuas/detector.py`; output unchanged; the clip now stays at 1.67 GB). About 4 frames/s on the 4060: DUT ~1.7 h, Anti-UAV test ~6 h, multi-view longer.
* **Mistakes.** Launching the full run before a full-length clip had been run once (the smoke test used two 3-s clips, too short to show the growth). Rule 17 #47 applies to length as well as code: smoke-test on the longest, most cluttered clip.
* **DUT complete (first scoring), 2026-10-22 evening.** 20 clips, 14 min: drone declared in 8 of 20 (sky 5/10, buildings 3/5, cloud 0/2, trees 0/3; median delay 4.8 s); pooled det recall ("drone" box) 0.40, any box 0.66, track cover 0.83; 1 "false" declaration. **That false one was the labelled drone itself** (video16, checked by eye: the track was on the drone when declared and drifted off later in its 43-s life). Scoring changed: a declaration is judged on the 2 s up to it, drift is counted apart, and every track's per-frame status is saved so `--report` can rescore without re-running. DUT and Anti-UAV were re-run with it (old results in `D:\cuas_data\field_test\_old_scoring\`). Rescored video16: declared after 2.9 s, one drift.
* **More sets (03 §2.4).** A background search pulled ~85 GB of test footage to `D:\cuas_data\datasets\` (catalogue `_search_2026-10-22.md`). Loaders added for `gareud` (night/dusk/trees, labels as explicit absences) and `fred` (test + dusk train sequences, several drones a frame, unlabelled frames unknown); labels checked on contact sheets. `cuas.replay` reads a list of image paths (`ImageList`). Two runs in parallel (the runner is CPU-bound, GPU ~9%): `run_v1.log` (DUT, Anti-UAV, multi-view) and `run_v1_gareud_fred.log` (the FRED train sequences need one more `--set fred` pass: they were added after that run started).
* **Restart with window recording (2026-10-22, ~21:00; user: report the per-condition table, then start on the decision layer).** Every classifier window (t, track age, usable, p, 21 features) is now saved per track (`field_test.Recording` passes the classifier through and keeps its rows). Tracking never depends on the classifier, so `experiments/e40_decision.py` replays any classifier and evidence threshold over the saved tracks offline, with the same scoring; replaying v1's classifier reproduces v1's declarations (0 of 6 clips differ on the first check). All earlier results moved to `_old_scoring\no_windows\`. Three runs: `run_v1.log` (DUT, Anti-UAV test; then `run_v1_multiview.log`), `run_v1_gareud_fred.log`, and `run_v1_train.log`: the TRAINING SOURCES for E40, `halmstad_train` (228 train-split videos, one labelled object each: drone 91, helicopter 49, airplane 47, bird 41) and `antiuav_trainval` (160 + 67 videos extracted to `D:\cuas_data\datasets\antiuav_trainval_video\`, day and night), written to `D:\cuas_data\field_test\cuas_v1_train\` and never into the test report. Detector time dominates on cluttered clips (100-300 ms/frame on DUT, GPU ~9% busy): a speed item for later (18).
* **Lesson and decision (provisional).** On an unseen dataset the system's weak link is not finding the drone but deciding it is one: the track classifier, fitted only on Halmstad tracks, calls DUT drone tracks not_drone, and tracks fragment. Read the full table before deciding; candidates are a classifier fitted on more than one dataset and the detector's class votes as a feature once the detector transfers.

---

## 3. Where Everything Lives

| Artefact | Location |
| :--- | :--- |
| E1 weights, curves, Ultralytics plots | Drive `cuas_runs/E1_halmstad_yolo26s/` (copy in `C:\cuas_data\runs\` for local use) |
| E2 weights etc. | Drive `cuas_runs/E2_zoomtrain_yolo26s/` (same) |
| Evaluation JSONs | `cuas_runs/<run>/cuas_eval_conf*.json` |
| Halmstad raw + YOLO tiles | Colab `/content/halmstad*` (rebuilt each session by cells 4, 6, 11) |
| Anti-UAV300 zip | `%USERPROFILE%\Downloads\Anti-UAV-RGBT.zip` (6.04 GB) |
| Anti-UAV eval frames | `C:\cuas_data\antiuav_val\`, `C:\cuas_data\antiuav_test\` |
| E12 data, weights | `C:\cuas_data\e12\` (tiles, mosaics, `data_e12.yaml`), `C:\cuas_data\runs\E12_multidata_yolo26s\` |
| E12 evaluation | `C:\cuas_data\runs\<run>\reports\eval_<run>.txt`, `<run>\e12_eval.json`, contact sheets in `C:\cuas_data\runs\e12_diag\` |
| **Layout of `C:\cuas_data\runs\` (from 2026-10-06, #80)** | One folder per run (weights, Ultralytics plots, cached detections `dets_*.json`, `e12_eval*.json`) with its own reports in `<run>\reports\` (`eval_`, `cmp_`, `e15_`, `e21_`, `e35_` files and training logs). Cross-run summaries stay at the top (`e35_metrics*.txt`, `e36_*`, `e30_*`, `e33_*`, `e34_*`, `antiuav_test_brightness.json`). After copying Drive downloads into `runs\`, run `python tools/tidy_runs.py --apply`: it unpacks the zips, sorts, never deletes, and parks the zips in `C:\cuas_data\_downloads\` |
| E23 frame stacks | `C:\cuas_data\e23_k3\` (mirrors the single-frame folders; `build_e23.json`, `data_e23.yaml`) |
| DUT train / val (converted), E39 tiles | `C:\cuas_data\dut_train\`, `dut_val\`, `e39\dut_train_tiles\`, `e39\data_e39.yaml`; the downloaded zips are `%USERPROFILE%\Downloads\train.zip`, `val.zip` |
| Colab uploads | `C:\cuas_data\colab\`: `cuas_code.zip`, `cuas_data_e14.zip`, `cuas_data_e22c.zip`, `cuas_data_e23.zip`, `cuas_data_e39.zip` |
| RF-DETR test environment | `C:\cuas_venv_rfdetr\` (layered on `C:\cuas_venv`; 14 §3) |
| Notebook | `experiments/E1_baseline_detector.ipynb`, generated by `experiments/make_e1_notebook.py` |

---

## 4. Practical Lessons (Colab and Local)

* **Colab disconnects wipe `/content`**, not Drive. Always train with `project=` pointing at Drive; resume with `train(resume=True)` on `last.pt`, after rebuilding the data cells.
* **Re-uploading code doesn't reload it.** Restart the session or `importlib.reload` the module.
* **A cell that relies on imports from earlier cells breaks after a restart.** Make diagnostic cells self-contained.
* **`gdown` hits Google Drive quotas on popular files.** Download in the browser, or copy into your own Drive.
* **Large frames must be tiled for evaluation.** Feeding a 1920-px frame to a 640-px model silently shrinks it 3×.
* **Look at the false alarms, not just the counts.** Pictures revealed that most "false alarms" were label errors, which changed the conclusion completely (#12).
* **Long local runs must not let the laptop sleep:** `experiments/train_local.py` blocks idle sleep while it trains. Keep the lid open and the charger in.
* **Datasets can carry burned-in overlays** (Anti-UAV's on-screen text). Always draw predictions on a few frames of a new dataset before trusting its numbers (#19).
* **Verify a new environment against a known result** before using it (`experiments/check_local_repro.py`, #24).
* **Positive-only tiles teach no clutter.** A model that only sees a new dataset's tiles *with* a drone learns that dataset's backgrounds (lamps, buildings) as drone context. Negatives need a safe source (#34).
* **Check the GPU and memory before a long local run** (`nvidia-smi`, free RAM/commit). A local LLM from another project (Ollama) took ~5 GB VRAM and ~18 GB commit and stalled E14's first start (#40). Queue GPU jobs rather than running them side by side.
* **Keep runs comparable:** same seed and schedule; no `cache='ram'`; record the runtime and versions ([14 §4](14_build_log_and_experiments.md#4-experiments)).

---

## 5. Entry Template

```
**#N Title.** Where (Colab T4 / local RTX 4060), versions if changed.
* Data: dataset + build command + counts.
* Config: model, epochs, batch, imgsz, seed, optimizer, augmentation changes.
* Time: iterations/epoch, time/epoch, total; any disconnect/resume.
* Result: key numbers (link the 14 results row).
* Issues and fixes.
* Artefacts: where the weights/results are.
```

From 2026-10-04 (user: "document every step, including the mistakes, where things didn't work, where things did work, our intentions, measurements"), every entry also states, in this order:

```
* Intention: the question, and what result would change what we do.
* What was done: commands, settings, how long.
* Measurements: the numbers, with their source file.
* What worked / what did not: including dead ends and anything that had to be redone.
* Mistakes: what was wrong in the plan, the code or an earlier claim, and how it was noticed.
* Lesson and decision: what changes from here.
```
