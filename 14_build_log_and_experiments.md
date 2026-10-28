# Build Log & Experiments

What has been built, how to run it, and the experiments that turn doc 13's assumptions into measurements. **Record every experiment's result here.** The method and the interpretation of results (capabilities, limitations, threats to validity) are written up in [15](15_detector_training_methods_and_findings.md), and the step-by-step history of every run is in [16](16_training_run_log.md).

> [!NOTE]
> Started 2026-09-23. Milestone 1 (offline detector baseline + motion cueing) is built and tested. E1 (`w_min` ≈ 16 px) and E2 (`w_min` 8 px at conf 0.4 on simulated distance, ~99.5% precise on clean labels) are done (§4, §5). Next: E11, E2 on real small drones (Anti-UAV300).

---

## 1. Scope This Code Serves

Decided 2026-09-23 (reasons in [13 §6](13_system_requirements_and_geometry_budget.md#6-decisions)):
* **Vision only** (no thermal) with **1-2 cameras**. The goal is to see how far a good visible camera plus software can go.
* **The core problem is the "drone eyes":** find drones, and reject birds, aircraft and non-drone decoys.
* **Compute:**
  * RTX 4060 Laptop (8 GB) for training and replay.
  * Google Colab T4 as a fallback trainer.
  * A Raspberry Pi 5 as a future capture node.
* **Private repository, no licence** until publication (the licence consequences are in [13 §5](13_system_requirements_and_geometry_budget.md#5-legal--operational-scope)).
* **Jev deferred** ([12](12_typesafe_jev_system_one_c2_threat_engine.md)). The pipeline doesn't depend on it.

## 2. Code Map

| Path | What it does |
| :--- | :--- |
| `cuas/tiling.py` | 640-px tiles for training (positive tiles at random offsets plus background tiles) and inference (full grid scan; crops around proposals). Never downscales targets |
| `cuas/motion.py` | `MotionCue`: jitter-compensated MOG2 motion proposals for a fixed camera. Optional sky mask. The registration reference **follows the scene** (slow blend of registered frames; restart above 16 px of drift), since 2026-10-01 |
| `cuas/detector.py` | Detector interface (swappable, e.g. for an Apache-2.0 model), Ultralytics adapter, crop-to-frame mapping and NMS |
| `cuas/eval.py` | Drone recall **binned by box width**, `w_min`; false "drone" boxes split into *on a real drone* (duplicates/misplaced), *on a distractor* and *on background* (binned by width); **missed drones that were boxed as another class**; tiled evaluation for large frames (`tile=640`); **per-video (cluster) bootstrap confidence intervals** (`video_of`, `image_counts`, `bootstrap_ci`, added 2026-09-27). CLI: `python -m cuas.eval` |
| `cuas/replay.py`, `cuas/__main__.py` | **The program** (`python -m cuas <video \| camera index \| stream URL>`): motion cueing → crops around movers and every live track (+ a periodic full scan) → detector → `TrackPipeline` → annotated video, per-frame CSV, events file (JSON lines), per-stage timing. A live source is timed by the clock at capture. `--weights blob` uses the simulator's stand-in detector |
| `cuas/datasets/halmstad.py` | Halmstad visible videos → YOLO tile dataset, **split by video** (no frame leakage). Skips frames past the end of a video's labels. Takes **no** background frames by default: unlabelled Halmstad frames can contain drones (`--neg_every` opts in, never from bird videos) |
| `cuas/datasets/antiuav.py` | Anti-UAV300 visible videos → full Full HD frames + YOLO labels for evaluation (every Nth frame with a drone, plus explicit target-absent frames); `--inpaint` removes the burned-in on-screen display. `offset` picks frames between an earlier build. `find_caption` locates the bottom-left caption the overlay mask misses (16 #37). Evaluate with `run_split(..., tile=640)` |
| `cuas/datasets/dut.py` | DUT Anti-UAV detection split (VOC XML assumed; fails loudly otherwise) → Ultralytics split, with size statistics. Held-out test only (E15) |
| `experiments/e11_antiuav.py` | E11: E1/E2 on Anti-UAV native + zoom-out k = 4/8/12/16, tiled, local GPU; writes `C:\cuas_data\runs\E11\e11_summary.json` |
| `experiments/build_e12.py` | Builds the E12 training set: Anti-UAV train frames (inpainted) → positive 640-px tiles + zoom mosaics, plus Halmstad; writes `C:\cuas_data\e12\data_e12.yaml` |
| `experiments/train_local.py` | Local training with the Colab settings (seed 0, 40 epochs, patience 15); blocks Windows idle sleep; `--resume last.pt`; `--arch` (E18 add-ons) and `--seed` (2026-09-30) |
| `cuas/arch.py` | **E18 add-ons** after YOLOBirDrone (01 §2.8; our reading, as the paper has no code): `DeformConv` (AELAN: DCNv1 in the Bottleneck 3×3 convs of layers 4/6/16/19), `MPDA` (P3, layers 4/16) and RMPDA (`MPDA(reverse=True)`, P4, layers 6/19), `ResidualECA` (cheap control). **Each starts as the identity** (zero offsets; residual with a zero-initialised scale) and keeps every original parameter name, so `yolo26s.pt` transfers completely. The spec travels in the checkpoint (`yaml['cuas_arch']`); `ArchTrainer` rebuilds it on fresh start and resume. `DeformConv` is deliberately not a `Conv` subclass, so `fuse()` cannot fold away its offsets. `register_onnx_deform()` maps it to ONNX `DeformConv` (opset 19) |
| `tests/test_arch.py` | 10 tests for `cuas/arch.py` (GPU venv only; skipped elsewhere): identity at initialisation, add-ons active once trained, full pretrained transfer, checkpoint round trip and resume, `fuse()` safety, gradients reach the zero-initialised gates, the exporter's `forward_split` path, spec parsing, `ArchTrainer.get_model`, ONNX export + ONNX Runtime parity |
| `experiments/e18_arch_bench.py` | Cost of each add-on on YOLO26s at one 640 tile: parameters, counted FLOPs, GPU fp16 and ONNX Runtime CPU (4 threads) latency, variants interleaved over 3 rounds (shared laptop); writes `C:\cuas_data\runs\e18_bench.json` |
| `experiments/colab_e18.py` | **One Colab cell** for E18, after `colab_run_all.py`: add-on tests on Colab's torch, a 1-epoch GPU smoke run, then 5 runs in priority order (base seed 1, full, attention only, deformable only, ECA), each evaluated (`e12_eval.py`) and compared pairwise with `E14_hardneg_yolo26s_L4`; E15 on DUT for all (`e15_e18.json`). Resumable (`DONE_E18_*` markers) |
| `experiments/e12_eval.py` | Chooses the operating threshold on Anti-UAV **val** by F1, then scores Anti-UAV val, the held-out **test** and Halmstad val; 95% per-video intervals on native frames; writes `<run>\e12_eval.json` and `<run>\dets_native.json` |
| `experiments/build_e14.py` | E14 data: mines E12's false alarms on Anti-UAV train frames (+ extra frames at offset 13, every 20th), drops caption / < 28-px / drone-cutting cases, caps per sequence and at 15%, writes negative tiles, contact sheets and `C:\cuas_data\e14\data_e14.yaml` |
| `experiments/run_e14_queue.sh` | Waits for a free GPU (< 1,500 MiB used), then E12 re-evaluation and E14 training; log `C:\cuas_data\runs\e14_queue.log` |
| `experiments/compare_runs.py` | Paired per-video bootstrap of run B − run A on native Anti-UAV val/test and Halmstad val frames (needs `e12_eval.py` run on both) |
| `experiments/e15_dut.py` | E15: runs on DUT Anti-UAV test at their Anti-UAV-val threshold; tiles large images, pads small ones (never upscales) |
| `experiments/e12_diag.py` | Same-threshold Halmstad comparison E2 vs E12, and contact sheets of background false alarms (`C:\cuas_data\runs\e12_diag\`) |
| `experiments/check_local_repro.py` | **Verifies the local GPU pipeline reproduces the Colab numbers** bin-for-bin (E1/E2 on Halmstad val). Passed exactly on 2026-09-24. Rerun after any environment change |
| `cuas/zoomout.py` | **Simulated distance:** shrinks frames k-fold into k×k mosaics, so recall can be measured below the sizes a dataset contains. Optimistic: shrinking removes noise and blur (§5) |
| `cuas/track.py` | **M2 tracker** (2026-09-28): constant-velocity Kalman filter in image coordinates, ByteTrack-style two-stage association gated on the *predicted* position (Mahalanobis, χ² 99%), timestamp-based life cycle (confirm after 3 hits, coast ≤ 1 s). Keeps each track's detection history for features |
| `cuas/track_features.py` | Per-track features: speed in body lengths/s (roughly range-independent), hover fraction, straightness, turn rate, acceleration, vertical share, size/aspect variation, Lomb-Scargle periodicity of aspect, silhouette area and brightness (wing flap). `silhouette()` = Otsu blob inside a box, the cheap alternative to a segmentation network |
| `cuas/track.py` `IMMFilter` | IMM (doc 09 §3.3): constant-velocity + constant-acceleration models on one shared state, mixed each step; `Tracker(motion="imm")`. Tested: lower error than CV through a hard turn, and the manoeuvre model's probability rises during it |
| `cuas/track_features.py` `ttc_expansion` | Time to collision from optical expansion: a straight-line fit of **1/width** over the last 2 s (exact for constant closing speed), reported only if the slope is 4 standard errors from zero and the boxes span 0.5 s (2026-10-01; before that 75% of non-closing tracks read under 10 s) |
| `cuas/classify.py` | `TrackClassifier`: gradient-boosted trees on named feature groups (`TRACK`, `KINEMATIC`, `SHAPE`, `DETECTOR`), **isotonic calibration cross-fitted on out-of-fold scores** (folds grouped by video), and `decide()` → drone / not_drone / **unknown** (track < 1 s, p in 0.3-0.7, or > 30% of features outside the training 1st-99th percentile). `ece()`. Save/load |
| `cuas/rules.py` | The doc 12 §4 rules floor without Jev (`rules_floor`, `combine`). The classifier's label wins over p; `drone_prob = None` or label "unknown" is *not yet classified*: if closing, it gets `slew_ptz_for_identification`. Tested on doc 12's cases (spoofed Remote ID inside/outside the zone, decide-now, loitering) |
| `cuas/pipeline.py` | `TrackPipeline.update(t, dets, gray)`: class-agnostic NMS → tracker → silhouettes → features every 5 updates over the last 2 s → calibrated p(drone) and label → TTC → rules response, per confirmed track. Bounded memory (2.5 s of history per track). Optional `decision=` (E26): the label comes from accumulated evidence instead of the latest window; each state also carries `window_label`, `p_fresh`, `p_usable` |
| `cuas/decide.py` | **Sequential track decision** (E26, 2026-10-05): `SequentialDecision` sums each reading's log-likelihood ratio (calibrated p against the classifier's training prior), weighted by the share of its window that is new so overlapping windows are not counted twice; declares drone / not_drone at the accept / reject bounds, clamped so a track can change its mind |
| `cuas/sim.py` | **Scene simulator** (2026-10-01): 3-D paths → pinhole camera → targets drawn at true pixel size on a sky → blur, mount shake, sensor noise; ground truth per frame. `from_px4_csv` reads a PX4 `vehicle_local_position` log. `BlobDetector`: a stand-in with the detector interface (dark blobs by black top-hat). For tracker / TTC / rules / program tests, **not** for detector training or scoring (19 §6) |
| `cuas/degrade.py` | **Training-time degradation** (E22, 2026-10-02): `Degrade` applies Gaussian blur (p 0.4, σ 0.3-1.5), white noise (p 0.5, σ 1-8) and JPEG (p 0.4, quality 30-95) to a training image, about 1 in 5 left clean; `install()` appends it to the Ultralytics training pipeline after the geometric augmentation, so the noise is at the final pixel scale (also when the pipeline is rebuilt as mosaic closes). No new dependency |
| `cuas/paste.py` | **Cut and paste a real drone** (E22c, 2026-10-04): `cut` makes a soft matte of a drone filmed against a plain background (refuses clutter); `paste` composites it elsewhere with the edge pixels un-mixed from the old background, and refuses spots where it would not stand out or whose brightness is far from where it was filmed |
| `experiments/e20_system_eval.py` | **E20, system level:** the pipeline over held-out Halmstad videos from cached detections; false drone declarations per hour on non-drone videos and seconds to declare a real drone, for detector votes / track only / both, with hold times |
| `experiments/e26_sequential.py` | **E26:** the sequential decision against the hold rule on the same tracks, over a grid of settings; operating points chosen on cross-fitted TRAIN videos, scored once on val; checks that its hold-rule replay reproduces E20 |
| `experiments/e21_degradation.py` | **E21:** the detector on real evaluation frames degraded by noise, blur, JPEG and haze, one at a time and combined; plus four **held-out** kinds E22 never trains on (motion blur, signal-dependent noise, WebP, and the three together). Takes `<run>@last` |
| `experiments/soup.py` | **E31:** averages the final weights of several runs of one recipe into one model (`<runs>/<name>/weights/last.pt`), scored as `<name>@last` |
| `experiments/e30_size_calibration.py`, `e33_center_match.py`, `e34_any_class.py` | **CPU checks on cached detections** (2026-10-03): per-size score calibration; centre matching for drones under 12 px; candidates from boxes of any class. E33 and E34 cache their own detections (`dets_small`, `dets_dut`) by CPU inference |
| `experiments/e35_metrics.py` | **Threshold-free scores and error measures** from cached detections (2026-10-04): drone AP, recall at a fixed false-alarm budget, ECE / Brier of the raw score, width and centre error (bias, MAE, RMSE) of correct boxes by size. Run after every training run. `--dut` adds DUT (infers once, caches); the daytime Anti-UAV videos are scored as `antiuav_day` (2026-10-05) |
| `experiments/e37_dut_sky.py` | **E37:** E35's threshold-free DUT scores split into images whose drones are all on sky and the rest (2026-10-06) |
| `experiments/e36_false_alarm_map.py` | **E36:** background false drones by size, sky/clutter and score for plain / E22a / E22c (three seeds), at 0.40 and at an equal count; contact sheets of the most confident ones (2026-10-05) |
| `experiments/build_e22c.py` | **E22c data:** Anti-UAV drones pasted into Halmstad frames and Halmstad drones into Anti-UAV tiles (3,000 tiles), `data_e22c.yaml`, a contact sheet. `pack_for_colab.py --e22c` zips them |
| `experiments/make_demo.py` | **Progress video** (2026-10-04): the whole program on three held-out videos, then plain against degraded-trained weights on a degraded video. Writes `C:\cuas_data\demo\progress_demo.mp4` |
| `cuas/datasets/dvb.py` | Drone-vs-Bird annotation reader, from the documented format (frame base and annotated classes unverified until the data arrives) |
| `experiments/m2_detect_videos.py` | GPU (Colab): runs a detector on every frame of the 285 Halmstad videos, saves detections (conf ≥ 0.05) to `dets_<run>.json.gz` |
| `experiments/colab_run_all.py` | **One Colab cell** (L4): setup, smoke test, E14 training (resumes after a disconnect), E14 evaluation, then E12 and E14 over the Halmstad videos for M2. Replaces `colab_e14_cells.md` |
| `experiments/m2_halmstad_tracks.py` | M2 experiments (CPU). `extract` from oracle boxes (default) or real detections (`--dets`); `classify --tag`: feature-group comparison, 5-fold grouped CV and (with detections) train-videos → held-out val videos, ECE before/after calibration, unknown rate; saves the pipeline classifier `track_clf_<tag>.pkl`. First version: ground-truth boxes with jitter/dropouts → tracker → features per 1/2/4 s window and whole track, at 30 and 10 fps → drone-vs-not-drone classifier, 5-fold grouped by video, AUC with per-video bootstrap, ECE, and a size/position confound probe. Outputs `C:\cuas_data\m2\` |
| `experiments/pack_for_colab.py`, `experiments/colab_e14_cells.md` | Zips E14's training + evaluation data (1.1 GB) and the code for Google Drive; paste-ready Colab cells (L4) for the E14 smoke test, run, resume and evaluation. The GPU scripts read an optional `CUAS_DATA` environment variable (default `C:/cuas_data`) |
| `tests/test_cuas.py` | 51 tests (numpy + OpenCV, scipy, scikit-learn; ~60 s): tiling, eval, crop mapping, motion cueing on synthetic jittering and changing skies, zoom-out mosaics, converters, per-video bootstrap, tracker (incl. confirmed-before-tentative), IMM, TTC (incl. box jitter), classifier, rules, pipeline, the simulator, and **the whole program on a simulated scene** |
| `experiments/E1_baseline_detector.ipynb` | Colab notebook, 16 numbered cells: installs, code upload, dataset, sanity check, conversion + zoom-out copies, smoke test, E1 run + eval (9-10), E2 data/run/eval (11-13), diagnostics (14 per-set breakdown, 15 false-alarm picture grid, 16 'empty'-image check, superseded: the rebuilt dataset has no 'empty' images, and the clean re-eval answered it). **Generated** by `experiments/make_e1_notebook.py`; edit that, not the .ipynb |
| `experiments/cuas_code.zip` | The `cuas/` package zipped for Colab upload (git-ignored; rebuild after code changes, §3) |

Box convention everywhere: `(x, y, w, h[, cls[, score]])` in pixels, top-left origin. Class 0 is always `drone`.

## 3. Setup

**The project folder** is `<path-to>\.Drone Detection` (moved from C: on 2026-09-25). **Data lives outside the project folder** and outside git: use `C:\cuas_data\` locally, `/content/` on Colab, and Google Drive for results. C: had ~28 GB free on 2026-09-23.

**Local GPU environment.** The global Python has a CPU-only PyTorch, so use a separate venv (~3 GB). Run in PowerShell:
```powershell
python -m venv C:\cuas_venv
C:\cuas_venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
C:\cuas_venv\Scripts\python -m pip install ultralytics mcos-decoder
C:\cuas_venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
**RF-DETR (E25) test environment, 2026-10-08.** RF-DETR's training extras would change numpy and add a headless OpenCV, so they live in a second venv that reads the GPU venv's packages through a `.pth` file (16 #85). `rfdetr` itself and `gdown` are installed in `C:\cuas_venv`.
```powershell
C:\cuas_venv\Scripts\python -m pip install rfdetr gdown
C:\cuas_venv\Scripts\python -m venv C:\cuas_venv_rfdetr
Set-Content C:\cuas_venv_rfdetr\Lib\site-packages\cuas_venv.pth 'C:\cuas_venv\Lib\site-packages'
C:\cuas_venv_rfdetr\Scripts\python -m pip install --no-deps accelerate aiohappyeyeballs aiohttp aiosignal attrs faster-coco-eval filetype frozenlist hotcoco lightning-utilities multidict peft pillow-avif-plugin propcache pyarrow pycocotools python-dotenv pytorch-lightning requests-toolbelt roboflow simplejpeg torch-hungarian torchmetrics ultrafast-pycocotools vernier yarl
C:\cuas_venv_rfdetr\Scripts\python experiments/train_rfdetr.py --data C:/cuas_data/e22c/data_e22c.yaml --name E25_smoke --epochs 1 --fraction 0.02 --workers 0
```

(The cu130 wheels are torch 2.14 for Python 3.12 on Windows, checked 2026-09-23. They need an NVIDIA driver new enough for CUDA 13; 616.56 is installed.)

**Colab:** open `experiments/E1_baseline_detector.ipynb` (or paste its cells in order) with a T4 runtime. Cell 2 uploads `experiments/cuas_code.zip`. Results and weights go to `MyDrive/cuas_runs/`, which survives disconnects. Run the **smoke test** (cell 8, 2 epochs on 10% of the data) before the full run. **Rebuild the zip after any change to `cuas/`** (it's git-ignored):
```powershell
python -c "import shutil; shutil.make_archive('experiments/cuas_code', 'zip', '.', 'cuas')"
```

**Run the same things locally:**
```powershell
git clone --depth 1 https://github.com/DroneDetectionThesis/Drone-detection-dataset C:\cuas_data\halmstad
C:\cuas_venv\Scripts\python -m cuas.datasets.halmstad --src C:\cuas_data\halmstad\Data\Video_V --out C:\cuas_data\halmstad_yolo
C:\cuas_venv\Scripts\yolo train model=yolo26s.pt data=C:\cuas_data\halmstad_yolo\data.yaml imgsz=640 epochs=60 batch=-1 seed=0 project=C:\cuas_data\runs name=E1_halmstad_yolo26s
C:\cuas_venv\Scripts\python -m cuas.eval --weights C:\cuas_data\runs\E1_halmstad_yolo26s\weights\best.pt --split C:\cuas_data\halmstad_yolo\val --json C:\cuas_data\runs\E1_eval.json
C:\cuas_venv\Scripts\python -m cuas.replay some_clip.mp4 --weights C:\cuas_data\runs\E1_halmstad_yolo26s\weights\best.pt --out annotated.mp4
```
`batch=-1` lets Ultralytics pick the largest batch that fits in 8 GB.

## 4. Experiments

Run them roughly in order, changing **one thing at a time** against the E1 baseline. **Comparability rule:** runtime changes (Colab High-RAM with more CPU cores, local GPU) only alter speed and augmentation order. Never use `cache='ram'` for compared runs (Ultralytics flags it as non-deterministic; `cache='disk'` is fine). Record the runtime in the results row, and confirm differences of a few points with 2-3 seeds. Report per-bin recall, `w_min` and false "drone" counts, not just mAP.

| ID | Question | What to change | Measure | Where | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E1** | Baseline: how small a drone does YOLO26s find, and how often are birds called drones? | Nothing: trained at native size only, default augmentation, 40 epochs, every 10th labelled frame (after a 2-epoch smoke test). Evaluated on native val **plus** zoom-out copies (k = 2, 3, 4, 6) | Recall per width bin, `w_min`, false drones on birds/aircraft/background at conf 0.1-0.55 | Colab | **Done** 2026-09-24 (results below) |
| E2 | Does training on simulated distance teach small targets? | Add zoom-out mosaics of the **train** split (k = 2-6) to training: 6,829 native + 1,707/758/426/189 mosaics = **9,909 images**. Same schedule and eval as E1 | Recall in the 4-16 px bins vs E1 | Colab | **Done** 2026-09-24: `w_min` 16 → 4-8 px |
| E2c | How much did the contaminated "background" images cost? | Rebuild the dataset with the fixed converter (no unlabelled frames as background) and retrain E2 | Recall by width vs E2; background FA on a val set without mislabelled "empty" images | Colab | Planned. Or fold into E12's retraining to save a run |
| E2b | Does less scale augmentation help? | `scale=0.2` (the default 0.5 shrinks images), then `mosaic=0` | Recall in the 4-16 px bins | Colab | Planned |
| E3 | What does the Pi-sized model cost? | `yolo26n.pt` vs `yolo26s.pt` | Recall per bin vs ms per crop (PC, then Pi 5 with NCNN) | Local + Pi | Planned. **Now also a hardware decision** ([18 §4.3](18_hardware_optimisation_and_realtime_latency.md#43-what-the-comparison-says)): an Orin Nano Super fits full-res scanning with n, likely not with s |
| E16 | How fast is the detector really on the laptop, and what do TensorRT and batching buy? | Benchmark YOLO26n/s: PyTorch vs TensorRT FP16, batch 1 vs K crops, 640 vs 320 input, `nms=False`. Also `cuas.replay` per-stage timing on a real clip | ms per tile and per frame, tiles/s, jitter (p50/p95) | Local, GPU free | Planned ([18 §8](18_hardware_optimisation_and_realtime_latency.md#8-recommended-path-measure-before-buying)) |
| E17 | Can small proposals use 320-px crops? | Evaluate on 320-px crops centred on the labelled drones vs 640 (same weights; targets never rescaled) | Recall by width, $w_{\min}$, ms per crop | Local | Planned: ~2-4× less compute per crop if recall holds |
| E4 | Is the NMS-free head worth it for small targets? | Predict with `nms=False` | Recall in small bins, latency | Local | Planned |
| E5 | How much does compression cost? | Re-encode val tiles at JPEG q95/75/50 and H.264 bitrates | `w_min` shift. Decides how the Pi records | Local | Planned |
| E6 | Motion cueing on real sky | `--var_threshold` 8/16/32, `--scale` 1/0.5, `refine` on/off, sky mask | Share of drone frames with a proposal nearby; proposals per frame; ms per frame | Local, own footage | Planned |
| E7 | P2 head (stride 4) | Only if E1-E2 miss targets under ~12 px. Check whether Ultralytics ships a YOLO26 P2 config (it did for YOLOv8) | Recall below 12 px vs cost | Colab | Conditional |
| **E18** | Do YOLOBirDrone's architecture changes help OUR problem (small targets, clutter)? | `cuas/arch.py` add-ons on the E14 recipe: base seed 1 (noise), full (AELAN+MPDA+RMPDA), attention only, deformable only, ECA; `experiments/colab_e18.py` | Anti-UAV test recall/precision and $w_{\min}$ with per-video intervals; **paired difference vs E14, set against the seed-to-seed difference**; E15 on DUT; cost (`e18_arch_bench.py`) | Colab L4, after E14 | **Done 2026-10-02: no add-on beats a second seed of the plain model** (19 §3.5, §3.6) |
| **E19** | How small a REAL drone do E12/E14 find? | Evaluate on held-out Drone-vs-Bird videos (our video-level split; drones-only labels), no training on it | Recall by width and $w_{\min}$ on real targets vs the zoom-out estimate; false drones per frame (includes unlabelled birds, so audit a sample) | Colab | Waiting for the videos |
| **E20** | What does the whole system do on whole videos? | `experiments/e20_system_eval.py`: detections → tracker → classifier → label over held-out Halmstad videos | False drone declarations per hour on non-drone videos; seconds to declare a drone | Local CPU | **Done 2026-10-01** (19 §3.3). This is E8 measured in-domain |
| **E21** | How much sensor noise, blur and compression can the detector take? | `experiments/e21_degradation.py`: degrade real evaluation frames | Recall and background FAs per condition | Local GPU | **Done 2026-10-01** (19 §3.2): at ~8.5 px, 0.93 → 0.08 under blur + noise + JPEG. Covers E5's JPEG part |
| **E22** | Does training on degraded images make small-target recall survive a real sensor? | **a:** E14 recipe + `--degrade` (blur, noise, JPEG on every training image, `cuas/degrade.py`). **b:** a + `--mosaic 0.1 --scale 0.3`. Three seeds each, 40 epochs, no early stopping, final weights, paired by seed with plain runs. c (pasted degraded drone crops) not built | Standard evaluation, DUT, and E21 with its held-out degradations | Colab | **a: done 2026-10-04, adopted** (~8.5 px under degradation 0.12 → 0.32, held out 0.01 → 0.10; clean test +0.02; 15 §6.14). b: one seed in, two running |
| **E31** | Can the seeds of one recipe be averaged into one model? | `experiments/soup.py` | The soup against its members | Colab (scoring only) | **Shelved 2026-10-03.** The seed soup found no drones: the drone row of the class layer starts from a seed-dependent random value, so the members did not share a start. The one valid soup (E12 + E14, both seed 0) was worse than its members (19 §3.8) |
| **E22c** | Is the drone class tied to the scenes it was filmed in? | E22a recipe + 3,000 tiles with a real drone pasted in from the other dataset (`build_e22c.py`), three seeds | DUT AP and recall at a fixed false-alarm budget against E22a of the same seed; standard evaluation; E21 | Colab | **Done, three seeds; adopted as the base recipe 2026-10-05** (16 #72, #75, 15 §6.17): DUT AP 0.195 against E22a 0.122 and plain 0.193; in-domain AP unchanged; at 30 false drones per 1000 DUT images below plain in every seed (0.099 against 0.124) |
| **E32** | Video with range truth over varied terrain: PX4 flight paths driving real drone crops pasted on real target-free background video | `cuas.sim.from_px4_csv` + `Camera.project` + a paste-and-degrade builder shared with E22c | Time to collision and range against truth; tracker and rules on known paths | Local CPU + WSL2 for PX4 | Planned (19 §6) |
| **E30** | Does a per-size calibration of detector scores make one threshold work across datasets? | `experiments/e30_size_calibration.py` on cached native detections | Recall on Halmstad and Anti-UAV test at a threshold chosen on Anti-UAV val | Local CPU | **Done 2026-10-03: no** (19 §3.10) |
| **E33** | Is small-target recall an artefact of IoU matching at a few pixels? | `experiments/e33_center_match.py`; `evaluate(center_below=12)` | Recall by width under both rules | Local CPU | **Done 2026-10-03: only below 4 px** (19 §3.10) |
| **E34** | What does it cost to let a box of any class start a track? | `experiments/e34_any_class.py` | Drones reached and candidates on nothing labelled, drone boxes against any class | Local CPU | **Done 2026-10-03** (19 §3.10) |
| **E23** | Does temporal input help below 8 px? | Detector input = registered grey frames t−6, t−3, t as the three channels (`cuas/temporal.py`, `experiments/build_e23.py`, `train_local.py --temporal`); control **E23ctl** = frame t in all three channels | Recall under 8 px and false alarms, E23 against E23ctl (time) and E23ctl against plain (colour), same images | Colab | **Built 2026-10-08, queued** (16 #83, #84) |
| **E24** | Does keeping the COCO backbone help out of distribution? | `train_local.py --freeze 10` | DUT recall (E15) and the standard evaluation | Colab | **Done 2026-10-06: adopted provisionally** (15 §6.19); E24b/c/d queued |
| **E25** | Does a foundation backbone generalise better? | RF-DETR Small (Apache-2.0, DINOv2 backbone, patch 16) at 640 on E22c's data: `experiments/train_rfdetr.py`, `cuas.detector.RFDETRDetector` | DUT AP (pooled and sky), daytime Anti-UAV R@10, `w_min`, against E24 and E22c of the same seed | Colab | **Built 2026-10-08, queued** (16 #85) |
| **E39** | How much does DUT's own train split add? | E24 recipe + 5,211 DUT train tiles (`experiments/build_e39.py`). Runs last: it ends DUT test's role as the unseen dataset | DUT test AP against E24 of the same seed; in-domain scores | Colab | **Built 2026-10-08, queued last** (16 #86) |
| **E27** | Does an Objects365-pretrained start generalise better? | `train_local.py --model yolo26s-objv1-150.pt` (the official s model itself started there; 19 §9) | DUT recall (E15) and the standard evaluation | Colab | Planned |
| **E28** | Do large, close drones need a second, coarser look? | `experiments/e28_multiscale_dut.py`: native tiles, plus the frame shrunk 2×, plus the whole frame in one tile; same weights | DUT recall by box width, false drones | Local GPU | **Done 2026-10-01** (19 §10.3) |
| **E29** | Are unlabelled objects in the training tiles being taught as background? | Run the detector over its own training tiles, review confident boxes that touch no label on contact sheets, add the real ones as labels, retrain | Standard evaluation; DUT | Local + Colab | **Not scheduled:** the count found 25 candidate boxes in 10,272 training tiles, mostly clutter (19 §10.2) |
| **E30** | Does a per-size calibration of detector scores make one threshold work across sizes? | Histogram-binning calibration per width bin on validation detections; no training | Recall at a fixed false-alarm count, per width bin and per dataset | Local CPU | Planned |
| **E26** | Can the track decision cut false declarations at the same delay? | Sequential test on the calibrated probability instead of "first drone window"; clutter tracks as classifier negatives | E20's table | Local CPU | **Done 2026-10-05, adopted** (16 #74, 15 §6.9): beats the hold rule at every matched rate; program default `--evidence 5.0` |
| E8 | Bird rejection from track history | Needs milestone 2 (tracker + track features) | False drones per hour on bird-only footage at fixed drone recall | Local | Milestone 2 |
| E10 | Optimizer | `optimizer='MuSGD', lr0=0.01` (YOLO26's own optimizer, which auto only picks above 10k iterations) vs the auto AdamW (lr 0.00125). One change against the best data setup | Same eval, with per-video confidence intervals | Colab | Low priority |
| **E11** | Do the Halmstad-trained models generalise to a **different dataset**? (Anti-UAV300 turned out to have large drones only, so it is not a real-small-drone test) | No training. **E11a:** E1/E2 on Anti-UAV300 visible val/test at native size (overlay inpainted, tiled at 640). **E11b:** the same frames shrunk into zoom-out mosaics, k = 4/8/12/16 | Recall by width, `w_min`, background false alarms (incl. explicit target-absent frames) | **Local RTX 4060** | **Done** 2026-09-25: class failure on the new dataset (results below) |
| **E13** | Does the class head swallow drones (E15: 38% of in-scope DUT drones called helicopter/airplane)? | **Drone-only training** (`train_local.py --classes 0`, Ultralytics' class filter): bird/airplane/helicopter labels become background, so nothing can win over "drone"; the M2 track classifier separates the rest. Helicopter is not a COCO class: the confusion comes from Halmstad's own helicopter class, so removing the COCO prior (`cls_remap=False`) alone would not address it | DUT sky recall (`e15_dut.py`), Anti-UAV test `--match` vs E14, **false drones on labelled birds/aircraft** (the cost) | Colab, first run in `colab_e18.py` | Ready 2026-10-01 (filter verified: an airplane tile's label is dropped, a drone tile's kept) |
| **E14** | Do hard negatives remove the clutter false alarms? | E12 data plus **mined hard negatives**: run E12 on Anti-UAV **train** frames and take 640-px tiles around its background false alarms as empty-label tiles. Check a sample by eye first; drop small (< 24 px) blobs on dark sky, which may be real drones. Cap negatives at ~15% of training images. Same schedule; one change vs E12.<br>**As built (2026-09-27, `experiments/build_e14.py`):** mine at conf ≥ 0.25; drop false alarms on the burned-in bottom-left caption (a dataset artefact, ~60% of the first pass, 16 #37) and those < 28 px; ≤ 40 tiles per sequence; mining pool = E12 train frames + extra train frames (every 20th from offset 13, mining only) | Anti-UAV test background FAs (1,568) and precision (0.78) at the val-chosen threshold, with **paired per-video bootstrap** vs E12 (`compare_runs.py`); recall must not drop; Halmstad val | Local → **Colab L4** | Data built (1,004 negative tiles, 6.5% of training; 16 #36, #40). The local run was paused at epoch 10 (16 #42) and is **superseded**: E14 is retrained fresh on a Colab L4 as `E14_hardneg_yolo26s_L4` (`experiments/colab_e14_cells.md`, 16 #44) |
| **E15** | Does it generalise to a dataset never trained on? | Evaluate E12/E14 on **DUT Anti-UAV** (detection images, Apache-2.0, Google Drive), held out entirely (03 §2.1 rule 3). The user downloads it in the browser; check its size statistics first. **Code ready (2026-09-27):** `cuas/datasets/dut.py` + `experiments/e15_dut.py` (test split 2,200 images per the paper; small images padded, never upscaled) | Drone recall by width, boxed-as-other, background FAs | Local → **Colab** | **Data ready 2026-09-29**: test split converted (`C:\cuas_data\dut_test`: 2,200 images, 2,245 drones, all class `UAV`, VOC XML as assumed; widths min/p10/median/p90/max 12/26/55/260/865 px; images 640-1920 px wide). Runs as stage 6 of `experiments/colab_run_all.py` |
| **E12** | Does multi-dataset training fix the drone class? | `experiments/build_e12.py` then `experiments/train_local.py`, following the combining rules in [03 §2.1](03_datasets_and_benchmarks_catalog.md#21-multi-dataset-strategy-2026-09-24):<br>**Train:** Halmstad train (clean, 6,694) + its zoom mosaics k2/3/4/6; Anti-UAV **train** sequences, every 40th frame, overlay inpainted, **positive 640-px tiles only** + zoom mosaics k2/4/8/16 (~55/28/14/7 px).<br>**Ultralytics val:** Halmstad val + Anti-UAV val tiles.<br>**Anti-UAV test stays held out.** Same schedule as E1/E2 (40 epochs, patience 15, seed 0, AdamW auto).<br>Two changes vs E2 (clean Halmstad labels + Anti-UAV); the label clean-up is small (135 images) | Anti-UAV val/test drone recall by width and boxed-as-other; Halmstad val (must not regress vs E2); background false alarms | **Local RTX 4060** | **Done** (2026-09-25): best epoch 21, early stop at 36; evaluated with `experiments/e12_eval.py` (results row below) |
| E9 | Frame rate vs wing-flap detection | 10 vs 40 fps capture (Pi HQ full-res vs binned) | Flap-feature separability, bird false alarms | Own footage | Milestone 3 |

**Caveats for Halmstad results:**
* The source videos are small, already-compressed frames.
* The small-size bins come from **zoom-out mosaics**, which are cleaner than real distant targets, so `w_min` measured this way is **optimistic**.
* Re-measure on your own camera's footage before trusting it for lens choice.

### Results

| ID | Date | Setup | `w_min` (recall ≥ 0.8) | Drone recall by width | False drones (distractor / background) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| smoke | 2026-09-23 | YOLO26s, 2 epochs, `fraction=0.1` (683 imgs), Colab T4 | - | 0 at every width (expected) | 0 / 0 | Pipeline OK end to end. Eval bins populated: 93 drones at 0-4 px, 920 at 4-8, 828 at 8-12, 450 at 12-16. Ultralytics' own val agrees (drone R = 0). ~60 img/s training → ~2 min/epoch on the full set. Env: Ultralytics 8.4.161, torch 2.11+cu128, Python 3.13 |
| **E12** | 2026-09-25 | Halmstad clean + Anti-UAV train (positive tiles) + zoom mosaics of both; 14,476 train / 3,221 val; YOLO26s, 40 epochs max, patience 15, seed 0; **local RTX 4060**, paused at epoch 10 and resumed ([16 #29, #33](16_training_run_log.md#2026-09-25-e12-multi-dataset-training-local)). Threshold chosen on Anti-UAV val by F1: **conf 0.4** | **Anti-UAV test: 8 px** (4-8 bin 0.79). Halmstad val: 8 px except a 32-64 dip to 0.79 (tool reports 64) | **Anti-UAV test (held out) @ 0.4:** pooled **0.867** (E2: 0.049); 0-4 0.61, 4-8 0.79, 8-12 0.86, 12-16 0.91, 16-24 0.90, 24-32 0.88, 32-64 0.95, 64+ 0.89; native 0.899, k16 0.826. Halmstad val @ 0.4: 0.888 (native 0.848) | Anti-UAV test: 0 distractor / **1,568 background over 2,217 images**, plus 819 on real drones; precision 0.78. Halmstad val: 4 / 19 over 2,550; precision 0.98 | **Drone class now transfers to unseen Anti-UAV videos** (boxed-as-other 5,064 → 13). But this test is new videos of a *training* dataset, not a new dataset. Background false alarms on Anti-UAV are the new main error (see findings). Ultralytics val (mixed, optimistic): drone 0.958 / 0.939 / 0.973 / 0.594; airplane R **0.675** (E1 0.81, E2 0.71) |
| E11 | 2026-09-25 | E1 @ 0.25 / E2 @ 0.4 on **Anti-UAV300 visible** val (1,462 drones) + test (2,009), overlay inpainted, tiled at 640; native + zoom-out k4/8/12/16. **Local RTX 4060**; pipeline verified identical to Colab |, (drone recall never reaches 0.8) | **Drone recall ≤ 4% at every size, both models, both splits.** Test native: E1 0.0%, E2 0.0% | Background: E1 264/370, E2 225/205 (val/test, mostly on mosaics) | **Class failure, not detection:** most drones are boxed but labelled airplane/helicopter (test native: E1 66%, E2 14%). E2 still boxes ~50-60% of shrunken unseen drones at 7-14 px (as airplane); E1 36% at k8, ~0% at k16 |
| E2 | 2026-09-24 | E1 + zoom-out train mosaics (9,909 images), same schedule. Colab T4 standard runtime; **disconnected and resumed at epoch 24**, ran to 40 (no early stop); Ultralytics 8.4.162 | **4 px** at conf 0.1/0.25; **8 px at conf 0.4** (4-8 bin 0.797) | conf 0.25: 0-4 **0.47**, 4-8 **0.86**, 8-12 **0.97**, 12-16 0.96, 16-24 0.99, 24+ 0.99-1.00. conf 0.4: 0.40 / 0.80 / 0.95 / 0.94 / 0.93 / 0.96 / 0.91 / 1.00 | conf 0.25: bird **0**, airplane 2, helicopter 5 / **447** background. conf 0.4: 0 / 2 / 4 / **109**. conf 0.1: 0 / 10 / 7 / 1,113 | Ultralytics val (native): drone P 0.863 / R 0.875 / mAP50 0.873 / mAP50-95 0.359; bird 0.885 / **0.763** / 0.852 / 0.455; airplane 0.784 / **0.713** / 0.776 / 0.423; helicopter 0.941 / 0.90 / 0.916 / 0.351. ~3 min/epoch |
| E1 | 2026-09-24 | YOLO26s, 6,829 train tiles, 40 epochs max, patience 15, Colab T4 | **16 px** (conf 0.1 and 0.25; optimistic, zoom-out) | conf 0.25: 4-8 px **0.03**, 8-12 **0.36**, 12-16 **0.74**, 16-24 0.92, 24-32 0.96, 32+ ≥0.99 (n = 93/920/828/450/479/312/428 drones; 0-4 px: 0/93) | conf 0.25: bird 13, airplane 18, helicopter 5 / **326 background over 2,585 images**. conf 0.1: 24/50/21 / 1,570 | **Ultralytics val on native size (best = epoch 12, early-stopped at 27, 1.12 h):**<br>drone P 0.85 / R 0.875 / mAP50 0.852 / mAP50-95 0.351<br>bird 0.87 / 0.71 / 0.80 / 0.418<br>airplane 0.73 / 0.81 / 0.835 / 0.415<br>helicopter 0.87 / 0.89 / 0.917 / 0.369<br>Train loss kept falling after epoch 12 while val stayed flat (memorising ~230 training videos). Val mAP50 swings ±3-5 points between epochs (~57 val videos) |

## 5. Findings So Far

* **2026-10-22: field test of a frozen model (16 #93).** Scope widened by the user the same day: night and tree backgrounds are in scope.
  * `C:\cuas_data\models\cuas_v1\` = E24 seed 1 final weights + threshold 0.55 (val) + track-only classifier + evidence 5.0. `experiments/field_test.py` runs it over whole videos exactly as `python -m cuas` does and scores, per clip and per condition (sky / cloud / trees / buildings / night): detection recall ("drone" box, any box), track cover, drone declared and its delay, false and unverified declarations (crops saved for an audit by eye). Sets: DUT tracking (20 videos), Anti-UAV test visible videos (91; same dataset as training, unseen videos), multi-view (centre points), or a folder of own clips.
  * First DUT clips: the tracker is on the drone ~85% of the time, but the classifier (Halmstad-only) calls many drone tracks not_drone and tracks fragment; 2 of the first 8 clips declared, 0 false declarations. Full results pending (`D:\cuas_data\field_test\cuas_v1\report.txt`).
  * Fixed on the way: the detector now runs crops in batches of 8 (one batch of every crop grew host memory to 6.6 GB on a cluttered clip).

* **2026-10-01 (night): wider literature sweep** (19 §10, 16 #59).
  * **The architecture is in line with current ground-to-air work** (the closest published model is a YOLO26-style detector with a P2 head); on resolution handling ours is at the strong end.
  * **What is wrong is around it:** clean-only training, no object-level augmentation, IoU matching at 4-8 px, one confidence threshold for all sizes, a thin bird class.
  * **Unlabelled objects in the training tiles are rare:** E12 over its own 10,272 training tiles gives 25 confident "drone" boxes that touch no label (13 Halmstad, 12 Anti-UAV), mostly clutter on the contact sheets. A lower bound (the model trained on those tiles), but too few to act on.
  * **The collapse on an unseen dataset is the field's normal:** the small-bird challenge baseline fell from AP50 46.4 (public test) to 15.4 (private).
  * **E28 (multi-scale inference on DUT, same weights):** recall 0.141 → 0.170, large drones 0.10 → 0.19 (64-128 px) and 0.06 → 0.12 (128-256 px), at about one extra false drone per drone gained. Scale is a small part of the DUT failure; the class is the large part. No coarse pass in the program for now.
  * LRDDv2/v3 (real small drones with range truth) exist but access is on hold; ask the lab.
* **2026-10-01 (late): our recipe against published ones** (19 §9, 16 #58).
  * The official `yolo26s.pt` records its own training arguments: Objects365 pretraining, 70 epochs, patience 100, scale 0.9, copy-paste 0.40. We use package defaults, which differ on almost every line.
  * The 2025 Drone-vs-Bird winner cut mosaic to 0.1 and scale to 0.3, pasted degraded drone patches, and gained +0.19 mAP50 from slicing the frame; our native tiles already go further on the last point.
  * YOLOMG: motion input +0.45 AP, small-object layer +0.02. Our own assigner test agrees: a P2 head doubles positive samples only for 12-16 px targets.
  * Plan changes: E22 gets an augmentation arm, E27 (Objects365 start) added, E23 ahead of E24, E7 moved down, DUT train split into training.
* **2026-10-04 (late): 80 epochs, and E22b over three seeds** (16 #70).
  * **80 epochs against 40 (plain recipe, seed 1):** the same test recall with half the false alarms on the trained datasets (7.9 against 17.2 per 1000 frames at 0.55; "better" at matched recall). No change on DUT (0.140 against 0.138); no better under degradation, and worse under sensor noise at native size. Experiments stay at 40.
  * **E22b not adopted:** over three seeds the same test recall as E22a (0.873 against 0.870) with 1.7 times the false alarms (24.8 against 14.4 per 1000), and DUT inside E22a's seed range. The first seed's DUT 0.170 was seed luck.
* **2026-10-10: E24b/c/d scored; E24 stays the base** (16 #89, #90, 15 §6.19). `e37_dut_sky.py` knows the three new arms; `colab_e18.py` rewrites the DUT table over all runs whenever runs were added (its marker had frozen it).
* **2026-10-08: E23, E25 and E39 built and queued** (16 #83, #87). E23: 18,016 frame stacks (three registered grey frames as channels) rebuilt from the source videos, frame t checked against the existing images (mean absolute difference at most 0.62 grey levels), 197 stacks with a cut or exposure jump reset to static; control set derived on Colab. E25: RF-DETR Small behind `cuas.detector.load` (a marker file in the run folder picks the wrapper); trained and scored end to end locally on a small sample. E39: DUT train and val downloaded and converted, 5,211 train tiles added to E22c's list. All three are in `colab_e18.py` after seed 1 of E24b/c/d; E39 is last.
* **2026-10-06: E24b/c/d queued** (16 #82): `train_local.py --unfreeze-at N` releases the frozen layers from epoch N (LP-FT), smoke-tested on CPU; nine runs added to `colab_e18.py`.
* **2026-10-06: E24 and E27 scored; E24 adopted provisionally** (16 #81, 15 §6.19). Freezing the backbone (`--freeze 10`) on the E22c recipe raises DUT AP from 0.195 to 0.311 (every seed) and, on DUT images whose drones are all on sky (E37), from 0.242 to 0.330, at a cost of daytime Anti-UAV recall at 10 false drones per 1000 frames (0.895 → 0.871). The Objects365 start (E27) is a tie in-domain and mixed on DUT. E37 also shows E22c below plain on sky DUT at strict budgets (R@30 0.125 against 0.190). `e35_metrics.py --out NAME` added; `tools/tidy_runs.py` sorts the runs folder.
* **2026-10-05: field-session plan** ([20](20_field_sessions_own_footage.md), 16 #77). A living plan for filming the user's drone-club drones with their flight logs: checklist, camera setup (no zoom, stabilisation off, focus at infinity, tilted low, originals never re-encoded), shot list S0-S7 (sync and calibration hovers, distance ladder at 30-4 px, approach and recede aimed at least 20 m to the side, crossings, manoeuvres, backgrounds, ≥ 20 min empty sky, two drones), distances per pixel size, the clip log, the processing steps (log reader, sync, camera pose, projected boxes checked by eye), and data rules (each session marked test or train before scoring; the first are test). Session types: A daytime, B terrain, C birds, D dusk and night (future).
* **2026-10-05: E36, what the false drones are** (16 #76, 15 §6.18). Sorted by eye at an equal count: on DUT a quarter are burned-in text and logos and most of the rest buildings and foliage (objects in the sky ~5%); on Anti-UAV, E22c's are mostly objects in the sky (night light streaks, an unlabelled white object) and night lights. Night is out of scope; **by day, at 10 false drones per 1000 frames, E22c finds 0.895 of drones against E22a 0.893 and plain 0.879**, so its in-domain false-alarm cost was night and sky objects.
* **2026-10-05: DUT scored without a threshold for all twelve runs; E22c adopted as the base recipe** (16 #75, 15 §6.17). DUT AP: plain 0.193, E22a 0.122, E22b 0.159, E22c 0.195 (three-seed means). E22c beats E22a in every seed and keeps its in-domain and robustness gains; against plain it is level on DUT AP but lower at a strict false-alarm budget (recall at 30 per 1000: 0.099 against 0.124). At strict budgets every recipe finds about one DUT drone in ten. Queued next on Colab: E27 (Objects365 start) and E24 (frozen backbone), three seeds each on the E22c recipe.
* **2026-10-05: E26, sequential track decision, adopted** (16 #74, 15 §6.9). On the same tracks, declaring by accumulated evidence beats the hold rule at every matched false-declaration rate (track-only classifier, cross-fitted train videos: at ~5 per hour 78 against 68 of 91 drone videos, 0.8 s sooner; val agrees, 18 against 16 of 23). The program now declares at evidence 5.0: about 2.3 s after a track starts instead of 1.0 s, against 145 false declarations per hour for the old first-window rule with that classifier.
* **2026-10-05: E22c, three seeds** (16 #73, 15 §6.17). Beats E22a on DUT in every seed (drone recall 0.126 → 0.203 at 0.40, same precision), in-domain AP unchanged (0.886 against 0.889) with 2-3.5× the low-scoring false alarms at 0.40, robustness unchanged. Against the plain recipe on DUT it is level at one threshold; DUT AP over the nine runs decides.
* **2026-10-05: E22c seed 0 scored** (16 #72, 15 §6.17). The first run to move DUT: drone recall 0.117 → 0.219 at 0.40 against E22a seed 0, mostly sky drones now named "drone" rather than airplane/helicopter, with three times the false drones. In-domain AP and robustness unchanged. On a single threshold it lies on the plain recipe's DUT curve; **DUT AP: plain seed 1 0.190, E22a seed 1 0.090**. `e35_metrics.py --dut` adds DUT to the threshold-free scores; `colab_e18.py` runs it over every run as its last step.
* **2026-10-04: E22c built** (16 #69).
  * **Why:** DUT recall has stayed at 0.12-0.16 through every change. E22c tests one explanation with the data we have: that the drone class is tied to its dataset's scenes. Each dataset's real drones are cut out and pasted into the other's training images.
  * **What looking at the data caught:** night drones (lights on black) pasted into day skies, halos of old sky, and source noise leaking into the new background. All fixed before any training; none would have shown in a loss curve.
  * **Limits:** no new drone models and no new terrain yet; a slight tint on Halmstad drones in Anti-UAV scenes.
  * The Colab queue now runs `E22c_s1/s2/s0` and E35 after every run.
* **2026-10-04: the scoring reviewed; E35** (16 #68, 19 §11).
  * **The validation design holds** (split by video, a held-out dataset, seeds plus per-video bootstrap). **The scoring had three weaknesses:** thresholds chosen by maximum F1 (they flip between 0.40 and 0.55 and confounded comparisons), no threshold-free score, and no measure of box error.
  * **Without a threshold, E22a is still better on clean data:** Anti-UAV test AP 0.853 → 0.889 (seed spread 0.013); recall at 10 false alarms per 1000 frames 0.831 → 0.856.
  * **Halmstad native val is saturated** (AP 0.995 for every run): the reported "Halmstad cost" of E22a was a score shift, not a worse model.
  * **Box width is over-estimated by 25% on drones under 8 px** (5% at 8-16 px), centre error 0.6 px. That is why time to collision reads long at a few pixels.
  * **E22b seed 1 adds nothing** (AP 0.899 against 0.894; same degradation table as E22a).
  * The Anti-UAV test and DUT have been scored after every run; one dataset must stay unscored until a recipe is final.
* **2026-10-04: E22a over three seeds: adopted** (16 #67, 15 §6.14).
  * **Recall under degradation, mean of three seeds, plain → E22a:** ~8.5 px 0.12 → 0.32 (held-out kinds 0.01 → 0.10); ~14 px 0.45 → 0.75 (0.53 → 0.68); ~34 px 0.37 → 0.85 (0.16 → 0.51). The three E22a seeds agree within 0.007 at ~8.5 px.
  * **Clean Anti-UAV test recall +0.02 in all three pairs** at matched validation recall, no clear change in false alarms.
  * **Costs:** Halmstad drones score lower (−0.16 recall at 0.55, −0.02 at 0.40); DUT not helped (0.126 against 0.138).
  * **`--degrade` becomes the base recipe.** E22b (mosaic 0.1 / scale 0.3), first seed: the same recall with 2.4 times the in-domain false alarms; DUT 0.170. Two seeds pending.
* **2026-10-03: three CPU checks; E22a second seed** (16 #66, 19 §3.10).
  * **E30 refuted:** the score gap between datasets is not a size effect (same-width correct boxes score 0.15-0.25 lower on Halmstad). Per-size calibration made Halmstad worse. Halmstad's own best threshold (~0.33) gives recall 0.99 at precision 0.97: set thresholds on site footage.
  * **E33:** matching by centre under 12 px moves recall only below 4 px (+0.12 to +0.22). `w_min` stands; `cuas.eval.evaluate(center_below=...)` is available.
  * **E34:** on the unseen dataset (DUT) boxes of any class reach 2.2-2.6 times as many drones as "drone" boxes (0.353 against 0.138 at 0.55) for 4 times the stray candidates, and still reach more at an equal stray-candidate rate; on the trained datasets it gains no drone and costs 2.5-5 times the candidates. The pipeline already tracks every class; keep it, with a strict track decision.
  * **E22a seed 2** agrees with seed 1 on the clean Anti-UAV test (recall +0.017 at matched validation recall), on lower Halmstad scores, and on the robustness gain (~34 px held-out combination 0.215 → 0.607; ~8.5 px trained kinds 0.111 → 0.320).
  * **E22a on DUT:** seed 1 is worse at every matched threshold as "drone" (0.068 against 0.138 at 0.55); seed 2 is level in recall with 2.3 times the false drones at its lower threshold. No gain on the unseen dataset, and probably a cost.
* **2026-10-03: E22a, first seed** (16 #65, 15 §6.14).
  * **Training on degraded images works for robustness, including on degradations it never saw:** at ~34 px, recall under the held-out combination 0.12 → 0.56; at ~14 px 0.51 → 0.71; at ~8.5 px 0.02 → 0.15 (trained kinds 0.15 → 0.31).
  * Clean Anti-UAV test not harmed. **Costs:** lower Halmstad recall at a given threshold; DUT worse (0.103 against 0.138, 246 background false drones against 81, many of them tiny).
  * One seed, and the two models sit at different thresholds; seeds 2 and 0 are queued.
* **2026-10-03: two seeds agree under the fixed protocol; the seed soup is invalid** (16 #64, 19 §3.1, §3.8).
  * **Plain seeds 1 and 2, 40 epochs, final weights:** Anti-UAV test at 0.55 is 0.859 / 0.784 / 17.2 against 0.843 / 0.780 / 15.2; DUT 0.138 against 0.157; no clear difference in test recall or false alarms at matched recall. The earlier seed spread was mostly early stopping and `best.pt`.
  * **E21 baselines for E22** (~8.5 px): clean 0.82 and 0.80; blur + noise + JPEG 0.15 and 0.11; the held-out combination 0.02 and 0.01.
  * **The seed soup found no drones.** The drone and helicopter rows of the class layer start from a seed-dependent random value ("Remapped 2/4"), so two seeds cannot be averaged as trained. E31 shelved.
  * Eight loader workers: 0.044 h per epoch against about 0.051 with four.
* **2026-10-02: E22 built; seed averaging; simulation and terrain plan** (16 #62, 19 §3.8, §3.9, §6).
  * **E22 is runnable:** `train_local.py --degrade` degrades every training image after the geometric augmentation. A 1-epoch local run trains, the plotted batches show the blur and JPEG blocks, and `--resume` turns it back on from a marker file in the run folder.
  * **E21 gained four held-out degradations** that E22 never trains on, so the gain is not measured only on the generator that made the training data (the E2 mistake, 17 #5).
  * **Seed dependency:** most of the "seed difference" at a shared threshold is the score scale (Halmstad 0.14 against 0.60 at 0.55, but 0.933 against 0.923 at 0.40). Remedies: per-model thresholds, fixed schedule, and E31 (the seeds' weights averaged into one model).
  * **E31, local check:** the final weights of E12 and E14 (two different recipes from the same pretrained weights; the only finished final weights on this machine) were averaged and scored (`eval_E31_soup_E12_E14_last.txt`). The average is a working detector (native Anti-UAV test recall 0.807, precision 0.830 at 0.40; it did not collapse), but **it is worse than either member at matched recall**: at test recall ~0.87 it has 81 background false alarms per 1000 frames against 23 for E14's final weights (+58 [+25, +98]) and 62 for E12's at recall 0.89. Its scores are also lower (Halmstad recall 0.64 at 0.40 against 0.94 and 0.97). So averaging is not free here. This pair differs in training data and in stopping epoch (36 and 28), so it is not the seed test; that is `E31_soup_base_s1s2` on Colab, and the other soups stay switched off until it reports.
  * **Simulation:** valuable for testing (range truth), not for training pixels. E32 = PX4 paths driving real drone crops on real background video.
  * **The Colab queue was rewritten** around the plain seeds, E22 and the soups; the finished E13/E18 runs are no longer rescored.
* **2026-10-02: E13 (drone-only) in-domain** (16 #61). Equal to plain seed 1: Anti-UAV test 0.852 / 0.792 / 12.3 at 0.55 against 0.843 / 0.792 / 12.8; 2 test background boxes at matched recall against 3; Halmstad 0.595 against 0.599. No bird, airplane or helicopter called drone on Halmstad val. Dropping the other classes is free in-domain.
  * **On DUT it is the worst run: recall 0.041, 97% of sky drones missed.** The four-class models box 35-52% of those drones as helicopter or airplane; the drone-only model boxes none of them. The airplane and helicopter heads are what find unseen drones. **Keep four classes** (19 §3.7); the drone class needs more varied drone data.
* **2026-10-02: the three E18 ablations; E18 closed** (16 #60, 19 §3.6).
  * **No add-on beats plain seed 1** on test recall, test false alarms at matched recall, Halmstad or DUT. Attention only is worse (97 background false alarms per 1000 frames at test recall ~0.85 against 13-34 for the others).
  * **The ECA control is called "better" than plain seed 0 as much as any add-on** (test recall +0.032, Halmstad +0.081). Plain seed 0 (E14, stopped at epoch 28) is the weakest of the six runs, so it was the wrong reference; use plain seed 1 until three seeds exist.
  * AELAN alone has the highest DUT recall (0.174), but at threshold 0.40 with 271 background false drones against 92 for plain seed 1 at 0.149.
  * All six runs miss at least 83% of DUT drones and the five E18 runs box 460-760 of them as another class.
* **2026-10-01 (late): `E18_full_s0`, and 40 epochs not 25** (16 #57, 19 §3.4, §3.5).
  * **The full add-on model (AELAN + MPDA + RMPDA) shows no gain.** At each run's threshold (0.55): Anti-UAV test recall 0.791 against 0.762 and 0.843 for the two plain seeds; Halmstad 0.398 against 0.138 and 0.599; DUT 0.106 against 0.076 and 0.149. On Anti-UAV val it has 80 background false alarms at matched recall against 15 and 12. 1.35× the training time per epoch.
  * **Epochs:** the "fixed 25" written earlier the same day was wrong; it leaned on validation-mAP best epochs. Later weights were better on the test at epochs 28 and 36, so the rule is the full 40 with no early stopping, and `E18_base_s1_e80` tests 80.
  * **Tooling:** `<run>@last` scores a run's final weights in `e12_eval.py`, `compare_runs.py` and `e15_dut.py`; `colab_e18.py` now does it for every run.
* **2026-10-01: design review** ([19](19_design_review_parameters_and_novel_directions.md), 16 #54, #56).
  * **Seed noise is as large as several reported effects.** `E18_base_s1` (the E14 recipe, seed 1) differs from E14 "clearly" under the per-video bootstrap: test background FAs −13 per 1000 at matched recall, Halmstad recall at 0.55 of 0.60 against 0.14, DUT recall 0.149 against 0.076. At test recall ~0.90, background FAs per 1000 frames are 133 (E12 best), 62 (E12 last), 159 (E14 seed 0 best), 46 (E14 seed 0 last) and 64 (E14 recipe, seed 1), so E14's hard-negative gain is not established. Three seeds per arm from now on.
  * **The checkpoint is a noise source too.** At matched test recall, `last.pt` has half (E12) to a third (E14) of the background FAs of `best.pt`. Fixed epochs and final weights from E22 on.
  * **E21: the 8-px figure holds only for clean images.** Blur 1.0 + noise 6 + JPEG 40 takes recall at ~8.5 px from 0.93 to 0.08. Training has never shown the detector a noisy small drone.
  * **E20: first system-level numbers.** In-domain, no false drone declaration in 5.9 minutes of non-drone video (upper bound 30 per hour) and a drone declared 1.03 s after it appears. The track-only classifier gives 71 false declarations per hour as a system despite 93% per-window accuracy.
  * **Time to collision was mostly noise on young tracks** (75% of non-closing tracks read under 10 s); now a 1/width fit with a significance test. On real tracks it still reads under 3 s on 1-2 distant aircraft or birds per minute, so it cannot trigger an alert by itself.
  * **16 code findings fixed**, each with a test that fails on the old code; the program runs end to end (`python -m cuas`).
* **2026-10-01: E14 confirmed; E15 (DUT) result** (15 §6.5-6.6, 16 #51).
  * **E14 vs E12 at matched recall** (`compare_runs.py --match`, E14 at 0.21), on test: background FAs −44% [−93, −24 per 1000 frames], precision +0.021, recall −0.023 (all significant). A small real gain, not the 9× seen at own thresholds. On Halmstad, E14 has more background FAs. This refines the first read below.
  * **E15:** E12 recall 0.141 and E14 0.076 on DUT test, verified correct on contact sheets.
    * `e15_dut.py` now splits drones by background (ring texture < 5 = sky, 57% of DUT).
    * Sky: E12 finds 57% under some class, but only 18% as "drone" (38% helicopter/airplane).
    * Clutter (out of scope): 86% missed.
    * E13 (class head) re-prioritised.
  * **M2 on real detector tracks** (`m2_halmstad_tracks.py extract --dets` / `classify`, venv Python; logs `C:\cuas_data\m2\classify_dets_E12.log`, `..._E14L4.log`):
    * The target was tracked in 271/285 videos (E12) and 267/285 (E14). The target track holds a median 100% of labelled frames (10th percentile 65%). Other confirmed tracks: 6.1 per minute (E12), 5.0 (E14).
    * AUC, drone vs not-drone ("held-out" = trained on train videos, tested on the detector's own validation videos):

    | Real E12 detections, 30 fps | 1 s CV | 1 s held-out | 2 s CV | 2 s held-out | whole track CV |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | track (motion + shape) | 0.931 | 0.918 [0.863, 0.961] | 0.948 | 0.933 [0.872, 0.977] | 0.967 |
    | motion only | 0.860 | 0.866 | 0.894 | 0.880 | 0.962 |
    | detector class votes only | 0.998 | 1.000 | 0.995 | 1.000 | 1.000 |
    | **confound probe** (box size + position) | 0.777 | 0.740 | 0.762 | 0.754 | 0.813 |

    * **Track features work on real detector output.** AUC is 0.92-0.93 on held-out videos from 1-2 s, 0.15-0.18 above the size/position probe. This is better than the oracle-box run (0.88 at 1 s), probably because the real boxes jitter less than the 5% oracle noise. E14 gives the same picture (track 0.908 / 0.926 held-out at 1 / 2 s).
    * **Detector class votes are perfect here (AUC 1.000) but only in-domain:** E12 was trained on Halmstad, and E15 showed the class head fails on new data. So `classify` now saves two pipeline classifiers:
      * `track_clf_<tag>.pkl` (detector + track): 0% unknown, 99.7% accurate, leaning on the votes;
      * `track_clf_<tag>_trackonly.pkl`: 12% unknown, **92.6% accurate on the rest**, ECE 0.066 → 0.027. **Use the track-only model outside Halmstad-like footage.**
    * Still to do: the recording-confound test proper (Drone-vs-Bird: drones and birds in the same videos).
* **2026-09-30: E14 result, first read: hard negatives shifted the scores, not the trade-off** (15 §6.5, 16 #50).
  * At its val-chosen threshold (0.55), E14 on Anti-UAV test reaches recall 0.762, precision 0.877, 14.7 background FAs per 1000 frames. E12 at 0.70 reaches 0.757 / 0.878 / 6.4. So the apparent 9× false-alarm cut (against E12 at 0.40) is mostly threshold movement.
  * **Halmstad:** both models' drones score 0.40-0.55 (E12 recall 0.848 at 0.40, 0.310 at 0.55; E14 at 0.55: 0.138). A threshold chosen on one dataset does not transfer.
  * **Tooling:** `compare_runs.py --match` (B at A's native val recall) and `--sweep`; E18 now compares with `--match`.
  * **The Colab M2 stage crashed after all inference:** `json.dump` of numpy float32 box coordinates. `UltralyticsDetector` now returns plain Python numbers; `m2_detect_videos.py` checks serialisability after the first video. Colab failures now print the last 20 output lines (the stage's filter had hidden the error).
* **2026-09-30: Drone-vs-Bird: training set only, drones only** (16 #49).
  * The organisers give the training videos only; the test set and its annotations are withheld. The training annotations are the public GitHub files (77 videos, 106,485 frames, 119,259 drone boxes; `C:\cuas_data\dvb\annotations`). They parse cleanly with `cuas.datasets.dvb`. Frames start at 0.
  * **Birds are not labelled.** For M2, "not drone" tracks are detector tracks matching no drone box, audited by eye on contact sheets.
  * **That is the recording-confound test M2 needed:** drone and non-drone tracks come from the *same* videos, so size, position and recording conditions cannot separate them.
  * **Real small drones:** median width 25 px, 29% under 16 px. This gives $w_{\min}$ on real distant drones (E19), replacing the optimistic zoom-out estimate.
  * Not comparable with published challenge scores (hidden test set). Use our own video-level split.
* **2026-09-30: E18 add-ons built and tested; cost measured** (16 #48).
  * 10 new tests pass in the GPU venv (the 38 + 18 elsewhere are unchanged). Each add-on is exactly the identity at initialisation (full-model output difference 0; patched blocks ≤ 6e-7 relative) and changes its block's output once its parameters move.
  * **End-to-end on CPU** (16 tiles):
    * Fresh start with all three add-ons; simulated disconnect after epoch 1; `train_local.py --resume` in a new process; finish.
    * Reload `best.pt` (4 attention blocks and 7 deformable convs present), predict, and load through `UltralyticsDetector`. All 708 pretrained tensors transfer, plus 154 new ones.
    * The first attempt took **zero optimizer steps**, so nothing had trained. Ultralytics accumulates to a nominal batch of 64, and 12 iterations at batch 4 never reach one step. Rerun with `nbs=4`: the add-ons and the pretrained layers moved.
  * **Deployment:** torchvision ships no ONNX exporter for `deform_conv2d`. `register_onnx_deform()` maps it to the ONNX `DeformConv` operator (opset 19), and ONNX Runtime on CPU matches PyTorch. TensorRT and Hailo support remain unverified.
  * **Cost on YOLO26s, one 640 tile.** Best of 3 interleaved rounds on the shared laptop. Read the ratios: the absolute times are inflated by power saving and a resident local LLM.

    | Variant | Params | Counted GFLOPs | GPU fp16 | ONNX Runtime CPU, 4 threads |
    | :--- | ---: | ---: | ---: | ---: |
    | YOLO26s (base) | 9.47 M | 20.8 | 1.00× (12.4 ms) | 1.00× (106 ms) |
    | + ECA | 9.47 M | 20.8 | 1.06× | 1.04× |
    | + MPDA (P3) | 10.04 M | 28.0 | 1.25× | 1.33× |
    | + RMPDA (P4) | 10.38 M | 23.7 | 1.23× | 1.15× |
    | + MPDA + RMPDA | 10.95 M | 31.0 | 1.51× | 1.41× |
    | + AELAN (deformable) | 9.52 M | 20.2* | 1.33× | 1.13× |
    | Full (the paper's M6) | 11.00 M | 30.3* | 1.78× | 1.57× |

    \* Counted FLOPs omit the deformable sampling itself; the latency includes it.
  * **Meaning for the edge budget (18 §4):** the full set costs ~1.6-1.8× in tiles/s, so it must buy a matching accuracy gain. ECA is nearly free.
* **2026-09-29: M2 verified after the rewrite** (16 #46).
  * All 38 tests pass. The ~180-s classifier test was **OpenMP oversubscription**: one small gradient-boosting fit took 3.1 s with the default 16 threads, 0.09 s with 4 and 0.06 s with 1. `TrackClassifier` and the experiment now use 1 thread (whole suite ~15 s).
  * **Calibration, measured honestly:** the first rerun printed ECE 0.10 → 0.000. That was in-sample: the isotonic calibrator had been fitted on the scores it was scored on. It is now cross-fitted as well. Oracle, 30 fps: ECE 0.086 → 0.018 (1-s windows), 0.093 → 0.034 (2 s), 0.098 → 0.065 (whole track, only 280 samples).
  * AUCs reproduce the first run exactly (1 s 0.876, 2 s 0.898, whole 0.923; 10 fps 0.844 / 0.876 / 0.924). AUC is computed on the raw scores: per-fold calibrators add ties.
  * **Pipeline classifier** (`track_clf_oracle.pkl`, motion + shape, 2-s windows at 30 fps): 23% of windows answered "unknown"; 88.4% accurate on the others.
* **2026-09-28: M2 first result: track motion separates drones from birds and aircraft well, even from short tracks** (`experiments/m2_halmstad_tracks.py`, 16 #43).
  * Setup: Halmstad ground-truth boxes with 5% centre and size jitter and 10% dropped detections → tracker → features → gradient-boosted classifier, 5 folds grouped by video (280 videos, 40% drone). **Oracle detections**, not detector output.
  * Drone vs not-drone AUC at 30 fps (95% interval over videos): **1 s of track 0.876 [0.848, 0.902]; 2 s 0.898; 4 s 0.899; whole ~10 s track 0.923 [0.889, 0.955]** (balanced accuracy 0.79 → 0.84).
  * **Kinematics carry most of it** (whole track 0.908 alone). Shape and flap features alone reach 0.67-0.78.
  * **Frame rate:** at 10 fps the combined result barely changes (whole track 0.924; 1 s 0.844 vs 0.876). But shape/flap alone at 2 s falls from 0.777 [0.739, 0.814] to 0.687 [0.641, 0.731], consistent with doc 18 §3.3 (10 fps can't see flaps above 5 Hz). Motion features make up for it here.
  * **Confound probe:** box size and position alone reach AUC 0.72-0.76, because each class was filmed in its own sessions. The motion model's 0.92 is well above that, but part of the separation may still be recording conditions. The real test is another dataset with birds (Drone-vs-Bird) and own footage.
  * **Not calibrated:** ECE 0.08-0.10. M2 needs a calibration step (isotonic or Platt on held-out folds) before scores feed the rules layer.
  * **Tracker on real labels:** the median video is one unbroken track. The worst 10% (mostly birds) fragment because the Halmstad **label jumps 7-20 box widths in one frame** (the annotator switches between flock members) or pauses up to 3 s. The tracker correctly refuses those jumps.
* **2026-09-28: Masks vs boxes.** No segmentation network: at 8-30 px a mask adds almost nothing over a box, no dataset here has mask labels, and a segmentation head costs extra compute per tile (doc 18). Instead, `silhouette()` thresholds the box against the sky (Otsu) for area and brightness over time, the flap signal, at negligible cost. Reconsider masks only for close-range identification by a zoom camera (M4). Recorded in 11.

* **2026-09-27: Anti-UAV has a second burned-in overlay: a bottom-left caption the overlay mask misses.** Details: [16 #37](16_training_run_log.md#2026-09-27-e14-hard-negative-mining-bootstrap-intervals-e15-preparation).
  * White Chinese text 民航飞机 ("civil aviation aircraft") at x 22-170, y 1006-1041 in night sequences: **train 31 sequences, val 12, test 0**. E12 calls it "drone" (scores up to ~0.74).
  * `overlay_mask` needs a pixel to be an edge in ≥ 30% of all frames, so an overlay present in only some sequences escapes it. **Lesson:** a dataset-wide persistence mask can't find sequence-specific overlays. Look at the false alarms on *every* split, not a sample of one.
  * Impact: E12's test clutter diagnosis stands (test has no caption). The val-chosen threshold counts caption false alarms (same for every model, so comparisons stay fair). It made up ~60% of E12's first-pass mined train false alarms, so it is excluded from E14 mining (`find_caption`).
* **2026-09-27: first E14 mining counts** (16 #36, #38): E12 at conf ≥ 0.25 on the 3,578 train frames gives 966 false alarms. 499 are the caption, 60 are < 24 px, and **407 real ones** remain: lamps and bokeh, buildings, cranes, pylons, haze edges. That is only 328 tiles (2.2%), so the mining pool was enlarged with more frames from the same train videos.

* **2026-09-25: E12: multi-dataset training fixes the drone class. The main error is now clutter false alarms.** Details: [16 #33, #34](16_training_run_log.md#2026-09-25-e12-multi-dataset-training-local).
  * **Held-out Anti-UAV test, conf 0.4 (chosen on val):**
    * Drone recall **0.867**, up from E2's 0.049. Native 0.899; shrunk to ~7 px (k16), 0.826.
    * `w_min` **8 px**; 0-4 px 0.61.
    * Drones boxed as another class: 5,064 → **13**. The COCO airplane prior no longer swallows drones once the drone class has seen varied drones, so E13 is deprioritised.
  * **Caveat:** these are unseen *videos* of a dataset E12 trained on (same cameras and sites), not an unseen dataset. The honest cross-dataset test is still to come (E15, DUT Anti-UAV held out; Drone-vs-Bird later).
  * **Precision on Anti-UAV is only 0.78:** 1,568 background false drones over 2,217 images (271 on the 2,038 native frames), plus 819 duplicate or misplaced boxes on real drones.
    * By eye they are night lights (~40%: lamps, windows, light streaks), ground clutter (~35%: buildings, cranes, pylons), dark foliage or cloud (~15%), and ~10% small grey blobs that may be unlabelled drones.
    * **Cause:** Anti-UAV contributed positive tiles only (03 §2.1 rule 1), so the model never saw its clutter as "nothing". Next is hard-negative mining (E14).
    * The planned sky-only, daytime camera would avoid much of this clutter, but not lamps, cranes or pylons above the skyline.
  * **Halmstad, same threshold 0.4:**
    * Pooled recall is unchanged: E2 0.887, E12 0.888.
    * Native-size recall fell (0.933 → 0.848), and background FAs rose from 4 to 19 (tiny sky specks on mosaics).
    * A mild in-domain cost, which lower thresholds recover (0.952 at 0.25).
  * **Airplane recall keeps declining:** 0.81 (E1) → 0.71 (E2) → 0.675 (E12, Ultralytics val). Airplanes are missed, not called drone (4 distractor FAs on Halmstad val).
  * **Training plateaued from epoch ~6;** best epoch 21 of 36. More epochs would not have helped.
* **2026-09-25: E11: the "drone" class does not generalise to a new dataset.**
  * On Anti-UAV300 (new camera, site, drone models, lighting), both Halmstad-trained models call almost no drones "drone" (≤ 4% at every size, val and test). The local pipeline was first verified to reproduce Colab exactly (`experiments/check_local_repro.py`), so this is not a bug.
  * **The detectors do see the drones:** at native size E1 boxes ~65% of them, but as *airplane* or *helicopter*.
  * **E2's small-target ability transfers as localisation:** on unseen footage shrunk to ~7-14 px, it still boxes ~50-60% of drones, while E1 manages 36% at k8 and ~0 at k16. The classification is what fails.
  * **Likely cause:** "bird" and "airplane" heads started from COCO (broad flying-object priors), while "drone" was learned only from Halmstad's handful of drone models, mostly dark silhouettes. Anti-UAV's bright white quads on grey sky match "airplane" better.
  * **Consequences:**
    1. E2's "8 px / ~450 m" figure holds only **in-domain**.
    2. The drone class needs **many drone types, cameras and skies** in training (E12).
    3. Consider a single "not-drone" class or no COCO class heads, so the airplane prior can't swallow drones (E13).
* **2026-09-24: Anti-UAV300 is not a small-drone test set, and has a burned-in overlay.** Measured from the zip: visible drones median 109-122 px wide, 43 of ~280k boxes under 16 px; 1920×1080 at 20 fps; PTZ-tracked (drones near centre). E11 redefined: **E11a** new camera/site at native size; **E11b** simulated distance on unseen footage (k = 4/8/12/16 → median ~29/14/10/7 px). The on-screen display (status text, reticle) is detected as "airplane" (conf up to 0.8), so it is inpainted before evaluation. The first smoke test also missed a bright white drone on grey sky entirely: a possible appearance/domain gap to watch in the full run. Real small drones must come from Drone-vs-Bird or own footage.
* **2026-09-24: Clean re-evaluation after removing mislabelled "empty" frames.** Dataset rebuilt with the fixed converter: val 1,759 native frames + 439/195/109/48 mosaics = 2,550 images (was 2,585; the 23 unlabelled-drone frames are gone). Cell 15 then found **0 background false alarms on native frames** for E2 @ 0.4, and 2 across all k3 mosaics (one faint 11×9 speck; one box smaller than its label).

  | | E1 @ 0.25 | E2 @ 0.4 |
  | :--- | ---: | ---: |
  | Recall by width 4-8 / 8-12 / 12-16 / 16-24 | 0.03 / 0.38 / 0.72 / 0.91 | **0.79 / 0.96 / 0.92 / 0.93** |
  | Pooled drone recall (3,499 drones) | 0.52 | **0.89** |
  | Extra/misplaced boxes on real drones | 231 | 5 |
  | Distractors called drone | 57 (bird 19, airplane 29, helicopter 9) | 6 (airplane 2, helicopter 4) |
  | Background false drones | 31 | **4** (all 4-12 px) |
  | Correct share of drone calls | ~0.85 | **~0.995** |
  | `w_min` | 16 px | 8 px |

  * **E2's background false alarms were almost entirely unlabelled drones:** 103 → 4 once they're removed. What's left is ~1 per ~2,200 source frames (4 over ≈ 8,800 frame-equivalents).
  * **Revised deployment estimate:** ~0.011 per full 4K scan, i.e. ~80 per hour at a 2 Hz full scan before tracking. **Caveat:** every val frame contains a labelled target, so this is the rate *around real targets on Halmstad skies*, not over hours of empty sky, cloud, birds and treelines. That needs verified-empty footage.
  * E1's background false alarms also fell (96 → 31). Its distractor count moved 36 → 57 when the mosaics were regrouped: small-sample noise, a reminder that differences of a few dozen boxes aren't meaningful yet.
* **2026-09-24: The "background false alarms" are mostly unlabelled drones (visual check, cell 15).**
  * E2 @ 0.4, 12 random native examples: **all 12 are clearly drones** (quad arms and rotors, white Phantom-style bodies) with no label.
  * 12 k3 examples: 11 are drones (9-22 px, plus faint 5-10 px specks); 1 is a box-size mismatch on a labelled object. None sit on mosaic seams.
  * **Cause:** in Halmstad, "no label on this frame" often means *not annotated*, not *no target*. The converter had used such frames as empty background examples: 23 in val, 135 in train.
  * **Consequences:** (1) the measured false-alarm rate is inflated, so E2's real precision is higher than ~96%; (2) the training background images can hold drones labelled as "nothing", which costs recall.
  * **Fix:** `cuas/datasets/halmstad.py` no longer takes unlabelled frames as background (`--neg_every 0` default; opt-in, never from bird videos). True negatives must come from footage verified empty (own recordings).
  * Cell 16 (count false drones on "empty" images) was **superseded** by the rebuild + clean re-eval below and was not run.
* **2026-09-24: Per-set breakdown (cell 14).** E1 @ 0.25 / E2 @ 0.4:

  | Set | E1 recall | E2 recall | E1 on-drone | E2 on-drone | E1 bg FA/img | E2 bg FA/img | E2 bg FA per source frame |
  | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | native | 1.00 | 0.93 | 29 | 0 | 0.017 | 0.012 | 0.012 |
  | k2 | 0.94 | 0.91 | 38 | 1 | 0.049 | 0.052 | 0.013 |
  | k3 | 0.52 | 0.93 | 28 | 2 | 0.066 | 0.106 | 0.012 |
  | k4 | 0.12 | 0.92 | 120 | 1 | 0.261 | 0.198 | 0.012 |
  | k6 | 0.00 | 0.75 | 15 | 2 | 0.041 | 0.327 | 0.009 |

  * **Mosaic seams are not the false-alarm source:** divided by the k² frames each mosaic packs in, E2's background false alarms are a flat **~0.012 per source frame** at every scale. They're genuine scene content (unlabelled objects, clutter) at a steady, low rate: ~1 per 80 frames.
  * On real (native) frames, both models raise almost no false alarms (~21-30 in 1,782).
  * E1's distractor confusions happen at small sizes (k3: 12 birds + 10 airplanes called drone; k4: 8 airplanes). E2 has almost none: **small-target training also improved small-size discrimination.**
  * E2 holds ≥ 0.91 recall down to k4 (~8 px median) and 0.75 at k6 (~6 px).
  * **Corrects the earlier deployment estimate:** the per-area rate is ~0.012 per 640×512 frame, and a 4K frame is ~25 of those, so **~0.3 false drones per full 4K scan** (not ~1.3; the per-image figure was inflated by mosaic density). At a 2 Hz full scan that is still ~0.6/s, i.e. thousands per hour before tracking. Temporal confirmation (M2) remains required.
* **2026-09-24: False alarms by size, and E2's overall trade-off.** Background false "drone" boxes by predicted width:
  | Width | 4-8 | 8-12 | 12-16 | 16-24 | 24-32 | 32-64 | Total |
  | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | E1 @ 0.25 | 0 | 1 | 93 | 163 | 37 | 28 | 326 |
  | E2 @ 0.25 | 114 | 106 | 80 | 32 | 35 | 78 | 447 |
  | E2 @ 0.4 | 26 | 24 | 23 | 7 | 20 | 9 | 109 |

  * **E2 at conf 0.4 dominates E1 at 0.25:** pooled drone recall 0.89 vs 0.52, with ~96% of drone calls correct vs ~83% (hits ÷ hits + all false drones), and fewer false alarms in every bin from 12 px up. The price is ~50 tiny (4-12 px) false alarms over 2,585 images.
  * **Confirmed with the three-way split:** of E1's 326 "background" false drones at 0.25, **230 were extra or misplaced boxes on real drones**; only **96** were on empty background. E2 at 0.4: 6 on drones, 103 on background, 6 on distractors. So **E2's small-target gain costs essentially no extra true background false alarms** (103 vs 96). They just shift smaller (4-12 px: 46 vs 1).
  * E1 often *saw* small drones but drew boxes that didn't match (IoU < 0.3), so some of its "misses" at 12-24 px were localisation failures, not detection failures. Training on small drones (E2) fixed both.
  * True background false drones by width after the split: E1 @ 0.25: 4-8: 0, 8-12: 1, 12-16: 25, 16-24: 38, 24-32: 20, 32-64: 11, 64+: 1. E2 @ 0.4: 24 / 22 / 21 / 7 / 20 / 9 / 0.
  * **Deployment translation (superseded by the per-set breakdown above: ~0.3 per 4K scan):** 109 / 2,585 ≈ 0.04 false drones per 640-px tile. A full 4K scan is 32 tiles, so ~1.3 per scan before any tracking. Per-frame detection alone can't meet a sensible false-alarm-per-hour target; **temporal confirmation (tracker, k-of-n frames, M2) is required**. Caveat: these val tiles are object-dense; real empty-sky rates must be measured on own footage.
* **2026-09-24: E2 answer: training on simulated distance cuts `w_min` from 16 px to 4-8 px.**
  * Recall at 4-8 px goes 0.03 → 0.86 and at 8-12 px 0.36 → 0.97 (conf 0.25), with no loss at native size (drone R 0.875, mAP50 0.852 → 0.873).
  * Scores are also better behaved: at conf 0.4, E2 keeps ≥ 0.8 recall from 8 px up with 109 background false drones, fewer than E1 at the same threshold (161) with far better small-target recall. **E2's operating point: conf ≈ 0.4.**
  * Birds called drones fell 13 → 0. Airplane recall fell 0.81 → 0.71 (possibly shrunken airplanes resemble small drones; borderline against the ±3-5 point noise).
  * **Most optimistic possible reading:** train and val mosaics come from the same generator, so the model may exploit mosaic-specific cues ("clean blob in a cell"). Background false alarms at conf 0.25 rose 326 → 447, and a 0-4 px recall of 0.47 means it now fires on near-dots.
  * Before claiming range: (a) false alarms binned by size, (b) look at them (cells 14-15), (c) degraded mosaics (`--noise`, blur, JPEG), (d) real long-range footage.
  * If 8 px holds up, range roughly doubles vs E1: 16 mm Pi HQ Mini ~323 m, FPV ~387 m, Mavic ~452 m.
* **2026-09-24: E1 answer: `w_min` ≈ 16 px, not the assumed 10.**
  * Recall collapses below 12 px (8-12: 0.36; 4-8: 0.03) for a model trained only on ≥ ~20 px drones.
  * With the doc 13 formula, 16 px cuts range to 10/16 = 62% of plan. For the 16 mm Pi HQ camera: Mini 258 → 161 m, FPV 310 → 194 m, Mavic 361 → 226 m.
  * `geometry_budget.py --scenario A --min_px 16` needs HFOV ≤ 16.3° (6 cameras for the 90° sector) instead of 25.8° (4 cameras).
  * E2 tests whether training on small targets recovers this.
* **2026-09-24: Detector confidence is not a probability.** Drone recall at 24-64 px drops from 0.96-0.99 at conf 0.25 to 0.62-0.67 at 0.4, and to ~0 at 0.55: this model's drone scores top out around 0.4-0.5. YOLO26 scores are quality-aware (they encode box overlap as well as class), and our boxes are loose (mAP50-95 0.35). **Operating point conf ≈ 0.25.** Any rule or classifier consuming these scores (M2, doc 12 rules) must use calibrated values, not raw ones.
* **2026-09-24: Birds are rarely called drones; background is the bigger false-alarm source.** At conf 0.25, 13 bird / 18 airplane / 5 helicopter boxes became "drone", out of roughly 5× the native val distractor count once the zoom copies are included. But there were 326 background false drones over 2,585 images, a pooled figure mixing native frames and zoom mosaics (whose seams may cause some). Diagnostic cells 14-15 split this by set and show examples.
* **2026-09-24: Optimizer and activation, as used.**
  * Checked in the Ultralytics source (main branch, 2026-09-24): conv layers use **SiLU** (`default_act = nn.SiLU()`).
  * `optimizer=auto` chooses **MuSGD (lr 0.01) above 10,000 iterations, otherwise AdamW** with lr = 0.002·5/(4+nc), i.e. 0.00125 for 4 classes. E1's log confirms AdamW at 0.00125.
  * **Swapping the activation isn't worth testing yet.** The COCO-pretrained weights were learned with SiLU, so changing it means retraining from scratch and losing the pretraining, for gains that are usually within noise. It only becomes relevant for Pi 5 speed (ReLU can be cheaper in some runtimes), measured in E3.
  * Optimizer choice is a modest lever (E10). E1 showed the bottleneck is data diversity.
* **2026-09-24: E1 training.** Birds are the weakest class (recall 0.71), which is the "drone eyes" problem in one number. The model plateaued at epoch 12, and training loss kept dropping after that while validation stayed flat, so the gains will come from data diversity, not more epochs. **Validation is effectively ~57 videos, not 1,782 frames**: differences under ~3-5 points between experiments are noise until the evaluation reports per-video bootstrap confidence intervals (to do). Drone mAP50 0.85 but mAP50-95 0.35: boxes are placed loosely (partly loose hand-drawn labels), so box-width range estimates from this model will be noisy.

* **2026-09-23: Halmstad (visible) has no small drones.** The E1 sanity check (Colab) covered 285 videos at 640×512, 89,233 frames. Box widths: drone median 33.8 px (10th percentile 23.0); bird 26.7 (22.2); airplane 40.9 (24.5); helicopter 72.4 (25.6); **0% under 8 px** for every class. On its own, Halmstad can't measure `w_min`, hence the zoom-out evaluation (`cuas/zoomout.py`). The label-drawing check on V_DRONE_001 was correct.
* **2026-09-23: Some Halmstad videos stop being labelled early**, mostly birds: V_BIRD_005 has 1 label for 302 frames, V_BIRD_009 37/325, V_BIRD_017 173/306 (first 10 listed were all birds). The converter now skips those tails and takes no background frames from bird videos. Otherwise unlabelled birds would be taught as background. When fine-tuning from COCO weights, Ultralytics maps the pretrained class heads by name. **Bird and airplane inherit COCO heads, while drone and helicopter start from scratch**, so the distractor classes get a head start. Keep this in mind when comparing per-class results. On Colab it flagged **12 short videos**. It wrote **6,829 train / 1,782 val tiles** (`--every 10`), plus zoom-out val mosaics k2/k3/k4/k6 = 445/198/111/49. Median drone width ~34 px maps to ~17/11/8/6 px at k = 2/3/4/6.

* **2026-09-23: Registration on sky is harder than it looks.** Plain phase correlation was only ~1 px accurate on a low-texture synthetic sky, because flat regions dominate it. Running it on gradient magnitude gave ~0.24 px (ECC gave ~0.03 px at ~4× the cost). But then one dark, high-contrast target outweighed every cloud edge: registration locked onto the *drone*, and the hit rate fell to 0.42 with ~3,000 false blobs. **Clipping gradient magnitude at its 99th percentile** fixed it (hit rate 1.0, 5 false blobs in 60 frames). Both cases are regression tests. On a real site, the static skyline strip will likely carry most of the registration texture.

## 6. Build Milestones

1. **M1 (done 2026-09-23):** tiling, motion cueing, detector interface, evaluation, replay, Halmstad converter, E1 notebook.
2. **M2 (in progress; 2026-10-01: the pipeline runs as a program on video, `python -m cuas`, and is scored at system level by E20; §5):** tracker (ByteTrack-style association with a constant-velocity Kalman filter first, IMM later, [09](09_monocular_depth_frame_tracking_and_kinematics.md)); per-track features (speed, straightness, hover, size and brightness oscillation spectrum); a calibrated drone-vs-not-drone classifier over tracks (ECE check, [12 §5](12_typesafe_jev_system_one_c2_threat_engine.md#5-before-trusting-jev-validate-it-on-your-tracks)); the rules floor from doc 12.
3. **M3:** Pi 5 capture node (HQ camera, locked exposure, raw or high-quality recording, motion cueing on-device), streaming crops to the PC.
4. **M4:** second camera, either a stereo pair for metric range and size, or wide + narrow for coverage. Decide after E1-E6.
5. **Much later, not a priority (user, 2026-09-28): predict where a drone is heading from its momentum.** Extends M2's tracker: a short-horizon trajectory forecast (the Kalman/IMM prediction step run forward, [09](09_monocular_depth_frame_tracking_and_kinematics.md)), later possibly a likely destination given the protected zone. Its cost depends on the compute left over after detection and tracking ([18](18_hardware_optimisation_and_realtime_latency.md)). Monocular range is coarse, so forecasts will be mostly bearing and time-to-collision, not metric paths, unless stereo is added (M4).
