# Design Review, Parameter Audit & Novel Directions

A stepped-back review of the whole project on 2026-10-01: the code, the method, every parameter that matters, and which newer research is worth testing. It records what was found, what was fixed, what the new measurements say and what to run next. Run-by-run detail is in [16 #54, #56](16_training_run_log.md#2026-10-01-design-review-fixes-system-evaluation-degradation-sweep); results and interpretation are folded into [14](14_build_log_and_experiments.md) and [15](15_detector_training_methods_and_findings.md).

> [!NOTE]
> Written 2026-10-01. Numbers come from files in `C:\cuas_data` (named where used) or from the Colab output of `colab_e18.py` pasted that day. Papers in §5 were checked at their arXiv pages; **only abstracts were read**, so their claims are the authors' and none has been run here.

---

## 1. Verdict

* **The measurement method is the project's strength:** per-video bootstrap, matched-recall comparison, the size/position confound probe, a held-out dataset.
* **The detector is overfit to its source datasets, not to its samples.** Training loss keeps falling (E12 box loss 2.05 → 1.33) while validation is flat from about epoch 6 (box loss 1.60-1.66; mAP50 peaks 0.889 at epoch 6, ends 0.84). In-domain test recall is 0.87-0.90; on DUT it is 0.14. It is not underfitting, so more capacity (E18's add-ons) addresses the wrong term.
* **The small-target result does not survive a realistic sensor** (§3.2). At ~8.5 px, recall falls from 0.93 to 0.08 under moderate blur, noise and compression together. This is now the first thing to fix, and it is a training-data fix.
* **One seed is not a result** (§3.1). Two seeds of the same recipe differ by more than several effects reported so far.
* **The system-level number existed nowhere** until E20 (§3.3): false drone declarations per hour, and seconds to declare a real drone.
* **Against the wider literature (§10) the architecture is mainstream.** What differs from the strongest published work is around it: clean-only training, no object-level augmentation, IoU matching at a few pixels, one threshold for all sizes. Two faults suspected from that reading were measured here and turned out small (one inference scale; unlabelled objects in training tiles).

---

## 2. Code Review: Findings, Closer Evaluation, Fixes

Each finding was re-measured before fixing. Every fix has a regression test that fails on the old code (checked against the committed tree) and passes now. Tests: `tests/test_cuas.py` went from 38 to 47 (plus `test_arch.py`'s 10, which need the venv); setup in [14 §3](14_build_log_and_experiments.md#3-setup).

| # | Finding | Closer evaluation | Fix |
| :-: | :--- | :--- | :--- |
| 1 | `.gitignore`'s `datasets/` also ignored the source package `cuas/datasets/` | 0 of its files on `origin/main`; a fresh clone could not import the tests or build any dataset | Rules anchored to the repo root (`/datasets/`). **The four converters are now untracked files waiting for a commit** |
| 2 | A tentative track could take a confirmed track's detection: one joint assignment on Mahalanobis distance favours the track with the *largest* covariance | Reproduced. Rare on Halmstad (14 in 24,988 updates) but it matters in clutter | Confirmed tracks are matched first, tentative ones on what is left. On real E12 detections: confirmed tracks 570 → 516, non-target tracks 6.1 → 5.1 per minute, target-track observations +1.4% |
| 3 | `optimizer=auto` was 3% below its switch point (E14: 9,680 of 10,000 nominal iterations) | From the Ultralytics 8.4.162 source: above 10,000 it switches AdamW lr 0.00125 → MuSGD lr 0.01. ~520 more training images would have changed optimizer and learning rate together with the data | `train_local.py` pins AdamW, lr 0.00125, momentum 0.9, no bias warm-up: identical to what auto chose for every run so far |
| 4 | An unclassified track was treated as "not a drone" (`drone_prob` NaN → 0.0), while a track labelled "unknown" with p ≥ 0.5 was treated as a drone | A closing object with no classification got `log_only`; a 0.07-s-old track could alert | `drone_prob` is `None` until classified, and the classifier's label wins over p. A closing unknown gets `slew_ptz_for_identification` (not an alert: see #16) |
| 5 | A kill during Ultralytics' in-place save leaves a truncated `last.pt`; the Colab cell then crashed on every re-run | A truncated copy of a real checkpoint raises `RuntimeError` in `torch.load` | `checkpoint()` sets it aside and resumes from `best.pt` (or starts fresh) |
| 6 | One object boxed under two classes started two tracks | 68 of 79,858 frames in-domain (0.1%); more where classes are confused (DUT) | Class-agnostic NMS before the tracker (`nms(class_aware=False)`); evaluation stays class-aware |
| 7 | Motion cueing registered every frame to the **first** frame for ever | On a synthetic sky that cross-fades to new clouds: hit rate 0.44, 56 false blobs per frame | The reference is a slow blend of registered frames; restart if the shift exceeds 16 px. Same test: hit rate 1.0, 0.3 false blobs per frame. `max_shift` is untuned without real footage |
| 8 | `0 < ttc` excluded a time to collision clamped to 0 | Only with noise or acceleration; fixed anyway | `0 <= ttc` |
| 9 | Per-frame cost and memory grew with track age | A track on a lamp for an hour meant 108,000 rows rebuilt per frame | The pipeline's tracker keeps 2.5 s of history and no finished tracks |
| 10 | `training_tiles` crashed on a box starting off-frame | `rng.integers(0, -4)`; not hit so far | Tile placed around the visible part; a mostly hidden target writes no tile |

**Found while fixing (not in the first review):**

| # | Finding | Evidence | Fix |
| :-: | :--- | :--- | :--- |
| 11 | **Time to collision was mostly noise on young tracks.** Detector box widths jitter 3-6% (measured, Halmstad tracks) | With the old fit, **75% of non-closing tracks read "TTC under 10 s" at least once in their first 3 s** | A significance test on the slope (z = 4), at least 0.5 s of boxes, and a 2-s window: under 0.5% false; a real 10 s is always seen. A 1-s window sees it 29% of the time |
| 12 | A flapping bird fakes optical expansion: its box widens 2-3× within one wing beat | The simulated scene produced "TTC 0.2 s" on the bird | The 0.5-s minimum span puts a whole beat in the fit, and the residuals reject it |
| 13 | The TTC fit of ln(width) needed a lag correction and was biased near impact | Bias −0.16 s at 0.5 s to go | 1/width is exactly linear in time for a constant closing speed; the line's zero crossing is the collision time. No correction, no bias |
| 14 | A hovering drone found by the periodic scan could never be confirmed | Crops followed only confirmed tracks; a tentative track expires in 0.25 s and the next scan is 0.5 s away | Crops follow every live track |
| 15 | TTC reads long while the target is a few pixels wide | Simulated: 9.6 s for a true 6.5 s at 4 px; 3.0 for 2.7 at 14 px. Blur adds a constant to a small width | Documented, not fixed. TTC is trustworthy from roughly 8 px up |
| 16 | **On real tracks TTC still raises false "closing" readings.** Box-size error is correlated in time (pose, banking, detector wobble), so the fit's noise test under-counts it | All 285 Halmstad videos, E12 detections: 2.9-5.7 tracks per minute read under 10 s at some point, and 0.8-1.9 per minute under 3 s, on footage of distant aircraft and birds. Width, height and √(w·h) give the same rates | An unknown closing track is looked at, never alerted on; an alert needs a "drone" label. **TTC needs footage with range truth before it can carry more weight** (PX4 logs, §6, or a drone flown at the camera) |

---

## 3. What the New Measurements Say

### 3.1 Seed noise

Two runs of the same recipe: E14 (seed 0) and `E18_base_s1` (seed 1), both on a Colab L4.

| Native frames | E14 (seed 0) | E18_base_s1 (seed 1) |
| :--- | :--- | :--- |
| Val-chosen threshold | 0.55 | 0.55 |
| Anti-UAV **test** at 0.55: recall / precision / background FAs per 1000 | 0.762 / 0.877 / 14.7 | 0.843 / 0.792 / 12.8 |
| Anti-UAV test at 0.70: recall | 0.468 | 0.787 |
| Halmstad val at 0.55: recall | 0.138 | 0.599 |
| Halmstad val at 0.40: recall | 0.933 | 0.923 |
| DUT test recall (own threshold) | 0.076 | 0.149 |
| Pooled Anti-UAV test `w_min` | n/a | 12 px |

At matched recall (`compare_runs.py --match`), the per-video bootstrap calls seed 1 **"better"** than seed 0 on test background false alarms (−13.2 per 1000 frames [−22.3, −5.5]), on Halmstad recall (+0.054 [+0.003, +0.116]) and on val precision (+0.023 [+0.003, +0.040]).

What follows:
* **The bootstrap interval is conditional on the trained model.** It measures test-set sampling error, not training variance. "Statistically clear" has meant "these two weight files differ", not "the recipe caused it".
* **E14's verdict (15 §6.5) is not established.** Five sets of weights at the same held-out test recall (~0.90, native Anti-UAV frames):

  | Weights | Threshold | Test recall | Background FAs per 1000 frames |
  | :--- | ---: | ---: | ---: |
  | E12, `best.pt` (no hard negatives) | 0.40 | 0.899 | 133 |
  | E12, `last.pt` (same run, epoch 36) | 0.55 | 0.894 | 62 |
  | E14, seed 0, `best.pt` (hard negatives) | 0.10 | 0.896 | 159 |
  | E14, seed 0, `last.pt` (same run, epoch 28) | 0.25 | 0.898 | 46 |
  | E14 recipe, seed 1 (`E18_base_s1`) | 0.25 | 0.900 | 64 |

  The spread (46-159) follows the seed and the checkpoint more than the recipe. The −44% reported for hard negatives was one pair of `best.pt` weights matched at one validation recall. Comparing final weights, hard negatives give 46 against 62: a smaller gain in the same direction, from one seed. It takes three seeds per arm to say.
* **The paired comparison on final weights** (`compare_runs.py --match E12_multidata_yolo26s@last E14_hardneg_yolo26s_L4@last`, 16 #59; `C:\cuas_data\runs\cmp_E12_E14L4_last.txt`), E12 at 0.55 and E14 at 0.37 (matched validation recall):
  * Anti-UAV test: background false alarms −35.8 per 1000 frames [−64.1, −15.5] (126 → 53 boxes), recall −0.017 [−0.046, +0.010].
  * Anti-UAV val: 256 → 11 background boxes.
  * Halmstad val: recall +0.226 [+0.133, +0.335] (0.765 → 0.991), with 15 background boxes against 0.
  * So the hard-negative run is the better pair of weights again, as it was with `best.pt`. That makes two pairings from the **same two training runs**, not two pieces of evidence about the recipe: the second seed of the hard-negative recipe (64 at test recall 0.90) is level with E12's final weights (62). The verdict stays "not established" until `E18_base_s2` and a second E12 seed exist.
  * The Halmstad difference is a threshold effect: E12's final weights lose a quarter of Halmstad drones at the threshold its Anti-UAV validation set chose (0.55), and none at 0.40. It is the size-and-dataset score bias of §10.2.
* **"E14 is worse than E12 on DUT" (0.076 vs 0.141) was noise:** the same recipe with another seed gives 0.149.
* **The score scale moves between seeds.** At one threshold, Halmstad recall is 0.14 or 0.60. Thresholds must be chosen per model, and `w_min` at a val-chosen threshold moved from 8 px (E12) to 12 px.
* **Rule from now on:** three seeds per arm before a recipe claim, reported as the mean with the seed spread, and compared at matched recall. `E18_base_s2` is queued for this.

**Update 2026-10-03: two seeds under the new protocol agree** (16 #64). `E18_base_s1` and `E18_base_s2`, both 40 epochs, final weights:

| Final weights, native frames | Plain, seed 1 | Plain, seed 2 |
| :--- | ---: | ---: |
| Threshold chosen on val | 0.55 | 0.55 |
| Anti-UAV test at 0.55: recall / precision / background FAs per 1000 | 0.859 / 0.784 / 17.2 | 0.843 / 0.780 / 15.2 |
| Anti-UAV test at 0.40 | 0.886 / 0.738 / 35.8 | 0.864 / 0.737 / 30.9 |
| Anti-UAV test at 0.25 | 0.906 / 0.693 / 67.2 | 0.884 / 0.700 / 56.9 |
| Halmstad val recall at 0.55 / 0.40 | 0.716 / 0.954 | 0.772 / 0.950 |
| DUT test recall / background false drones | 0.138 / 81 | 0.157 / 90 |
| Pooled test `w_min` | 12 px | 12 px |
| E21, ~8.5 px, clean | 0.816 | 0.804 |
| E21, ~8.5 px, blur 1.0 + noise 6 + JPEG 40 | 0.150 | 0.111 |
| E21, ~8.5 px, held-out: motion 5 + shot noise + WebP 30 | 0.019 | 0.006 |
| E21, native size (~34 px), clean → trained-kind combination → held-out combination | 0.716 → 0.232 → 0.115 | 0.772 → 0.425 → 0.215 |

* **The large seed differences above were mostly the protocol, not the seed.** With a fixed schedule and final weights, the two seeds are within 0.02 of each other in test recall at every threshold and within about 15% in false alarms. The earlier pair differed by 0.08 in recall and 4× in Halmstad recall; one of them (E14) had stopped at epoch 28 and both were `best.pt`.
* At matched validation recall the bootstrap finds no clear difference in test recall or false alarms; it still calls seed 2 better on Halmstad recall (+0.189) and seed 1 better on test precision (−0.034), which is seed 2 sitting at a lower threshold (0.46).
* **What a seed difference now looks like:** about 0.02 in test recall, 0.02 in DUT recall, 0.05 in Halmstad recall at 0.55, and a factor of up to two under heavy degradation at native size (0.232 against 0.425). An E22 effect has to be clearly outside those.
* A third seed (`E18_base_s0`) is still queued.

### 3.2 E21: degraded real frames

Source: `C:\cuas_data\runs\E12_multidata_yolo26s\reports\e21_E12_multidata_yolo26s.txt`.

E12 at its threshold (0.4). Each evaluation image is degraded just before the detector sees it. Drone recall:

| Condition | Halmstad native (~34 px) | Halmstad k4 (~8.5 px) | Anti-UAV test k8 (~14 px) |
| :--- | ---: | ---: | ---: |
| Clean | 0.848 | 0.931 | 0.857 |
| Noise σ 4 grey levels | 0.663 | 0.619 | 0.815 |
| Noise σ 8 | 0.248 | 0.239 | 0.674 |
| Noise σ 16 | 0.028 | 0.043 | 0.117 |
| Blur σ 0.75 px | 0.745 | 0.869 | 0.858 |
| Blur σ 1.5 px | 0.629 | 0.349 | 0.768 |
| JPEG quality 50 | 0.841 | 0.569 | 0.738 |
| JPEG quality 25 | 0.805 | 0.280 | 0.651 |
| JPEG quality 10 | 0.228 | 0.014 | 0.631 |
| Haze, contrast × 0.5 | 0.799 | 0.880 | 0.739 |
| Haze, contrast × 0.25 | 0.834 | 0.763 | 0.537 |
| **Blur 1.0 + noise 6 + JPEG 40** | **0.373** | **0.076** | **0.462** |

* **The 8-px working figure holds only for clean images.** A sensor noise of 4 grey levels costs a third of small-target recall; the combined "field-like" row leaves 8%.
* The zoom-out mosaics averaged noise and blur away in training as well as in testing (15 §7.3 flagged the test side). The detector has never seen a noisy small drone.
* Haze alone (lower contrast) is tolerated well. Noise and compression are the problem.
* The degradations are simple models (white Gaussian noise per channel, Gaussian blur, JPEG). They show sensitivity, not the intended camera's actual noise; that needs its footage.
* **This is a data fix, and a cheap one (E22, §8):** degrade the training images the same way. `cuas.zoomout --noise` already exists and was never used for training.

### 3.3 E20: the whole system on held-out videos

Source: `C:\cuas_data\m2\e20_system_dets_E12.txt`.

`experiments/e20_system_eval.py` runs detections → tracker → track classifier → label, frame by frame, over the 57 Halmstad validation videos (23 drone, 34 non-drone; 5.9 minutes of non-drone video). The track classifier is fitted on the train videos only.

| Decider (hold before declaring) | False drone declarations on non-drone videos | Drone videos declared | Seconds to declare: median / 90th pct |
| :--- | :--- | ---: | ---: |
| Frame alarm (detector only, no tracking) | 0 frames | 23 / 23 | n/a |
| Detector votes (0 s) | 0 (upper 95% bound 30 per hour) | 23 / 23 | 1.03 / 1.22 |
| Detector + track (0 s) | 0 (upper bound 30 per hour) | 23 / 23 | 1.03 / 1.22 |
| **Track only** (0 s) | **7 = 71 per hour** (upper bound 134) | 21 / 23 | 1.03 / 1.50 |
| Track only (0.5 s hold) | 2 = 20 per hour (upper bound 64) | 19 / 23 | 1.53 / 2.50 |
| Track only (1.0 s hold) | 1 = 10 per hour (upper bound 48) | 18 / 23 | 2.03 / 4.31 |

* **In-domain, the detector's own class is enough:** no false drone on 5.9 minutes of birds, airplanes and helicopters. Six minutes can only bound the rate at 30 per hour; a real figure needs hours of footage.
* **The track-only classifier ("93% correct on decided windows") gives 71 false declarations per hour as a system.** A track is re-read several times a second on overlapping windows, so it gets many chances to be called a drone once. Per-window accuracy overstates system performance.
* **A hold time trades that against delay:** 1 s of unbroken "drone" cuts false declarations 7 → 1 and costs 3 of 21 drone videos and a second of delay. The principled version is a sequential test on the calibrated probability (§5, rank 6).
* The 1-s floor on the delay is the classifier's `min_duration`.
* **Alerts.** With the first version of the unknown rule (alert under 3 s), the rules floor raised 4 alerts in 5.9 minutes of non-drone video, none of them real (#16). With the final rule it alerts only on tracks labelled "drone" that read as closing: none on the non-drone videos and 13 tracks on the drone videos with the detector's votes (3 and 13 with the track-only classifier). Whether those 13 drones were really closing is unknown: there is no range truth.
* Caveats: in-domain footage; the track-only classifier's negatives are birds and aircraft only (no clutter tracks, §4.4).

### 3.4 Checkpoint choice is a noise source too
Ultralytics picks `best.pt` by 4-class mAP50-95, which is not this project's objective. E12's `best.pt` is epoch 21 and its `last.pt` epoch 36 of the **same run**. Native frames, recall / precision / background false alarms per 1000 frames:

| E12 | `best.pt` | `last.pt` |
| :--- | :--- | :--- |
| Anti-UAV test at 0.40 | 0.899 / 0.689 / 133 | 0.911 / 0.685 / 125 |
| Anti-UAV test at 0.55 | 0.858 / 0.782 / 48 | 0.894 / 0.752 / 62 |
| Anti-UAV val at 0.55 | 0.964 / 0.730 / 107 | 0.986 / 0.684 / 173 |
| Anti-UAV val at 0.70 | 0.917 / 0.859 / 16 | 0.964 / 0.746 / 141 |
| Halmstad val at 0.40 | 0.848 / 0.995 / 0 | 0.943 / 0.997 / 0 |
| Halmstad val at 0.55 | 0.310 / 0.995 / 0 | 0.765 / 0.996 / 0 |

* **At matched test recall (~0.895), `last.pt` has half the background false alarms** (62 against 133). That is as large as the hard-negative effect E14 was run to measure. On Anti-UAV val the order reverses (at recall 0.964: 141 against 107).
* **E14 shows the same thing more strongly** (`best.pt` epoch 13, `last.pt` epoch 28; `C:\cuas_data\runs\best_vs_last.txt`): at test recall ~0.90, 46 against 159; test at 0.55, recall 0.840 against 0.762 with 7.9 against 14.7 false alarms; and here Anti-UAV val agrees (at recall ~0.97: 3.4 against 16.9). In both runs the later weights are better on the held-out test.
* **`best.pt` is not better on this project's metric, and the choice of checkpoint moves the result as much as the experiments do.** Early stopping also ends each run at a different epoch (E12 36, E14 28, E18_base_s1 40), so "last" is not the same thing across runs either.
* **Rule from now on:** a fixed number of epochs, no early stopping (`--patience 100`), and the final (EMA) weights for every run, scored with `e12_eval.py <run>@last`, so the checkpoint is not a free variable.
* **How many epochs: 40, not 25.** This document first said 25, reasoning that every `best.pt` came at epoch 12-21. That used the criterion this section shows to be wrong (validation mAP). The only direct evidence points the other way: the later weights were better on the held-out test at epochs 28 and 36. 40 is also the schedule the learning-rate decay and the mosaic-free last 10 epochs were set for in every run, and `E18_base_s1` and `E18_full_s0` already ran all of it. **Whether more than 40 helps is untested:** `E18_base_s1_e80` (seed 1 again, 80 epochs) is queued to answer it against `E18_base_s1`.

**Update 2026-10-04: 80 epochs against 40** (`E18_base_s1_e80` against `E18_base_s1`, same seed; 16 #70):

| Plain recipe, seed 1, final weights | 40 epochs | 80 epochs |
| :--- | ---: | ---: |
| Anti-UAV test, native, at 0.55: recall / precision / background FAs per 1000 | 0.859 / 0.784 / 17.2 | 0.864 / 0.793 / 7.9 |
| At 0.40 | 0.886 / 0.738 / 35.8 | 0.888 / 0.752 / 19.6 |
| At 0.25 | 0.906 / 0.693 / 67.2 | 0.902 / 0.718 / 39.7 |
| Test background boxes at matched validation recall | 35 | 11 |
| Halmstad val recall at 0.55 / 0.40 | 0.716 / 0.954 | 0.787 / 0.982 |
| DUT: recall / background false drones | 0.138 / 81 | 0.140 / 54 |
| ~8.5 px: clean / trained-kind degradation / held out | 0.816 / 0.150 / 0.019 | 0.836 / 0.137 / 0.016 |
| ~34 px: clean / noise σ 4 / noise σ 8 | 0.716 / 0.450 / 0.309 | 0.787 / 0.366 / 0.166 |
| Training time | 1.8 h | 3.6 h |

* **Twice the schedule halves the false alarms on the trained datasets and leaves recall where it was.** At matched validation recall: test background false alarms −11.8 per 1000 frames [−21.3, −3.0], precision +0.025 [+0.014, +0.036], recall −0.003 (no clear difference). 7.9 per 1000 at 0.55 is below all three 40-epoch plain seeds (11.8-17.2).
* **It does nothing for the two problems that matter.** DUT recall is unchanged (0.140 against 0.138). Robustness is unchanged or worse: under sensor noise at native size the 80-epoch model loses more (0.166 against 0.309 at σ 8), a sign of fitting the clean training images more tightly.
* **Decision: experiments stay at 40 epochs.** Arms are compared at equal schedules, and 40 costs half. A final model for deployment could take 80, on the adopted recipe, once the recipe is settled. Limits: one seed pair, on the plain recipe; not yet tried with `--degrade`.

### 3.5 E18: the full add-on model

`E18_full_s0`: AELAN + MPDA + RMPDA on the E14 recipe, seed 0, all 40 epochs (16 #57). Each run at its val-chosen threshold, which is 0.55 for all three:

| `best.pt`, native frames | Base, seed 0 (E14) | Base, seed 1 | **Full add-ons, seed 0** |
| :--- | ---: | ---: | ---: |
| Anti-UAV test recall | 0.762 | 0.843 | **0.791** |
| Anti-UAV test precision | 0.877 | 0.792 | **0.841** |
| Anti-UAV test background FAs per 1000 | 14.7 | 12.8 | **11.3** |
| Test background FAs at val recall 0.944 (count over 2,038 frames) | 30 | 3 | **13** |
| Halmstad val recall | 0.138 | 0.599 | **0.398** |
| DUT test recall | 0.076 | 0.149 | **0.106** |
| Anti-UAV **val** background FAs at val recall 0.944 (count over 1,481 frames) | 15 | 12 | **80** |

* **On the held-out test, Halmstad and DUT the add-on model sits between the two seeds of the plain model.** The bootstrap calls it "better" than seed 0 on two measures (test background FAs −8.3 per 1000 [−16.9, −0.5]; Halmstad recall +0.088 [+0.012, +0.168]), and seed 1 of the plain model beat seed 0 by more on both (§3.1).
* **On Anti-UAV val it is worse than both** (80 background false alarms against 15 and 12).
* **It costs about 1.35× the training time per epoch** (0.069 h against 0.051 h for the plain model, from the two runs' final epochs on the same GPU type) and 1.6-1.8× per tile at inference (14 §5).
* **Verdict: no evidence of a gain.** This is what §1 predicted from the fit diagnosis: capacity was not the missing term. The three ablations can only show which module contributes to a difference that is not there; they are moved to the end of the queue.

### 3.6 E18: the three ablations and the control

The user's running cell (the old queue) finished the three ablations before the new queue was pasted (16 #60, output received 2026-10-02). These are `best.pt` weights under the old early-stopping rule, so they carry the checkpoint noise of §3.4.

| `best.pt`, native frames | Plain, seed 0 (E14) | Plain, seed 1 | Full | Attention only | Deformable only (AELAN) | ECA control |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Epochs run (patience 15) | 28 | 40 | 40 | 29 | 39 | 40 |
| Threshold chosen on val | 0.55 | 0.55 | 0.55 | 0.40 | 0.40 | 0.55 |
| Anti-UAV test at 0.55: recall | 0.762 | 0.843 | 0.791 | 0.802 | 0.805 | 0.849 |
| Anti-UAV test at 0.55: background FAs per 1000 | 14.7 | 12.8 | 11.3 | 40.2 | 8.8 | 26.0 |
| Background FAs per 1000 at test recall 0.83-0.85 (threshold) | 33.9 (0.40) | 12.8 (0.55) | 25.0 (0.40) | **97.2** (0.40) | 21.6 (0.40) | 26.0 (0.55) |
| Test background boxes at val recall 0.944 (2,038 frames) | 30 | 3 | 13 | 24 | 10 | 14 |
| Val background boxes at val recall 0.944 (1,481 frames) | 15 | 12 | **80** | 10 | 2 | 5 |
| Halmstad val recall at 0.55 | 0.138 | 0.599 | 0.398 | 0.240 | 0.425 | 0.575 |
| Halmstad val recall at 0.40 | 0.933 | 0.923 | 0.919 | 0.930 | 0.900 | 0.950 |
| DUT test recall (own threshold) | 0.076 | 0.149 | 0.106 | 0.108 | 0.174 | 0.089 |
| DUT background false drones (2,200 images) | 95 | 92 | 27 | 132 | 271 | 117 |

* **The control is the result.** ECA adds four small tensors and about 5% compute. Against plain seed 0, the bootstrap calls it "better" on test recall (+0.032 [+0.002, +0.060]) and on Halmstad recall (+0.081 [+0.010, +0.161]): as much as any add-on. It is level with plain seed 1 (test recall 0.849 against 0.843, Halmstad 0.575 against 0.599). A near-null change that "beats" the baseline says the baseline is the odd one out.
* **Plain seed 0 (E14) is the weakest of the six runs** on test recall at 0.55 and on Halmstad at 0.55, and it stopped earliest (epoch 28). Every comparison in `cmp_*.txt` is against it, so every "B better" line flatters the add-on. The fair reference is plain seed 1.
* **Against plain seed 1, no add-on is better on any held-out measure:** test recall at 0.55 is 0.79-0.85 against 0.843; test background boxes at matched validation recall are 10-24 against 3; Halmstad at 0.55 is 0.24-0.58 against 0.60.
* **Attention only is worse.** At test recall ~0.85 it has 97 background false alarms per 1000 frames against 13-34 for every other run, and at matched recall it loses Halmstad recall to seed 0 (−0.121 [−0.226, −0.040]).
* **Deformable convolutions (AELAN) are the only add-on with a consistent direction** (fewest validation and test false alarms at matched recall among the add-ons). Its DUT recall of 0.174 is the highest of any run, but at threshold 0.40 with 271 background false drones (precision 0.52); plain seed 1 reaches 0.149 at 0.55 with 92 (precision 0.70). That is an operating point, not a gain.
* **The parts do not add up:** full = attention + deformable, and it is worse on validation than either part.
* **Every run misses at least 83% of DUT drones**, and the five E18 runs box 460-760 of them as helicopter, airplane or bird. The DUT failure is common to all six; architecture did not move it.
* **Verdict: E18 is closed.** No add-on beats a second seed of the plain model. Nothing here is worth three seeds; if one arm were ever revisited it would be AELAN alone (1.33× per tile on the GPU, 1.13× on CPU; 14 §5). The new queue still scores these runs' final weights (`<run>@last`, ~20 min each) because that is cheap.

### 3.7 E13: drone-only training

Sources: `C:\cuas_data\runs\eval_E13_droneonly_s0.txt`, `cmp_E13_droneonly_s0.txt`, `e15_E13_droneonly_s0.txt` (16 #61).

E15 found that the four-class model boxes many DUT drones as helicopter or airplane. E13 tested the obvious fix: train with drone labels only (`--classes 0`), so nothing else can take the box.

| `best.pt`, native frames | Drone-only (E13) | Plain, seed 1 | Plain, seed 0 (E14) |
| :--- | ---: | ---: | ---: |
| Anti-UAV test at 0.55: recall / precision / background FAs per 1000 | 0.852 / 0.792 / 12.3 | 0.843 / 0.792 / 12.8 | 0.762 / 0.877 / 14.7 |
| Test background boxes at val recall 0.944 | 2 | 3 | 30 |
| Halmstad val recall at 0.55 | 0.595 | 0.599 | 0.138 |
| **DUT test recall** (own threshold) | **0.041** (0.40) | 0.149 (0.55) | 0.076 (0.55) |
| DUT sky drones: called drone / boxed as another class / missed | 0.03 / 0.00 / **0.97** | 0.18 / 0.33 / 0.49 | n/a |

* **In-domain it is the same model as plain seed 1,** and no labelled bird, airplane or helicopter in Halmstad val is called a drone.
* **On DUT it is the worst of the seven runs.** The drones the four-class models box as helicopter or airplane are not boxed at all.
* **So the misnaming was never a drone detection with the wrong label.** The drone head does not fire on DUT drones in either model. What finds them in the four-class model is the airplane and helicopter heads. (Corrected 2026-10-03: only the airplane row starts from a COCO class; the helicopter row is learned from Halmstad alone, and it takes more DUT drones than airplane does, so "COCO's broader idea of a flying object" explains part of this at most. A helicopter is the nearest thing in the training data to a large drone.) An earlier version of this line said both started from COCO classes and so carry a broader idea of a flying object than our two drone datasets teach.
* **Verdict: keep four classes.** They are a second, more general detector of airborne objects, and they cost nothing in-domain.
* **What follows for the system:** an unseen drone is more likely to arrive as "an airborne object of some class" (any-class recall 0.49-0.66 on DUT sky across the four-class runs) than as "drone" (0.11-0.22). The pipeline already tracks boxes of every class and lets the track decide (`cuas/pipeline.py`); outside Halmstad-like footage that decision should be the track-only classifier, not the detector's class votes (14 §5). What that costs in false candidates has not been measured.
* **What follows for training:** the drone class needs drones that look different (the DUT train split, Drone-vs-Bird), not a different head. E27 (Objects365 start) tests the same idea from the pretraining side.
* One seed, `best.pt`, and DUT thresholds differ between runs (0.40 against 0.55); the gap (0.97 missed against 0.34-0.51) is far larger than any seed difference seen so far.

### 3.8 Reducing the seed dependency

Asked 2026-10-02: can the result be made less dependent on the seed? Where the dependency comes from, by the evidence here:

| Source | Evidence | Remedy | State |
| :--- | :--- | :--- | :--- |
| **The score scale moves between runs** | At threshold 0.55, Halmstad recall is 0.14 (seed 0) or 0.60 (seed 1); at 0.40 it is 0.933 and 0.923. Most of that "seed difference" is one threshold on two uncalibrated scales | Set each model's threshold at a fixed false-alarm budget on validation footage, never a shared number; calibrate scores per size (E30) | E30 planned, CPU only |
| **Checkpoint choice and early stopping** | Two checkpoints of one run differ by 2× in false alarms (§3.4); the six E18 runs stopped at 28-40 epochs | Fixed 40 epochs, final weights | **Adopted, and it worked:** two seeds now agree within 0.02 in test recall (§3.1, update of 2026-10-03) |
| **Few independent videos** | ~230 Halmstad videos and ~160 Anti-UAV sequences; validation flat from epoch 6 while training loss falls (§1) | More varied data (§3.9) | The real fix, and the slow one |
| **One sample of the weights is deployed** | Whatever the above leave | E31: average the final weights of the seeds into one model ([model soups](https://arxiv.org/abs/2203.05482)) | **Shelved** (below): invalid as run, and no longer needed |

* **E31 is a test, not a given.** Averaging works when the runs end in the same basin, which is usual for fine-tunes from one set of pretrained weights at a small learning rate, and was shown for image classifiers, not detectors.
* **Local check** (16 #62): the final weights of E12 and E14 (two different recipes from the same pretrained weights; the only finished final weights on this machine) were averaged and scored (`eval_E31_soup_E12_E14_last.txt`). The average is a working detector (native Anti-UAV test recall 0.807, precision 0.830 at 0.40; it did not collapse), but **it is worse than either member at matched recall**: at test recall ~0.87 it has 81 background false alarms per 1000 frames against 23 for E14's final weights (+58 [+25, +98]) and 62 for E12's at recall 0.89. Its scores are also lower (Halmstad recall 0.64 at 0.40 against 0.94 and 0.97). So averaging is not free here. This pair differs in training data and in stopping epoch (36 and 28), so it is not the seed test; that is `E31_soup_base_s1s2` on Colab, and the other soups stay switched off until it reports.
* **The seed soup collapsed, for a reason that makes the test invalid** (16 #64). `E31_soup_base_s1s2` (the final weights of plain seeds 1 and 2, averaged) finds **no drone at all** at any threshold, on every set; what boxes it draws are "airplane".
  * **Why:** the detector's class layer has four rows. Ultralytics copies the rows whose names exist in the pretrained model (the logs say "Remapped 2/4": bird and airplane) and leaves **drone and helicopter at a fresh random start, which depends on the seed**. Weight averaging needs every run to start from the same weights; for the drone row ours did not. The local E12 + E14 soup worked because both were seed 0.
  * **So this is a broken precondition, not evidence about the seeds' basins.** A valid test needs every seed to start from one saved four-class model.
  * **Verdict: E31 is shelved.** The one soup that did meet the precondition (E12 + E14) was worse than either member, and the fixed protocol has already brought the seeds within 0.02 of each other (§3.1), so the problem the soup was meant to solve is much smaller than it looked. The other soups stay switched off.
* **Reporting does not change:** every arm is still three seeds, reported as the mean with the spread. The soup is a fourth model per arm, scored the same way.
* Not chosen: an ensemble of the three seeds (three times the inference cost on an edge budget), and longer training alone (tested separately by `E18_base_s1_e80`).

### 3.9 Varied terrain and drone data

Asked 2026-10-02. The detector's unseen-dataset failure (§3.6, §3.7) is a data-variety failure. Sources, by value for cost:

| Source | What it gives | Cost and caveat |
| :--- | :--- | :--- |
| **Own target-free footage** from the intended sites (any camera to start), hours of it | Terrain and sky variety as backgrounds and hard negatives; **the false-alarm rate per hour** (§8 row 7), which nothing else can give | Recording time. No licence question. The highest-value item on this list |
| **Real drone crops pasted on those backgrounds** (E22c) | Drones over terrain we have no drone footage of, at any pixel size, degraded after pasting. What the 2025 Drone-vs-Bird winner did | **Builder written 2026-10-04** (`cuas/paste.py`, `build_e22c.py`), first version pasting across our two datasets; own footage plugs in as one more target folder. The detector can learn paste edges, so score on real footage only |
| **DUT Anti-UAV train and val splits** (7,800 images; 03) | Drone models and backgrounds unlike Halmstad and Anti-UAV | A download. **DUT test then stops being an unseen dataset**, so run it as its own arm and name a new unseen set first |
| **Drone-vs-Bird** (77 videos; waiting for the videos) | Real small drones with birds in the same scenes | Granted, not yet received |
| **SOD4SB / SMOT4SB** (open) | Small birds over varied backgrounds, as negatives and bird tracks | Downloads; birds only |
| **Game-engine sets** (01 §8.2, §8.3, §8.5) | Terrain, weather and lighting on demand | SDV-W "will be publicly released upon paper acceptance" (arXiv abstract, read 2026-10-02); DroneGround on request. Rendered pixels; a supplement at best |

### 3.10 Three checks on cached detections: E30, E33, E34

Run 2026-10-03 on the final weights of plain seeds 1 and 2 and E22a seeds 1 and 2 (16 #66). No training.

**E30: per-size score calibration.** Source: `C:\cuas_data\runs\e30_size_calibration.txt`.

| Median score of correct "drone" boxes, 32-64 px wide | Plain s1 | Plain s2 | E22a s1 | E22a s2 |
| :--- | ---: | ---: | ---: | ---: |
| Anti-UAV val | 0.76 | 0.78 | 0.78 | 0.77 |
| Anti-UAV test | 0.78 | 0.78 | 0.80 | 0.79 |
| Halmstad val | 0.60 | 0.63 | 0.54 | 0.57 |

* **The score gap is the dataset, not the size.** Correct drone boxes of the same width score 0.15-0.25 lower on Halmstad than on Anti-UAV, in all four models.
* **Per-size calibration fitted on Anti-UAV val does not help, and makes Halmstad worse:** with the threshold chosen on Anti-UAV val by maximum F1, Halmstad recall is 0.15-0.26 on raw scores and 0.11-0.18 calibrated; the Anti-UAV test is unchanged (within 0.013).
* **A threshold chosen on Halmstad itself (0.32-0.34) gives recall 0.99 at precision 0.97-0.98.** The detector ranks Halmstad drones correctly; it is only the cut-off that does not transfer.
* **Verdict: refuted.** The fix for threshold transfer is to set the threshold on footage from the site, which is one more use for the target-free recordings (§3.9). §10.2 item 4 ("one threshold for all sizes") is the wrong diagnosis for our data: it is one threshold for all *datasets*.
* A second rule tried (the lowest threshold with at most 10 background false alarms per 1000 validation frames) is unstable on this validation set, whose false alarms include the burned-in caption (15 §7.3): it lands anywhere from 0.45 to 0.83. Not used.

**E33: centre matching for tiny targets.** Source: `e33_center_match.txt`. The five smallest evaluation sets pooled (7,101 drones), the same detections scored by IoU ≥ 0.3 and with drones under 12 px matched by centre (`cuas.eval.match_quality`).

| Recall by drone width | 0-4 px | 4-8 px | 8-12 px | `w_min` | Extra boxes on drones |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Plain s1: IoU → centre | 0.41 → 0.53 | 0.68 → 0.68 | 0.79 → 0.79 | 12 → 12 px | 136 → 44 |
| Plain s2 | 0.51 → 0.69 | 0.69 → 0.70 | 0.80 → 0.80 | 12 → 8 px | 174 → 40 |
| E22a s1 | 0.48 → 0.70 | 0.73 → 0.74 | 0.83 → 0.84 | 8 → 8 px | 265 → 92 |
| E22a s2 | 0.51 → 0.72 | 0.71 → 0.72 | 0.81 → 0.81 | 8 → 8 px | 270 → 108 |

* **IoU matching was not distorting the small-target figures from 4 px up:** recall moves by 0.01 at most. Only under 4 px does it matter (+0.12 to +0.22), where two-thirds of the "extra boxes on drones" were correct detections placed a pixel or two off.
* **`w_min` is unchanged in three runs.** In the fourth it moves from 12 to 8 px on a bin sitting at exactly 0.80, which says more about `w_min` hinging on one bin (§4.3) than about the matching rule.
* **Verdict: a small correction, not a hidden gain.** Keep IoU as the reported rule; quote the centre figure for the 0-4 px bin. §10.2 item 3 shrinks accordingly.
* Side reading: E22a's `w_min` is 8 px in both seeds against 12 px for the plain runs, each at its own threshold.

**E34: letting a box of any class start a track.** Source: `e34_any_class.txt`.

Plain seed 1 and E22a seed 1, final weights. "Reached" = a candidate box overlaps the drone (IoU ≥ 0.3); "on nothing" = candidates per 1000 images that touch no labelled object, each one a track the track classifier would have to reject.

| Plain seed 1 | Threshold | Drone boxes only: reached / on nothing | Any class: reached / on nothing |
| :--- | ---: | ---: | ---: |
| DUT test (never trained on) | 0.55 | 0.138 / 37 | **0.353** / 149 |
| DUT test | 0.40 | 0.189 / 73 | 0.416 / 275 |
| DUT test | 0.25 | 0.223 / 146 | 0.447 / 463 |
| Anti-UAV test | 0.55 | 0.859 / 17 | 0.859 / 82 |
| Anti-UAV test | 0.40 | 0.886 / 36 | 0.886 / 119 |
| Halmstad val | 0.40 | 0.954 / 0.6 | 0.954 / 8.5 |

* **On the unseen dataset, any class reaches 2.2-2.6 times as many drones.** At an equal rate of stray candidates (about 147 per 1000 images) it reaches 0.35 of the drones against 0.22 for drone boxes at a lower threshold, so this is a better trade there, not only a looser threshold.
* **On the datasets it trained on, it is pure cost:** no drone gained, and 2.5-5 times the stray candidates on Anti-UAV (5-8 times with E22a). On Halmstad it also adds about 900 candidates on labelled birds, airplanes and helicopters, which is what it should do: those are real objects for the track classifier to name.
* **E22a seed 1 gives the same picture** (DUT at 0.55: 0.068 as drone, 0.332 any class), with fewer drones under the drone head and about as many under any class as the plain model.
* **What this does not measure:** a candidate is not a declaration. Whether the track classifier rejects the extra candidates needs video; DUT is still images. The system-level number is E20's, and E20 should be re-run on footage with clutter once there is some (own target-free recordings, §3.9).
* **Decision: keep the pipeline as it is** (every class starts a track; the track decides), and use the track-only classifier outside Halmstad-like footage. The stray-candidate rate is the reason the track decision has to be strict (E26).

---

## 4. Parameter Audit

Verdicts: **keep** (justified by evidence here), **pin** (right value, wrong mechanism), **change** (evidence says so), **untested** (a guess that works so far).

### 4.1 Training

| Parameter | Value (from `args.yaml` and the run logs) | Verdict | Why |
| :--- | :--- | :--- | :--- |
| Optimizer | AdamW, lr 0.00125, β1 0.9, weight decay 0.0005, linear decay to 1%, 3 warm-up epochs | **Pin** (done) | Right for a fine-tune under 10,000 iterations. Not the bottleneck: validation plateaus by epoch 6 whatever the schedule does. Was one data change away from silently switching |
| Epochs / patience | 40 / 15 | **Changed: 40, no early stopping** | Every run completes the same schedule (§3.4). 80 epochs was tested on one seed: half the in-domain false alarms at the same recall, no gain on DUT or under degradation, twice the time. Keep 40 for experiments; consider 80 for a final model |
| Batch | 16 (nominal 64 by accumulation) | Keep | Fits 8 GB; identical across runs |
| Activation | SiLU body, sigmoid per class in the head | Keep | Changing it discards the pretrained weights. Sigmoid means classes do not compete, so one object can be "helicopter 0.5, drone 0.3" (E15) |
| Loss | BCE classification, CIoU + L1 box, no DFL; one-to-many weight 0.8 → 0.1; assignment floors target sides to 16 px (read from the installed source) | Keep | Built for small targets. IoU is unstable below ~8 px; NWD (§5.5) is the alternative, after the data fixes |
| Model selection | `best.pt` by 4-class mAP50-95, early stopping (patience 15) | **Change** | Not the project's objective, and a noise source as large as the effects under test (§3.4). Fixed epochs, final weights |
| Seeds | 1 per arm | **Change** | §3.1 |
| Scale augmentation | ±0.5 (default) | Untested | Rescales targets 0.5-1.5×, so training is not at "true pixel size" anyway. E2b was planned and never run |
| Photometric augmentation | HSV only; blur p = 0.01; no noise, no JPEG | **Changed: add `--degrade`** | E22a over three seeds (15 §6.14): recall at ~8.5 px under blur + noise + JPEG 0.12 → 0.32, under held-out degradations 0.01 → 0.10; ~34 px 0.37 → 0.85; clean test recall +0.02. No gain on DUT |
| Mosaic | 1.0, off for the last 10 epochs | Keep | Default; no evidence against |
| Backbone | all layers trained (`freeze` unset) | Untested | Full fine-tuning can hurt out-of-distribution accuracy (§5.4). `--freeze` is now a `train_local.py` argument |
| Classes | 4; the bird and airplane rows start from COCO's, drone and helicopter from a random start (the training log says "Remapped 2/4") | **Keep** | E13 (`--classes 0`, §3.7) equals plain seed 1 in-domain and is the worst run on DUT (recall 0.041; 97% of sky drones missed). The airplane and helicopter heads are what find unseen drones |

### 4.2 Data

| Parameter | Value | Verdict | Why |
| :--- | :--- | :--- | :--- |
| Split | by video, stratified, 80/20 | Keep | Correct. But Halmstad has **no test split**: the same validation videos choose the epoch, the threshold and the reported number |
| Frame sampling | every 10th labelled frame (Halmstad), every 40th (Anti-UAV) | Keep | Neighbouring frames are near-duplicates |
| Zoom-out k | 2, 3, 4, 6 (Halmstad); 2, 4, 8, 16 (Anti-UAV) | **Change** | Clean by construction (§3.2). A k = 16 mosaic holds 256 drones in one image against 0-1 per tile in deployment; untested whether that density feeds clutter false alarms. Pasting shrunk, degraded drone crops onto native backgrounds would fix both |
| Hard negatives | conf ≥ 0.25, width ≥ 28 px, 40 per sequence, 15% share | Untested in effect | The settings are reasoned (16 #36, #39), but the gain is inside the seed and checkpoint spread (§3.1) |
| Background tiles | none from Halmstad or Anti-UAV positives | Keep, and fill the gap | Right given unlabelled drones. But the detector has no verified-empty sky at all |

### 4.3 Evaluation

| Parameter | Value | Verdict | Why |
| :--- | :--- | :--- | :--- |
| Match rule | IoU ≥ 0.3, greedy | **Keep** | E33 (§3.10): matching by centre under 12 px changes recall by at most 0.01 from 4 px up; only the 0-4 px bin moves (+0.12 to +0.22) |
| `w_min` | lowest bin edge with recall ≥ 0.8 in every bin of ≥ 20 drones | **Change** | Hinges on one bin (17 #14) and moved 8 → 12 px between seeds. Fit a recall-against-width curve and report the 80% point with a per-video interval |
| Threshold choice | maximum F1 on Anti-UAV val, 5-point grid | **Change** | F1 weighs a miss and a false alarm equally at a prevalence (a drone in every frame) that never occurs in deployment. Fix a false-alarm budget and take the recall it buys |
| Precision | reported throughout | **Change** | Same prevalence problem. Report false alarms per image or per hour, on target-free footage |
| Bootstrap | 2,000 resamples of videos | Keep | Correct for test-set error; say what it does not cover (§3.1) |
| Inference scales | native tiles only | Keep for now | E28 (§10.3): a whole-frame pass adds 0.03 recall on DUT at one false drone per drone gained. Re-test on weights trained with large drones |

### 4.4 Tracker, time to collision, rules, classifier, motion cueing

| Parameter | Value | Verdict | Why |
| :--- | :--- | :--- | :--- |
| Association | Mahalanobis gate 9.21 (99%, 2 dof), Hungarian | Keep, **order fixed** | §2 #2 |
| `q` (acceleration PSD) | 2e5 px²/s³ | Untested | Sized for a fast crosser at 10 fps. Never tuned on real tracks; target-track coverage is already a median 100% of labelled frames |
| `v0` | 1500 px/s | Untested | Wide enough for a second frame at 10 fps; it is also why tentative tracks out-competed confirmed ones |
| `n_confirm` / coast | 3 hits; 1.0 s confirmed, 0.25 s tentative | Keep | 0.25 s assumes ≥ 8 fps; a 4 fps source would never confirm a track |
| Start / continue score | 0.4 / 0.1 | **Per model** | The score scale moves between seeds (§3.1). Take the start score from the model's own sweep |
| Motion model | CV (IMM available) | Keep CV | The IMM is tested on one synthetic turn only; nothing shows it changes association or classification on real tracks. Parked |
| TTC window, z, minimum span | 2 s, 4, 0.5 s | **Changed** (was 1 s, none, none) | §2 #11, #13, measured |
| Imminent / decide-now | 10 s / 3 s | Site policy, **TTC unvalidated** | With 5% white width jitter a 10-s TTC is known to ±2 s, a 3-s one to ±0.3 s. Real box error is correlated, and 1-2 tracks a minute of distant aircraft read under 3 s (§2 #16) |
| Classifier | gradient-boosted trees, isotonic calibration, band 0.3-0.7, ≥ 1 s, 2-s windows | Keep the model, **change the decision** | Right model for ~270 tracks of tabular features. Declaring on the first "drone" window inflates false declarations (§3.3) |
| Classifier negatives | birds, airplanes, helicopters | **Change** | No clutter tracks. A false track on a lamp has zero speed, like a hovering drone. E20's non-target tracks (5.1 per minute) are the missing class |
| Motion cueing | MOG2 history 300, variance threshold 16, half scale | Untested on real video | Scored only on synthetic sky. A hovering drone is absorbed into the background; the periodic scan is its only route in |
| Reference update | blend 0.02 per frame, restart above 16 px | New, untuned | §2 #7 |

---

## 5. Novel Directions, Ranked Against This Project's Measured Failures

Ranked by whether the mechanism matches a failure measured here, then by cost. A new idea earns its place against the seed spread (§3.1) and its compute cost (17 §9, principle 12).

| Rank | Direction | The failure it targets | Evidence elsewhere (abstracts) | Cost | Experiment |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | **Degradation-aware training**: noise, blur, JPEG on training tiles and zoom-out mosaics | E21: 8-px recall 0.93 → 0.08 | Standard practice; the 2025 Drone-vs-Bird winner used augmentation and scale, not a new architecture ([Laroca et al.](https://arxiv.org/abs/2504.19347)) | One training run | **E22** |
| 2 | **Temporal input**: a stack of registered frames (t, t−k, t−2k) or a motion-difference map beside the RGB frame | Below ~8 px one frame cannot tell a drone from a bird or noise; pixel motion can | [YOLOMG](https://arxiv.org/abs/2503.07115) (RGB + motion difference map), [dual-interval cues](https://arxiv.org/abs/2605.22605), [TransVisDrone](https://arxiv.org/abs/2210.08423), [Dogfight](https://arxiv.org/abs/2103.17242) | Dataset rebuild from video (all three sources are video; registration exists) + one run. A grey 3-frame stack keeps the pretrained first layer | **E23** |
| 3 | **Three seeds per arm** | §3.1 | n/a | 2 extra base runs | `E18_base_s2` (queued) |
| 4 | **Keep the pretrained features**: freeze the backbone, or probe then fine-tune | E15: out-of-distribution recall 0.14 | [LP-FT](https://arxiv.org/abs/2202.10054) (classification) | One argument (`--freeze 10`), one run | **E24** |
| 5 | **A foundation backbone**: a real-time detector on DINOv3 features | E15 again: COCO pretraining is narrow | [DINOv3](https://arxiv.org/abs/2508.10104); [DEIMv2](https://arxiv.org/abs/2509.20787) reports COCO AP 50.9 at 9.7 M parameters; [RF-DETR](https://arxiv.org/abs/2511.09554) | A second detector behind `cuas/detector.py`. Check the licence and the feature stride first: a ViT patch of 16 px puts an 8-px drone in half a token | **E25**, built 2026-10-08 with RF-DETR Small (Apache-2.0, patch 16; 16 #85) |
| 6 | **Sequential decision on the track**: accumulate log-likelihood from the calibrated probability and declare at a threshold (Wald's test) instead of on the first "drone" window | E20: 71 false declarations per hour from a 93%-accurate classifier | Classical; frame-to-frame consistency is also what the Drone-vs-Bird winner post-processes with | Small code change in `cuas/pipeline.py`, scored by E20 | **E26, done 2026-10-05 (15 §6.9): adopted.** Better than the hold rule at every matched false-declaration rate (track-only, ~5/h: 78 against 68 of 91 drone videos, 0.8 s sooner); `python -m cuas --evidence 5.0` |
| 7 | **P2 head (stride 4), P5 branch removed** (moved down by §9.2: it doubles positive samples only for 12-16 px, and YOLOMG's small-object layer was worth 0.02 AP) | Targets under 8 px cover less than one stride-8 cell; targets under 64 px never use P5 | Standard for small objects; [SPD-Conv](https://arxiv.org/abs/2208.03641) attacks the same loss of detail in downsampling | Config change + one run; about 4× the cells at the top level | **E7**, after E22 |
| 8 | **Learned track classifier**: a 3D CNN on the track's image crops | The confound probe (size and position alone reach AUC 0.75) and hand-built features | [Akyon et al.](https://arxiv.org/abs/2207.10409): R(2+1)D beat an LSTM and a Transformer on drone-vs-bird tracks | Needs Drone-vs-Bird videos (shared scenes); ~270 Halmstad tracks are too few | After the videos arrive |
| 9 | **NWD for tiny boxes** in loss, assignment or evaluation matching | IoU jumps with one pixel of error under 8 px | [Wang et al.](https://arxiv.org/abs/2110.13389) | A patch to the Ultralytics loss, or evaluation only (cheap) | Evaluation first |
| 10 | E18 add-ons (deformable convolution, multi-kernel attention) | None measured: the detector is not underfitting | Ties YOLOv8 in its own paper (01 §2.8) | 1.35× the training time per epoch here; 1.1-1.8× per tile at inference; deformable convolution is unverified on TensorRT and Hailo | **Run (`E18_full_s0`, §3.5): inside the spread of the two base seeds on the test, Halmstad and DUT; worse than both on Anti-UAV val false alarms.** No evidence of a gain |

**CNN, RNN, transformer or hybrid?**
* **Per-frame detection stays a CNN, and YOLO26 is already a hybrid** (self-attention at the lowest resolution). At 4-30 px the information is local, and with ~400 training videos the locality prior is what makes the data go far. A plain vision transformer tokenises at 16 px.
* **A recurrent network is the wrong tool for detection and too data-hungry for the tracks.** Time enters in two cheaper places: the pixels (rank 2) and the track (rank 6 and 8).
* **A transformer earns one experiment** (rank 5), for its pretraining, not its architecture.
* **State-space (Mamba) models:** still no drone-detection result and no edge support (11 §5). Not ranked.

---

## 6. Simulation: PX4, Gazebo and `cuas.sim`

This revisits [11 row 9](11_research_deliberations_and_user_pushback_log.md#1-pushback--resolution-log) and keeps its conclusion while making it usable.

* **Use PX4 for the flying, not Gazebo for the pixels.** PX4 SITL gives real multicopter dynamics: hover jitter, turn and braking profiles, wind response. Those are exactly what the track features and the tracker need, and hand-written paths do not have them. A Gazebo camera image of a drone 6 px wide teaches nothing the detector can use, and its noise model is a Gaussian.
* **`cuas/sim.py` is the bridge.** Paths in metres → pinhole camera → target drawn at its true pixel size on a sky → blur, mount shake, sensor noise. `from_px4_csv` reads a PX4 log directly: `ulog2csv -m vehicle_local_position flight.ulg`, whose columns are `timestamp` (µs), `x` north, `y` east, `z` down (checked in the PX4 message reference, 2026-10-01).
* **Noisy data for the detector comes from real frames** (E21, then E22), not from a renderer.
* **Setup when wanted:** PX4 SITL with Gazebo needs WSL2 on Windows, Ubuntu 22.04 or 24.04 (PX4 docs, checked 2026-10-01). The sibling `../Drone_to_Drone` needs it anyway, so set it up once there.
* **What the simulator is for:** tracker, TTC and rules tests with a known answer (it found findings 12, 14 and 15); many-target and crossing scenes; later, checking the M2 features on PX4 flight logs against real tracks.
* **Asked again 2026-10-02: would Gazebo or PX4 be valuable, can they simulate footage and environments, can they test or improve the pipeline?**
  * **Yes, they can simulate footage:** Gazebo renders a camera sensor in a 3-D world while PX4 flies the vehicle, with exact position truth.
  * **For testing the pipeline: yes, valuable.** A stream with known range and velocity is the one thing no dataset here has. It would test the tracker, time to collision (never checked against a true range, §2 #16), the rules and end-to-end latency.
  * **For improving the detector: not Gazebo's pixels.** The published gains from rendered drone data use Unreal Engine (01 §8.2, §8.3, §8.5), and even those are supplements. Our measured gaps are sensor degradation at a few pixels (E21) and drone appearance at 30-250 px on another dataset (E15); a Gazebo drone model answers neither.
  * **The useful combination (E32, not built):** PX4 for the path, real pixels for the picture. A PX4 flight log through `from_px4_csv` gives position against time; `Camera.project` gives the box at its true pixel size; a **real drone crop** is pasted there on **real target-free background video** and the frame is degraded. That yields video with range truth, over any terrain we can film, with real drone appearance. It shares the builder with E22c.
  * **Cost:** WSL2 and a PX4 install (shared with `../Drone_to_Drone`). PX4 can run headless, without Gazebo's renderer, when only the flight log is wanted.
* **What it is not for:** training or scoring the detector, or validating the bird classifier (a scripted bird teaches the script).

---

## 7. The Program

`python -m cuas <video | camera index | stream URL>` runs the whole pipeline (`cuas/replay.py`): motion cueing → crops → detector → tracker → track classifier → TTC → rules floor, with an annotated video, a per-frame CSV and an events file (one JSON line when a track is declared a drone or raises its response).

```
python -m cuas clip.mp4 --weights C:/cuas_data/runs/E12_multidata_yolo26s/weights/best.pt --clf C:/cuas_data/m2/track_clf_dets_E12.pkl --out out.mp4 --events events.jsonl
python -m cuas 0 --weights best.pt --clf track_clf.pkl --show
python -m cuas.sim --out C:/cuas_data/sim/approach.avi
python -m cuas C:/cuas_data/sim/approach.avi --weights blob --out C:/cuas_data/sim/approach_out.mp4
```

* The detector needs the GPU venv (`C:\cuas_venv\Scripts\python -m cuas ...`); `--weights blob` (the simulator's stand-in) does not.
* A live source is timed by the clock at capture, a file by its frame number.
* **Tested end to end** on a simulated scene (three targets; only the closing one escalates; TTC within 0.6 s of truth at the end) and scored on real held-out video by E20.
* **Run on real video** with the E12 detector and the track classifier (`C:\cuas_data\demo\`): V_DRONE_037 declared a drone 1.07 s after it appeared; V_BIRD_034 and V_AIRPLANE_017 produced no declaration and no alert. About 52 ms per frame for the detector (one 640×512 tile, batch 1, laptop GPU shared with another job) and 8 ms for tracking and classification.
* **Not yet run on a live camera.** The code path differs only in the time stamp (clock at capture instead of frame number).

---

## 8. Next Experiments, in Order

| # | Run | Question | Cost |
| :-: | :--- | :--- | :--- |
| 1 | **Plain seeds under the new rule:** `E18_base_s2`, then `E18_base_s0` (seed 0 again; E14 stopped at epoch 28), each 40 epochs with no early stopping, final weights scored | The seed spread of the plain recipe | ~1.7 h + ~35 min scoring each. Queued in `colab_e18.py` |
| 1b | **`E18_base_s1_e80` (done, §3.4):** 80 epochs halve in-domain false alarms at the same recall; nothing on DUT or robustness | Experiments stay at 40 | - |
| 1c | The three add-on ablations (attention only, deformable only, ECA) | **Done (§3.6):** none beats plain seed 1; attention only is worse; the ECA control "beats" seed 0 as much as any add-on | Only their final weights remain to be scored (~1 h, in the queue) |
| 2 | **E22a (done, three seeds; 15 §6.14): adopted.** **E22b (done, three seeds; 15 §6.14): not adopted.** The same test recall as E22a (0.873 against 0.870 at 0.55) with about 1.7 times the false alarms in every seed, and no gain on DUT. **E22c (done, three seeds; 16 #72, #73, 15 §6.17):** E22a plus 3,000 tiles with a real drone pasted in from the other dataset. **Beats E22a on DUT in all three seeds** (drone recall 0.126 → 0.203 at 0.40; DUT AP 0.122 → 0.195, 16 #75); in-domain AP unchanged; robustness unchanged. **Adopted 2026-10-05 as the base recipe.** Against plain: DUT AP level (0.195 against 0.193) but recall at 30 false drones per 1000 DUT images lower in every seed (0.099 against 0.124) | Is the drone class tied to the scenes it was filmed in? | 3 runs, ~2.4 h each with scoring |
| 2b | **E31 (shelved 2026-10-03):** seed averaging. The seed soup collapsed because the drone row of the class layer starts from a seed-dependent random value; the one valid soup was worse than its members; and the seeds now agree under the fixed protocol (§3.1, §3.8) | - | - |
| 3 | **E26 (done 2026-10-05, 15 §6.9, 16 #74): adopted.** The sequential decision beats the hold rule at every matched rate on cross-fitted train videos and agrees on val; the program declares at evidence 5.0. Still open: clutter tracks as classifier negatives (needs footage with clutter tracks; own target-free footage) | False declarations per hour at the same delay | CPU only |
| 4 | **E23 (built 2026-10-08, queued; 16 #83, #84)**: temporal input. Grey frames t−6, t−3, t registered and stacked as the three channels; control E23ctl has frame t in all three. Plain recipe on the E14 data (the degradation's JPEG step and hue/saturation jitter mix channels) | Recall under 8 px and false alarms: E23 against E23ctl (what time adds), E23ctl against plain (what colour was worth). Not scored on DUT (still images) | 6 runs (3 seeds × 2), each about as long as a plain run |
| 5 | **E24 (done 2026-10-06, 15 §6.19, 16 #81): adopted provisionally.** `--freeze 10` on E22c: DUT AP 0.195 → 0.311 (every seed +0.09-0.14), on sky-only DUT 0.242 → 0.330; daytime Anti-UAV R@10 0.895 → 0.871 | Next (not built): **E24b** `--freeze 5`; **E24c** freeze 10 for 30 epochs then unfreeze for 10 (LP-FT); **E24d** plain recipe + `--freeze 10` (do degradation and pasting still help once the backbone is frozen?). Three seeds each, ~1.7 h a run on an L4 | 9 runs |
| 5b | **E24b / E24c / E24d (done 2026-10-10, 15 §6.19, 16 #90): none adopted; E24 stays the base.** DUT AP 0.166 / 0.227 / 0.169 against 0.311. The gain needs layers 5-9 kept as pretrained for the whole run, and the E22c data | The freeze question is closed at this model size. Next lever on the backbone: E25 | - |
| 5c | **E25 (built 2026-10-08, queued; 16 #85):** RF-DETR Small at 640 on E22c's data, three seeds, scored by the same scripts. A model-family comparison (its own augmentation and schedule), not a one-change arm | Does a foundation backbone beat a frozen COCO backbone on DUT? | 3 runs of 20 epochs, no time cap, resumable (16 #88) |
| 5d | **E39 (built 2026-10-08, queued last; 16 #86):** E24 + DUT train split (5,211 tiles) | How much of the DUT failure is missing data? After it, DUT test is no longer an unseen dataset | 3 runs |
| 5e | **E24 at 80 epochs (queued 2026-10-08, 16 #88):** the candidate final model, three seeds, against E24 at 40 epochs of the same seed | Does the longer schedule cut false alarms on the frozen-backbone recipe as it did on the plain one (§3.4)? | 3 runs, about twice an E24 run each |
| 5b | **E27 (done 2026-10-06, 15 §6.19): not adopted** (tie in-domain, DUT AP +0.08 / 0.00 / +0.01 by seed; same two class rows carried over as from COCO). Was: Objects365-pretrained start (`--model yolo26s-objv1-150.pt`, verified in ultralytics/assets v8.4.0, 20.9 MB) on the **E22c** recipe, three seeds (queued first in `colab_e18.py`) | Does broader pretraining help on DUT? (§9) | 3 runs |
| 6 | E13 (done, §3.7): drone-only training | **Refuted:** equal to plain seed 1 in-domain, worst of all on DUT (0.041). Keep four classes | - |
| 6b | **E34 (done, §3.10):** on the unseen dataset (DUT) boxes of any class reach 2.2-2.6 times as many drones as "drone" boxes (0.353 against 0.138 at 0.55) for 4 times the stray candidates, and still reach more at an equal stray-candidate rate; on the trained datasets it gains no drone and costs 2.5-5 times the candidates. The pipeline already tracks every class; keep it, with a strict track decision. | - | - |
| 7 | Hours of target-free footage from any camera | A false-alarm rate per hour that is not an upper bound | Recording time; shot S6 of the field plan, [20](20_field_sessions_own_footage.md) §4 |
| 8 | Footage with range truth (a drone flown at the camera with its flight log, or PX4 paths through `cuas.sim`) | Is TTC usable as an alert trigger (§2 #16)? | A field session: shots S0-S2 of [20](20_field_sessions_own_footage.md) (2026-10-05) |
| 9 | **E30 (done, §3.10): refuted.** The score gap is per dataset, not per size; a per-size calibration made Halmstad worse | Set thresholds on site footage | - |
| 10 | **E33 (done, §3.10):** centre matching under 12 px changes recall only below 4 px; `w_min` stands | Keep IoU; quote the centre figure for 0-4 px | - |
| 11 | Ask the Drexel iMaPLe lab for LRDDv2/v3 (§10.5) | Real small drones with range truth | An email |

**From E22 on, every run uses the full 40 epochs with no early stopping, is scored on its final weights (`<run>@last`), and is repeated over three seeds** (§3.1, §3.4). If `E18_base_s1_e80` beats `E18_base_s1`, the schedule becomes 80.

---

## 9. Our Recipe Against Published Recipes

Added 2026-10-01 (16 #58) after the question: has every detail of our training been compared with the most successful papers, repositories and models? Until this section it had not. §4 judged each parameter against our own measurements; this section sets our recipe beside four primary sources.

**Sources and how each was read:**
* **The official YOLO26s checkpoint itself.** `yolo26s.pt` stores the arguments it was trained with (`train_args`); read directly with `torch.load`. This is the recipe behind the weights we start from. Its starting model is named `yolo26s-objv1-150.pt`; that this means Objects365 is read from the name (Ultralytics' YOLO26 page does not state the recipe).
* **The 2025 Drone-vs-Bird challenge winner** ([Laroca et al.](https://arxiv.org/abs/2504.19347)), full text.
* **YOLOMG** ([Guo et al.](https://arxiv.org/abs/2503.07115)), full text, for its ablations.
* **D-FINE's repository README** (fine-tuning guidance for a DETR-family detector).
* The two full texts were read through an automated page summary, so check a number against the paper before quoting it outside this project.

**Not covered:** RT-DETR and DEIM configuration files, VisDrone leaderboards, the thermal Anti-UAV challenge winners. Hugging Face was searched for drone-detection models: the most downloaded are community fine-tunes of YOLOv8/11/12 and RT-DETR with under 100 downloads each and no stated recipe, so there is nothing to compare. **None of the alternatives below has been trained here.**

### 9.1 Side by side

| Setting | Ours (E14 recipe) | Official YOLO26s (its own `train_args`) | Drone-vs-Bird 2025 winner | YOLOMG | D-FINE fine-tuning |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Starting weights | COCO `yolo26s.pt` | **Objects365** (`yolo26s-objv1-150.pt`, going by the file name), then COCO | Pretrained YOLOv11 | COCO YOLOv5s | **Objects365 "highly recommended"** |
| Model size | s (9.5 M) | s | **m** | s, slimmed | n/a |
| Input | 640-px tiles at native resolution | 640 | 640; inference on the full frame **plus four overlapping 55% segments** | 640 and 1280 | 640 |
| Epochs / early stop | 40 / patience 15 | **70 / patience 100** | not stated / patience 15 | 100 | n/a |
| Optimizer | AdamW, lr 0.00125, wd 0.0005 | MuSGD, lr 0.00038, final lr × 0.88, wd 0.00027 (evolved values) | SGD, lr 0.01, momentum 0.9 | Adam, lr 0.01 | AdamW, lr 0.0005, **backbone lr 0.000025**, wd 0.0001 |
| Batch | 16 (accumulated to 64) | 128 | 16 | 8 | 32 |
| Mosaic | 1.0 | 0.99 | **0.1** | n/a | n/a |
| Scale jitter | ± 0.5 | ± 0.9 | **± 0.3** | n/a | n/a |
| Rotation / shear | 0 / 0 | ~0 / ~0 | 10° / 10° | n/a | n/a |
| HSV saturation / value | 0.7 / 0.4 | 0.35 / 0.19 | 0.7 / 0.4 | n/a | n/a |
| Mixup / copy-paste | 0 / 0 | 0.05 / 0.40 | **copy-paste of drone and bird patches, degraded first** (blur, noise, pixel dropout, brightness) | n/a | n/a |
| Loss gains box / class | 7.5 / 0.5 | 9.83 / 0.65 | defaults | n/a | n/a |
| Use of time | tracker after the detector | none | **± 6-frame consistency**: missed detections interpolated between neighbours | **3-frame difference map as input** (frames t ± 2, homography-aligned) | none |
| Merge of tile predictions | NMS at IoU 0.5 | n/a | NMS at IoU 0.1 | n/a | n/a |
| Extra training data | Halmstad + Anti-UAV | n/a | challenge data + USC-GRAD-STDdb + another set + **DUT Anti-UAV (10,000 images)** | ARD100 | n/a |

Reported effects in those sources:
* Winner: multi-scale inference raised validation mAP50 from 0.552 to 0.739.
* YOLOMG on ARD100 at 640: the motion-difference input raised AP from 0.33 to 0.78; **its small-object layer added 0.02** (0.76 → 0.78).

### 9.2 What the label assigner gives a small drone (run here)

The installed assigner was run on boxes of each size with perfect predictions (venv, Ultralytics 8.4.162). Positive samples per ground-truth box in the one-to-many head (top-k 10):

| Box side | P3-P5 model (ours) | P2-P5 model (`yolo26s-p2.yaml`) |
| :--- | ---: | ---: |
| 4, 6, 8 px | 5.3 | 5.3 |
| 12, 16 px | 5.3 | **10** |
| 24 px and larger | 10 | 10 |

* **A drone of 16 px or less gets half the positive samples of a larger one.** The small-target assignment (floor each side to 16 px) is what guarantees it any: without it a 4-px box would often contain no anchor point.
* **A P2 head doubles the samples for 12-16 px targets and changes nothing below 8 px** (its floor is 8 px). That matches YOLOMG's ablation: its small-object layer was worth 0.02 AP.
* The official P2 model exists in the installed package: 9.77 M parameters, 27.8 GFLOPs against 22.8 (1.22×).

### 9.3 What this confirms, and what it changes

**Confirmed (no change):**
* **Native-resolution tiles.** The challenge winner got +0.19 mAP50 from slicing the frame into four segments, which still downscales each one; our tiles never downscale. SAHI ([arXiv 2202.06934](https://arxiv.org/abs/2202.06934)) reports the same direction for both training and inference.
* **Fine-tuning from pretrained weights, 640 input, batch 16.** All four sources do the same.
* **The optimizer family does not decide the result.** The winner used SGD, YOLOMG Adam, D-FINE AdamW, the official model MuSGD.
* **No early stopping.** The official model trained with patience 100, as §3.4 now requires.
* **Temporal information matters more than a small-object head** (YOLOMG: +0.45 against +0.02), which is the order §5 already had.

**Changes to the plan, in order of expected value:**

| # | Difference | Evidence | Experiment |
| :-: | :--- | :--- | :--- |
| 1 | **Augmentation for small targets.** We run mosaic 1.0 and scale ± 0.5; the winner cut mosaic to 0.1 and scale to ± 0.3, and added copy-paste of **degraded** drone patches | The winner's recipe; [Kisantal et al.](https://arxiv.org/abs/1902.07296) (+7.1% relative on small objects); our E21 (the detector has never seen a degraded small drone) | Fold into **E22**: (a) degradation on training images; (b) mosaic 0.1 / scale 0.3; (c) copy-paste of shrunk, degraded drone crops onto native frames in place of whole-frame zoom-out mosaics, which also removes the 256-drones-per-image density |
| 2 | **Starting weights.** The official s model was pretrained on Objects365 before COCO, D-FINE recommends the same for fine-tuning, and Ultralytics distributes `yolo26s-objv1-150.pt` | Both sources; [Scale Match](https://arxiv.org/abs/1912.10664) on why pretraining data matters for tiny objects | **E27**: `train_local.py --model yolo26s-objv1-150.pt`. Objects365 has its own dataset terms; fine for research, check before publishing weights |
| 3 | **Temporal input** | YOLOMG's ablation is the strongest single effect in any source read | **E23**, now ahead of E24. Start from YOLOMG's design: frames t ± 2, registered, as a difference map beside the RGB frame |
| 4 | **Backbone learning rate.** D-FINE trains the backbone at 1/20 of the head's rate | D-FINE README; [LP-FT](https://arxiv.org/abs/2202.10054) | **E24** as a softer option than `--freeze 10`. Ultralytics has no per-layer rate argument, so freezing is the cheap first test |
| 5 | **Use of time after detection.** The winner fills missed detections from neighbouring frames | Winner's post-processing | Already covered more generally by the tracker's coasting and low-score second stage; no change |
| 6 | **Model size m** as a reference point | The winner used m | One run after E22, to know what size costs or buys (with E3's n) |
| 7 | **Tile-merge NMS at IoU 0.1** (ours 0.5) | Winner | Cheap evaluation-only check on existing detections |
| 8 | **P2 head** | §9.2 and YOLOMG: little expected below 8 px | **E7 moved down**, after E23 |
| 9 | Official evolved values (loss gains 9.8 / 0.65, translate 0.27, lower HSV) | Tuned for COCO from Objects365 at batch 128; no reason to expect transfer | Not planned |
| 10 | **Training on DUT Anti-UAV.** The winner trained on it; we hold its test split out | Winner's data list | Add DUT's **train** split (5,200 images) to training and keep the test split held out. This is the "more varied data" lever of 15 §6.6 |

---

## 10. Wider Literature Sweep: What We Are Doing Wrong

Added 2026-10-01 (16 #59) after the request for a larger net than §9's four sources. About 30 further sources: arXiv papers and three journal papers (*Sensors*), covering long-range drone datasets, the bird and drone challenges, tiny-object detection and tracking, robustness and domain shift, missing labels, calibration, open-vocabulary detectors and event cameras. They are listed in [01 §2.10, §8.6 and §9](01_research_papers_compendium.md#210-wider-sweep-long-range-data-ground-to-air-detectors-challenges-checked-2026-10-01).

**How they were read:** arXiv abstract pages, and the full text of four of them through an automated page summary (marked in `papers_database.json`). The three journal papers were verified by DOI through Europe PMC (abstracts; the publisher's pages refused the request). Two sources that came up in search were **not** opened and are not used as evidence: the 2023 Drone-vs-Bird challenge review and a 2025 Anti-UAV workshop tracker paper. Nothing in this section has been trained here except §10.3.

### 10.1 How our architecture compares

| Part | Ours | What the literature uses | Verdict |
| :--- | :--- | :--- | :--- |
| Detector family | YOLO26s, one model | Ground-to-air: [SDD-YOLO](https://arxiv.org/abs/2603.25218) is a YOLO26-style model plus a P2 head; the Drone-vs-Bird winner YOLOv11m. Small-bird challenge winners: ensembles of Cascade R-CNN and CenterNet on large backbones (1,600 M parameters), or YOLOv7/8 | **Mainstream.** The ensembles are out of reach of an edge budget and their private-test score is still 30 AP50 |
| Resolution | Native 640-px tiles, never downscaled | [LRDDv3](https://arxiv.org/abs/2605.25942): the same model goes from mAP50 0.54 at 640 input to 0.82 at 1920. Small-bird winners raise resolution and slice; the tracking champion trains on 1280-px slices with 20% overlap | **Right, and at the strong end** |
| Scales looked at | One (native) | Winners add multi-scale inference or test-time augmentation | A gap, but a small one when measured: +0.03 recall on DUT at one false drone per drone gained (§10.3) |
| Small-object head | None (P3 is the finest) | SDD-YOLO adds P2; YOLOMG's was worth 0.02 AP | Minor (§9.2) |
| Label assignment | YOLO26's small-target floor | [RFLA](https://arxiv.org/abs/2208.08738) (receptive-field distance); NWD in the small-bird winner | The built-in answer is present; NWD is the next step: two prize-winning entries in the small-bird challenge used it |
| Use of time in the detector | None | YOLOMG, TransVisDrone, dual-interval cues | **A gap** (E23) |
| Association | Mahalanobis distance to the predicted position; no appearance | The small-bird tracking champion: appearance is "virtually unusable", so expanded boxes with a distance penalty and a smoothed velocity. The challenge scores by Dot Distance because IoU breaks down | **Agrees** |
| Track classification | Hand-built motion and shape features, boosted trees | BiLSTM on coordinates ([Nesteruk et al.](https://doi.org/10.3390/s26030755), trained on simulated tracks); 3D CNN on crops (Akyon et al.) | Fine for ~270 tracks; revisit with more tracks |
| Static-camera motion stage | MOG2 on registered frames, then the detector | [Seidaliyeva et al.](https://doi.org/10.3390/s20143856): background subtraction, then a CNN classifier; its stated limit is a moving background | Same design, same weakness (§2 #7) |
| Foundation models, zero-shot | Not used | [Open-vocabulary detectors on aerial images](https://arxiv.org/abs/2601.22164): best F1 27.6%, the others under 4% recall | Do not expect a zero-shot shortcut; E25 only makes sense fine-tuned |
| Sensor | Frame camera | [EV-UAV](https://arxiv.org/abs/2506.23575): event camera, targets averaging 6.8 × 5.4 px | Out of the current plan; the alternative if frame rate becomes the limit ([18](18_hardware_optimisation_and_realtime_latency.md)) |

**The architecture is not what we are doing wrong.** It matches what current ground-to-air work uses, and on resolution handling it is ahead. The errors are in the data and the evaluation around it.

### 10.2 What we are doing wrong, roughly in order of expected cost

1. **Training on clean images only.** [Michaelis et al.](https://arxiv.org/abs/1907.07484) measured detectors falling to 30-60% of clean performance under corruption, worst for the texture-destroying kinds (noise, JPEG) and mild for blur and contrast. E21 reproduced exactly that pattern on our detector. The remedy in that literature is training-time: degraded or stylised training images, or a physically modelled augmentation ([PhysAug](https://arxiv.org/abs/2412.11807), [OA-DG](https://arxiv.org/abs/2312.12133)). This is E22.
2. **No object-level augmentation and no staging.** The small-bird winner and the Drone-vs-Bird winner both paste objects; one prize-winning entry used hard augmentation for the first half of training and light augmentation after. We run mosaic at 1.0 for 30 of 40 epochs and paste nothing. Folded into E22 (§9.3).
3. **Matching by IoU for targets of a few pixels.** The small-object tracking challenge replaced IoU with Dot Distance for this reason, and NWD exists for the same one. Our `w_min` rests on IoU ≥ 0.3 at 4-8 px, where one pixel of offset costs 0.2-0.4 of IoU. Evaluation-only change: match by centre distance below ~12 px and report both.
4. **One confidence threshold for every size, on uncalibrated scores.** Detector confidence is biased by box size ([Gilg et al.](https://arxiv.org/abs/2112.01901)) and position ([Küppers et al.](https://arxiv.org/abs/2004.13546)). We saw the symptom (scores top out near 0.5; a threshold chosen on one dataset removes most of another's drones; 15 §7.2) and treated it per dataset. Calibrating per size bin on validation detections needs no training. **E30.**
5. **A thin bird class.** Halmstad has 12,791 bird boxes from 51 videos; the small-bird challenge set has 137,121 birds, 95% under 32 px, and its tracking sequel has 211 videos. "Birds are rarely called drones" (15 §7.1) rests on one dataset.
6. **One seed, a selected checkpoint** (§3.1, §3.4), and **no time in the detector** (§5, §9).
7. **No real target with range truth.** `w_min` is measured on shrunken mosaics and time to collision has never been checked against a true range (§2 #16). The one dataset found that has both, LRDDv3 (102,532 4K images at 0-200 m, range on almost every image, birds in the same scenes), **is not downloadable today**: its project page says access is on hold, under a licence with a US export-regulation condition (03). This is a request to send, not a mistake to fix.

**Two faults the reading suggested, which measurement shrank:**

* **Looking at one scale.** Winners add a coarse pass over the whole frame, and E15's recall for large drones was 0.06-0.12. Measured in §10.3: a coarse pass finds a few more large drones and costs as many false ones.
* **Treating sparsely labelled data as fully labelled.**
  * The worry: Halmstad labels one target per frame and Anti-UAV one drone per sequence. Unlabelled *frames* are no longer used as background (14 §5), but an unlabelled second drone inside a training tile would still be trained as "nothing". The literature has remedies for exactly this ([Zhang et al.](https://arxiv.org/abs/2002.05274), [SparseDet](https://arxiv.org/abs/2201.04620)).
  * The count (16 #59): E12 `best.pt` over its own 10,272 training tiles. Confident (≥ 0.4) "drone" boxes that touch no label: **13 in 6,694 Halmstad tiles** (6 videos) and **12 in 3,578 Anti-UAV tiles** (11 sequences).
  * The contact sheets (`C:\cuas_data\runs\unlabelled_halmstad.png`, `unlabelled_antiuav.png`): the Halmstad boxes are specks of 15-40 px in sky and treeline, a few visibly an object; the Anti-UAV boxes are mostly night lights and building edges, with two that look like drones.
  * **So at most a handful of unlabelled drones in 10,272 tiles: not a fault worth an experiment.** E29 (reviewed pseudo-labels) is not scheduled.
  * Limits: the model was trained on these tiles to call such objects background, so this is a lower bound; only "drone" boxes were counted, not unlabelled birds. A stricter check needs weights that never saw the tiles (cross-fitting), worth doing only if a false-alarm analysis points back here.

### 10.3 E28: a second, coarser look at the frame (run here)

Source: `C:\cuas_data\runs\E12_multidata_yolo26s\reports\e28_E12_multidata_yolo26s.txt`.

**Question.** E15 found recall of only 0.06-0.12 for DUT drones wider than 64 px. Native 640-px tiles cut a close drone at tile edges and show it larger than most of the training data. Winners of the challenges run the whole frame as well as slices. Does a coarser look find those drones?

**Method.** `experiments/e28_multiscale_dut.py`: every DUT test image (2,200 images, 2,245 drones) is run three ways with the same weights (E12 `best.pt`) and the same threshold (0.4), and the detections are merged by NMS. *Tiles*: native 640-px tiles, as now. *Half*: the image shrunk 2×, then tiled. *Coarse*: the whole image shrunk into one tile. No training.

| Variant | All | 16-32 px | 32-64 | 64-128 | 128-256 | 256+ | Extra boxes on drones | Background false drones |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiles (now) | 0.141 | 0.10 | 0.25 | 0.10 | 0.06 | 0.01 | 87 | 184 |
| Coarse only | 0.048 | 0.00 | 0.03 | 0.12 | 0.09 | 0.01 | 53 | 60 |
| Tiles + coarse | 0.165 | 0.10 | 0.25 | 0.17 | 0.12 | 0.02 | 133 | 235 |
| Tiles + half + coarse | 0.170 | 0.10 | 0.25 | 0.19 | 0.12 | 0.02 | 148 | 277 |
| Drones in the bin | 2,245 | 386 | 851 | 396 | 362 | 235 | | |

(15 drones are under 16 px; none is found by any variant.)

* **The scale hypothesis is a small part of the answer.** A coarse pass nearly doubles recall for drones of 64-256 px (0.10 → 0.17-0.19 and 0.06 → 0.12), so some were missed because of tile edges or size. But more than 80% of them are still missed with all three scales, and drones over 256 px stay at 0.02 even when shrunk to an ordinary size.
* **The miss is mostly the class, not the scale:** with all three scales, 526 of the missed drones are still boxed as helicopter (298), airplane (215) or bird (13), against 547 with tiles alone. The model sees the object and names it wrong (15 §6.6).
* **It costs about one false drone per drone gained:** tiles + coarse finds 55 more drones and adds 51 background false drones and 46 extra boxes on drones.
* **Decision: not added to the program.** One more tile per frame is cheap, but at this exchange rate it does not pay, and it is a patch on a data problem. The lever for large, close drones is training data that contains them named "drone" (the DUT train split, §9.3), and the drone-only head of E13 for the naming. Re-run E28 on the weights from those runs: a model that knows large drones may gain more from the coarse pass than this one does.
* One model, one threshold, one dataset; no interval was computed.


### 10.4 What the sweep says we are doing right

* **The collapse on an unseen dataset is the field's normal, not our bug.** In the small-bird challenge the baseline fell from AP50 46.4 on the public test to 15.4 on the private one, from the same collection campaign; the winner reached 30.3. Our 0.87 → 0.14 on DUT belongs to the same pattern, and the remedy there was also data and scale, not a new backbone.
* **Native-resolution tiles, a split by video, a wholly held-out dataset, and motion-only association** each match what the strongest entries do.
* **The hardest cases named by the Drone-vs-Bird organisers** ([Coluccia et al.](https://doi.org/10.3390/s21082824)): very distant drones and moving cameras. The first is our stated problem; the second we avoid by design (a fixed camera).

### 10.5 Changes to the plan

| # | Change | Replaces or adds |
| :-: | :--- | :--- |
| 1 | **Ask the Drexel iMaPLe lab for LRDDv2/v3** (access is on hold). If granted: converter, split by clip, recall by width and **by range** (the real `w_min`), time to collision and range against truth, its birds as negatives | A second route to E19 beside Drone-vs-Bird; the only route found to §2 #16 |
| 2 | **E22** as already widened in §9.3, with a frequency-domain augmentation as an optional arm | - |
| 3 | **E29 (reviewed pseudo-labels) not scheduled:** 25 candidate boxes in 10,272 training tiles, mostly clutter (§10.2) | Decided by measurement |
| 4 | **No coarse pass in the program for now** (§10.3). Re-run E28 on weights trained with large drones named "drone" (DUT train split, E13) | Decided by measurement |
| 5 | **E30**: per-size calibration of detector scores, judged by recall at a fixed false-alarm count per size bin | New |
| 6 | Centre-distance matching under 12 px in `cuas/eval.py`, reported beside IoU | Evaluation only |
| 7 | Small-bird challenge images as bird negatives; its tracking set as bird tracks for M2 | After 1 |

---

## 11. Validation and Scoring: What We Use and Whether It Is Right

Added 2026-10-04 (16 #68) after the question: evaluate the type of validation and the way results are scored (MAE, RMSE, recall, F1 and the rest). The short answer: **the validation design is sound; the scoring had three real weaknesses**, two of which had already distorted comparisons in this document. `experiments/e35_metrics.py` computes what was missing, from cached detections.

### 11.1 The validation design

| Element | What we do | Verdict |
| :--- | :--- | :--- |
| Split | Hold-out, by video (Halmstad 80/20) and by sequence (Anti-UAV's own train / val / test) | **Right.** Frames of one video are near-duplicates; a random split by frame would leak |
| A whole dataset held out | DUT test is never trained on | **Right, and the most informative thing we have.** It is what showed that E12-E22 do not transfer |
| Cross-validation (k-fold) | Not used for the detector; used, grouped by video, to calibrate the track classifier | **Acceptable.** Five folds mean five trainings per arm. Three seeds on a fixed split cost less and measure the noise that mattered here. Its cost: all conclusions rest on one particular set of 57 + 91 test videos |
| Halmstad has no test split | Its validation videos are the reported set | **Was a leak, now mostly closed.** While `best.pt` was chosen on validation mAP, the reported videos chose the checkpoint. With final weights and the threshold chosen on Anti-UAV val, nothing is selected on them any more |
| Repeated use of the test sets | The Anti-UAV test and DUT have been scored after every run, and recipe decisions (E14, E22a) were taken on them | **A weakness.** "Score the test once" (17 principle 6) has not been kept. Each look is a small selection step, so the test numbers of an *adopted* recipe are slightly optimistic. Remedy: keep one set unseen until the end (Drone-vs-Bird or own footage) |
| Error bars | Per-video bootstrap; three seeds per arm | **Right.** The bootstrap covers which videos happened to be in the test; the seeds cover training luck. Neither covers a new camera or site |
| Threshold choice | Maximum F1 on pooled Anti-UAV validation, from a five-point grid | **Wrong tool** (§11.2) |

### 11.2 Each score, and whether it fits

| Score | Where we use it | Fits? |
| :--- | :--- | :--- |
| **Recall** (by target width) | The main detector number | **Yes**, but only at a stated false-alarm rate. Quoted at each model's own threshold it compares operating points, not models |
| **Precision** | Reported beside recall | **Weak.** It depends on how often a drone is present. Every evaluation frame has one; a deployed camera sees one rarely. Keep as a diagnostic, never as a headline |
| **F1** | Chooses the threshold | **No.** It weighs a miss and a false alarm equally at the test set's drone rate, which is not the deployment's. It is also flat near its peak, so the choice flips between 0.40 and 0.55 from one run to the next. That flip confounded the E21 and DUT columns of E22a (15 §6.14) and several earlier comparisons |
| **Accuracy** | Track classifier only ("93% on decided tracks") | Not defined for a detector (there is no count of correct "nothing here"). For the classifier it hides the share left undecided, so it is always quoted with it |
| **mAP50, mAP50-95** (Ultralytics, four classes) | Printed during training; used to pick `best.pt` until 2026-10-01 | **Not our objective.** It averages over classes we do not care about equally, and mAP50-95 punishes loose boxes on tiny targets. Now a training monitor only |
| **AP of the drone class** (area under precision-recall) | Not computed before E35 | **Yes: the missing headline number.** It needs no threshold |
| **Recall at a fixed false-alarm budget** | Not computed before E35 | **Yes.** It is the question a deployment asks |
| **ROC AUC** | Track classifier (0.92) | Fine there (two classes, both countable). Not usable for a detector |
| **False alarms per 1000 frames / per hour** | Throughout; per hour only as an upper bound (E20) | **Yes, and per hour is the real target.** It needs hours of target-free footage |
| **`w_min`** | The design number that sets range | **Fragile.** It is the edge of one bin crossing 0.80; in E33 it moved from 12 to 8 px on a bin at exactly 0.80. A fitted recall-against-width curve with an interval is still owed (§4.3) |
| **MAE, RMSE, bias** | Not computed before E35 | **They are for quantities, not for found / missed.** They fit box width and position (which feed range and time to collision), and range and time to collision themselves. For the last two there is no truth in any dataset we have |
| **ECE, Brier score** (calibration) | Track classifier (ECE 0.02-0.07) | Now also the detector (E35): its scores are not probabilities, and how wrong they are depends on the dataset |
| **IoU threshold for a match** | 0.3 | Checked in E33: only the 0-4 px bin is sensitive to it |
| **Paired difference at matched validation recall** | Every comparison since E14 | A good idea with a flaw: it still fixes one threshold per model. A paired difference in AP would be cleaner |
| **Time to declare, declarations per hour** | E20 | **Yes: the system-level scores.** In-domain only so far |
| **Tracking scores (MOTA, IDF1, HOTA)** | Not used; we count track coverage and stray tracks per minute | Low priority while every video has one target; identity switches are unmeasured |
| **Training loss** | Fit diagnosis only (§1) | Right use |

### 11.3 E35: the scores that were missing

Source: `C:\cuas_data\runs\e35_metrics.txt`. Final weights, native frames, "drone" boxes at IoU ≥ 0.3.

| Anti-UAV test | Plain: seeds 1 / 2 / 0 | E22a: seeds 1 / 2 / 0 | E22b: seed 1 |
| :--- | ---: | ---: | ---: |
| AP | 0.856 / 0.858 / 0.845 = **0.853** | 0.894 / 0.882 / 0.891 = **0.889** | 0.899 |
| Recall at ≤ 10 background false drones per 1000 frames | 0.836 / 0.828 / 0.830 = 0.831 | 0.880 / 0.850 / 0.839 = 0.856 | 0.844 |
| Recall at ≤ 30 per 1000 | 0.876 / 0.863 / 0.869 = 0.869 | 0.906 / 0.882 / 0.887 = 0.892 | 0.889 |
| ECE of the raw score | 0.09-0.10 | 0.10-0.12 | 0.13 |

* **E22a's clean-data gain is confirmed without any threshold:** AP +0.036, against a seed spread of 0.013.
* **Halmstad val is saturated:** AP 0.993-0.997 and recall 0.996-1.000 at 10 false alarms per 1000 frames, for every run. So the "Halmstad cost" of E22a (recall 0.58 against 0.74 at threshold 0.55) is not a worse model. It is the same ranking with lower scores. Halmstad native frames can no longer tell our models apart; only its shrunken and degraded versions can.
* **The detector's score is not a probability, and the error depends on the dataset:** ECE about 0.10 on Anti-UAV and 0.25-0.28 on Halmstad, where correct boxes score around 0.6.
* **E22b adds nothing on these scores** (AP 0.899 against E22a's 0.882-0.894; recall at 10 per 1000 is 0.844 against 0.839-0.880).

**Box error of correct detections** (plain seed 1; native sets plus the small-target sets; the other runs agree within 0.2 px):

| True width | Boxes | Width bias | Width MAE | Width RMSE | Bias, % of width | MAE, % | Centre error |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| under 8 px | 1,780 | +1.2 px | 1.5 px | 1.8 px | **+25%** | 30% | 0.6 px |
| 8-16 px | 2,561 | +0.6 px | 1.6 px | 2.0 px | +5% | 14% | 0.7 px |
| 16-32 px | 1,275 | +1.0 px | 2.4 px | 3.5 px | +4% | 11% | 1.2 px |
| 32-64 px | 506 | +1.5 px | 5.7 px | 7.6 px | +3% | 13% | 3.0 px |
| over 64 px | 1,498 | −4.4 px | 16.6 px | 27.6 px | −1% | 11% | 9.8 px |

* **The centre is good at every size** (0.6 px on a target under 8 px), which is why the tracker works on tiny targets.
* **The width is biased large on tiny targets: +25% under 8 px.** Range from box width (09) is then read about 20% short, and the error falls to 5% by 8-16 px.
* **This explains a known defect.** A drone approaching from 6 to 12 px true width is measured as about 7.5 to 12.6 px: the box grows 1.7 times when the drone grew 2 times. Time to collision from box growth is therefore read long at a few pixels, which is what the simulator showed (§2 #15).
* **The error is steady in proportion above 8 px** (MAE 11-14% of width). A single box gives range to about that accuracy at best, before the assumed drone size is even considered.
* Limits: the truth here is a hand-drawn label, itself loose (15 §7.2), and the small-target sets are shrunken frames whose labels were scaled. RMSE is 1.2-1.7 times the MAE, so the errors have a tail but are not dominated by a few outliers.

### 11.4 What changes

| # | Change | Why |
| :-: | :--- | :--- |
| 1 | **Headline detector scores from now on: drone AP and recall at 10 false alarms per 1000 frames**, mean and spread over seeds. `e35_metrics.py` after every run | No threshold to confound the comparison |
| 2 | **The maximum-F1 threshold is kept only as a convenience for the per-width table.** A deployed threshold comes from a false-alarm budget on site footage (§3.10) | F1 is the wrong objective and an unstable one |
| 3 | **Keep one dataset unscored until a recipe is final** (Drone-vs-Bird when it arrives, or own footage) | The current test sets have been looked at after every run |
| 4 | **Halmstad native frames are a sanity check, not evidence.** Use its shrunken and degraded sets | Saturated at AP 0.995 |
| 5 | **Quote the width bias beside any range or time-to-collision figure**; correct it per size when range is built | +25% under 8 px |
| 6 | Still owed: a fitted recall-against-width curve for `w_min`; a paired AP difference in `compare_runs.py` | §4.3 |
| 8 | **Score the in-scope part of DUT too (2026-10-06, E37):** `e37_dut_sky.py` splits DUT into images whose drones are all on sky and the rest; quote the sky part beside the pooled DUT score | E22c looked level with plain on pooled DUT AP but was worse on sky at strict budgets in every seed (15 §6.19) |
| 7 | **Score the in-scope condition (2026-10-05, E36):** E35 adds the daytime Anti-UAV videos (`antiuav_day`); night is out of scope (13 §6) and its lights were 21-40% of the confident false drones. On DUT a quarter of confident false drones are burned-in text: mask overlays before quoting a DUT false-alarm rate | Counts that mix out-of-scope scenes and dataset artefacts mislead recipe choice (15 §6.18) |

### 11.5 What this review got wrong on the way

* **The threshold confound was visible for days before it was fixed.** 16 #50 already recorded that comparing runs at their own thresholds compares operating points; the remedy then (`--match`) still matched on one point. E22a's degradation table was first reported at 0.40 against 0.55 with a caveat, when a threshold-free score took one minute to compute.
* **"Halmstad recall fell" was reported as a cost of E22a twice** (15 §6.14, first and three-seed versions) before AP showed there was no loss in ranking.
* **Box error had never been measured**, although range and time to collision are built on box width and the time-to-collision bias had already been seen in simulation.

