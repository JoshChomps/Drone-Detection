# Hardware, Optimisation & Real-Time Tracking Latency

What the system needs to run on in the field, how to make the pipeline cheaper, and how much delay real-time tracking can tolerate. This is a **first-pass analysis to start the discussion**: the arithmetic is done, the hardware facts are checked at their sources, and every number that still needs measuring is marked. Nothing here should be bought on these numbers alone. §8 says what to measure first.

> [!NOTE]
> Written 2026-09-28. Builds on [13](13_system_requirements_and_geometry_budget.md) (geometry, latency budget, §6 decisions), the measured $w_{\min}$ of 8 px ([15](15_detector_training_methods_and_findings.md)), and the pipeline in `cuas/` ([14](14_build_log_and_experiments.md)). The numbers come from `cuas.tiling.grid_tiles` and the IMX477 datasheet values below; the script is reproduced in §9.

---

## 1. What Decides the Hardware

Three numbers, in this order:
1. **Pixels per second from the camera** (§2). This fixes range, field of view and frame rate together.
2. **Frame rate needed for tracking** (§3). Fast crossing targets, wing-flap cues and confirmation delay all need frames often.
3. **Detector tiles per second** (§4). Crops around motion plus a periodic full scan, compared with what each accelerator can run.

Latency (§5) matters less than you might expect for *warning* time, but a lot for *handing a track to a narrow camera*.

---

## 2. The Camera: a Fixed Pixel Budget

The candidate camera is the **Raspberry Pi HQ camera** (Sony IMX477; 4056 × 3040, 1.55 µm pixels, rolling shutter, C/CS mount; checked on the [Raspberry Pi camera page](https://www.raspberrypi.com/documentation/accessories/camera.html), 2026-09-28). Its video modes are 2028 × 1080 at 50 fps, 2028 × 1520 at 40 fps and 1332 × 990 at 120 fps. Full resolution runs at ~10 fps (13 §6).

With a 16 mm lens and the measured $w_{\min}$ = 8 px:

| Mode | $f_x$ (px) | HFOV | Pixel rate | Mini (0.25 m) | FPV (0.30 m) | Mavic (0.35 m) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4056 × 3040, ~10 fps | 10,323 | 22.2° | 123 MP/s | 323 m | 387 m | 452 m |
| 2028 × 1520 (2×2 binned), 40 fps | 5,161 | 22.2° | 123 MP/s | 161 m | 194 m | 226 m |
| 2028 × 1080 (binned), 50 fps | 5,161 | 22.2° | 110 MP/s | 161 m | 194 m | 226 m |

**The sensor delivers about the same ~120 MP/s in every mode.** You choose how to spend it:
* **Range + field of view, few frames:** full resolution at 10 fps.
* **Field of view + frames, half the range:** binned at 40 fps.
* **Range + frames, half the field of view:** binned at 40 fps **with a 32 mm lens** (f_x back to ~10,300 px, HFOV ~11°). Covering the same sector then takes twice the cameras.

Range, field of view, frame rate: **pick two per camera.** This is the central hardware trade-off, and it is why the second-camera decision (13 §6) matters: one camera could be spent on range, the other on frame rate. A sensor with a higher pixel rate (industrial USB3 or MIPI cameras) moves the whole trade-off, at higher cost and with its own driver work; not yet surveyed.

---

## 3. What Real-Time Tracking Needs From the Frame Rate

### 3.1 How far a target moves between frames
A 40 m/s FPV crossing the view (the worst case; a head-on approach barely moves in the image):

| Range | Full res, 10 fps | Binned, 40 fps | Binned, 50 fps |
| :--- | ---: | ---: | ---: |
| 100 m | **413 px / frame** | 52 | 41 |
| 200 m | 206 | 26 | 21 |
| 387 m (first detection) | 107 | 13 | 11 |

At 10 fps a close crossing drone jumps 413 px, about 50 of its own widths and more than half a 640-px crop. Consequences:
* **Box overlap (IoU) association fails.** ByteTrack-style IoU matching assumes boxes overlap frame to frame. The tracker must gate on the **Kalman-predicted position** with a distance gate, and crops must be cut at the predicted position, not the last one.
* **At 40-50 fps the same target moves 11-52 px:** easy association, and crops can simply follow the track.

### 3.2 Confirmation delay is paid in frames
Doc 13 §6 budgets **0.3 s** to confirm a track. Confirming over 3 frames costs 3 frame intervals plus pipeline latency (§5):
* **10 fps:** 0.3 s in frame intervals alone, ~0.4-0.5 s with the pipeline. **Over budget.**
* **40 fps:** 75 ms + pipeline, ~0.15 s. **Within budget.**

A frame is 100 ms at 10 fps; no inference optimisation can recover that.

### 3.3 Wing-flap cues need frames too
Separating birds by wing-flap periodicity needs a frame rate of at least twice the flap frequency (Nyquist; E9). Flap rates range from a few hertz for large birds to over 10 Hz for small ones; exact figures still need a verified source before E9. **10 fps cannot see flapping above 5 Hz.**

**First evidence (M2, 2026-09-28, [14 §5](14_build_log_and_experiments.md#5-findings-so-far)):** on Halmstad tracks, shape/flap features alone lose accuracy at 10 fps (2-s windows: AUC 0.777 → 0.687). The full classifier barely changes (whole tracks 0.923 → 0.924), because motion features carry most of the separation. So 10 fps costs flap cues and some accuracy on short tracks (1 s: 0.876 → 0.844), not the classifier as a whole. This is still oracle boxes on one dataset.

### 3.4 Consequence
**Full resolution at 10 fps gives the most range but tracks poorly. Binned at 40 fps tracks well but halves the range.** One camera cannot do both. The candidates:
* **(a)** two cameras: a full-res long-range cueing camera plus a binned fast one;
* **(b)** switch a camera's mode, or its region of interest, once a track exists;
* **(c)** a higher-throughput sensor.

This is the first thing to discuss (§10).

---

## 4. Compute: Detector Tiles per Second

### 4.1 Demand
**Tiles per second = frame rate × crops per frame (K) + tiles per full scan × scan rate.** A full scan covers the frame with 640-px tiles at 20% overlap (`cuas.tiling.grid_tiles`): **48 tiles** at full resolution, 12 binned (2028 × 1520), 8 at 2028 × 1080.

| Setup | K = 2 | K = 5 | K = 10 |
| :--- | ---: | ---: | ---: |
| Full res 10 fps, scan every 2 s | 44 | 74 | 124 |
| Full res 10 fps, scan 1 Hz | 68 | 98 | 148 |
| Full res 10 fps, scan 2 Hz | 116 | 146 | 196 |
| Binned 40 fps, scan every 2 s | 86 | 206 | - |
| Binned 40 fps, scan 2 Hz | 104 | 224 | - |

K (motion proposals plus tracks per frame) is **unknown until the motion cue runs on real sky** (E6). It is the single most important unmeasured number for hardware choice.

### 4.2 Capacity (640-px tiles per second, batch 1)
Published by Ultralytics (docs checked 2026-09-28: [YOLO26](https://docs.ultralytics.com/models/yolo26/), [Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/), [Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/)):

| Device | Model, runtime | ms / 640 | Tiles / s |
| :--- | :--- | ---: | ---: |
| Raspberry Pi 5 CPU | YOLO26n, NCNN FP32 | 67.0 | 15 |
| Jetson Orin Nano Super | YOLO26n, TensorRT FP16 | 4.57 | 219 |
| Jetson Orin Nano Super | YOLO26n, TensorRT INT8 | 3.80 | 263 |
| Jetson Orin NX 16 GB | YOLO26n, TensorRT FP16 | 4.13 | 242 |
| NVIDIA T4 (reference) | YOLO26s / n, TensorRT | 2.5 / 1.7 | 400 / 588 |

Not published or not yet measured:
* **YOLO26s on Jetson.** It has 3.8× the FLOPs of n (20.9 vs 5.5 B). A rough guess is 2-3× slower than n, i.e. ~70-110 tiles/s on the Orin Nano Super. **Unmeasured.**
* **The laptop RTX 4060.** The only number so far is 31 ms per tile inside the E11 evaluation (batch 8, FP32, PyTorch, Windows, with tiling overhead), which is not a runtime benchmark. **Measure with TensorRT (E16).**
* **Hailo accelerators** (AI HAT+ for the Pi 5). Community reports give ~28-30 fps for YOLOv8s on the Hailo-8L, but those are secondary sources, and **whether YOLO26 compiles for Hailo is unverified**.

### 4.3 What the comparison says
* **The Pi 5 CPU alone cannot run detection** (15 tiles/s against a minimum of ~44). It stays a capture and motion-cueing node.
* **An Orin Nano Super running YOLO26n fits full-res 10 fps** with K up to ~10 and a 2 Hz scan (196 of 219 tiles/s), but with no headroom. **With YOLO26s it likely fits only K ≈ 5 and a slow scan.** So **E3 (does YOLO26n keep $w_{\min}$ = 8 px?) is a hardware decision**, not only a model one.
* **A laptop-class GPU** almost certainly fits everything. It costs mains power and bulk in the field.
* **Two cameras double the demand.**

---

**Architecture add-ons cost tiles/s** (E18, 2026-09-30, 14 §5).
* YOLOBirDrone's full set (deformable convs + MPDA + RMPDA) costs ~1.8× per 640 tile on the laptop GPU and ~1.6× on a 4-thread CPU (ONNX Runtime, the Pi stand-in). The Pi 5 row above would drop by that factor.
* ECA channel attention costs ~5%.
* Deformable convs export to ONNX (opset 19) and run in ONNX Runtime. TensorRT and Hailo support are unverified.
* Adopt an add-on only if E18 shows a gain beyond seed noise.

## 5. Latency: Glass to Confirmed Track

### 5.1 Where the time goes
Estimates for the research setup: Pi 5 capture and motion cueing, laptop detection over wired Ethernet. **Every row is to be measured** (E6, E16).

| Stage | Full res, 10 fps | Binned, 40 fps | Notes |
| :--- | ---: | ---: | :--- |
| Exposure + rolling-shutter readout | ≤ 100 ms | ≤ 25 ms | One frame period at most |
| ISP and buffering to the application | 10-100 ms | 5-25 ms | Depends on queue depth. Keep it at 1 |
| Motion cueing on the Pi 5 | **unknown** | **unknown** | MOG2 on 12 MP is likely too slow at full scale. `MotionCue` has `scale` (E6) |
| Crop encoding + Ethernet | ~5-10 ms | ~5-10 ms | 5 crops ≈ 0.3 MB per frame. **Lossless or high-quality only:** compression erases small drones (E5) |
| Detection (batch of K crops) | ~10-30 ms | ~10-30 ms | E16 |
| Tracking + filter | < 5 ms | < 5 ms | 13 §3 |
| **Per-frame total** | **~0.15-0.3 s** | **~0.05-0.1 s** | |
| Confirmation (+2 frames) | +0.2 s | +0.05 s | §3.2 |
| **To a confirmed track** | **~0.35-0.5 s** | **~0.1-0.15 s** | Budget in 13 §6: 0.3 s |

### 5.2 How much latency matters
* **For warning time, little.** An FPV first detected at 387 m has ~8 s before a 50-m zone. Half a second of pipeline latency is ~6% of that. Range, set by the lens and $w_{\min}$, dominates (13 §3).
* **For handing a track to the narrow PTZ camera, a lot.** A system that is L seconds behind points the PTZ where the drone *was*. At 40 m/s and L = 0.3 s that is 12 m: 3.4° at 200 m, or 6.9° at 100 m, which is as wide as a 5-10° PTZ view. Missing the handoff wastes the 1-3 s slew.
* **Therefore:**
  * **Timestamp every frame at exposure** (sensor timestamp, not arrival time). Run the tracker in capture time, and predict forward to the moment a command takes effect.
  * **Jitter matters more than the mean.** A steady 200 ms can be predicted away; an unpredictable 50-500 ms cannot.
  * **Never let queues grow.** When a stage falls behind, drop to the newest frame instead of processing a backlog; a backlog turns into latency.

---

## 6. Hardware Options

Verified facts as of 2026-09-28. Prices are only listed where checked.

| Option | What it is | For | Against | Open questions |
| :--- | :--- | :--- | :--- | :--- |
| **A. Pi 5 + laptop** | Pi 5 captures and motion-cues; the laptop's RTX 4060 detects and tracks over wired Ethernet | **Already owned.** Fastest GPU available; iterate in Python | Laptop needs mains power in the field; two boxes and a cable | Motion-cueing cost on the Pi (E6); laptop tiles/s (E16) |
| **B. Pi 5 + AI HAT+** | Hailo NPU on the Pi: AI HAT+ **13 TOPS (Hailo-8L)** or **26 TOPS (Hailo-8)**, INT8; **AI HAT+ 2: Hailo-10H, 40 TOPS INT4, 8 GB** ([Raspberry Pi docs](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html)) | One small, low-power box; camera stack integrates with the NPU | **YOLO26 support unverified.** INT8/INT4 quantisation may cost small-target recall. ~30 fps (YOLOv8s, community reports) covers crops only, not full scans | Does YOLO26 compile (Hailo Dataflow Compiler)? $w_{\min}$ after quantisation |
| **C. Jetson Orin Nano Super** | 67 TOPS, 1,024 CUDA cores, 6-core Arm CPU, up to 25 W, **US$249** developer kit ([NVIDIA](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)) | TensorRT path already assumed in 05 §4 and 13 §4; YOLO26n at 4.57 ms (FP16) | YOLO26s likely too slow for full scans (§4.3). Pi HQ camera support needs a third-party driver (unverified) | Keep the Pi as capture node and send crops, or connect the camera directly? |
| **D. Jetson Orin NX 16 GB** | 100 TOPS; YOLO26n 4.13 ms (FP16) | Headroom for two cameras or YOLO26s | Cost and power above C | Only if the measured demand needs it |

---

## 7. Optimisation Levers, Largest First

| Lever | Effect on tiles/s or latency | Risk | Checked by |
| :--- | :--- | :--- | :--- |
| **Motion cueing** (planned) | 48 tiles per frame → K crops | Misses hovering drones → hence the periodic full scan | E6 |
| **Adaptive full-scan rate** | Scan often only where tracks are absent or the sky is busy | Slower pickup of hovering drones | Replay on own footage |
| **Smaller crops for small proposals** (320 px instead of 640) | ~2-4× less compute per crop. The detector is fully convolutional and targets are never rescaled, so it can run at 320 | Less context around the target; recall must be re-measured | **New: E17** |
| **TensorRT FP16** | Orin Nano Super: PyTorch 15.6 → 4.57 ms (3.4×) | None known for accuracy | E16 |
| **YOLO26n instead of s** | 3.8× fewer FLOPs | Small-target recall may drop | **E3 (now decides hardware)** |
| **Batch the K crops of a frame** | Higher throughput per frame | Adds a little latency per crop | E16 |
| **Track, then detect** | Confirmed tracks get a small crop every frame; full detection less often | A lost track needs re-acquisition | M2 |
| **NMS-free head** (`nms=False`) | Removes NMS time and its jitter | Possible small-target cost | E4 |
| **INT8** | A further ~17% on the Orin Nano Super (4.57 → 3.80 ms) | Quantisation can erase few-pixel targets | Re-measure $w_{\min}$ |
| **Don't:** downscale frames or compress crops hard | - | Throws away the pixels that set range (13 §2) | E5 |

---

## 8. Recommended Path: Measure Before Buying

1. **Now, laptop only (no purchase):** E16, a latency benchmark on the RTX 4060 (PyTorch vs TensorRT, FP16, batch 1 vs K, 640 vs 320, YOLO26n vs s), plus E3 (n vs s on the existing evaluation sets). Run it when the GPU is free (E14 is paused for another project).
2. **Capture node (already planned, M3):** Pi 5 + HQ camera + 16 mm lens. Record real sky at full resolution and in binned 40 fps mode, and run `cuas.replay` on it. This measures **K**, motion-cueing time on the Pi (E6), and the real false-alarm rate.
3. **Then choose the accelerator from measured demand** (§4): up to ~100 tiles/s with YOLO26n and a clean 8 px → Orin Nano Super; more than that, or YOLO26s required → laptop-class GPU or Orin NX. Hailo only if YOLO26 support is confirmed and INT8 keeps $w_{\min}$.
4. **Second camera** after 1-3, using the §3.4 trade-off.

---

## 9. Reproduce the Numbers

```python
import math
from cuas.tiling import grid_tiles

PIX = 1.55e-6                      # IMX477 pixel pitch (m)
fx = 16e-3 / PIX                   # 16 mm lens at full resolution -> 10,323 px (binned: divide by 2)
print(fx * 0.30 / 8)               # FPV range at w_min 8 px: 387 m
print(len(grid_tiles(4056, 3040))) # 48 tiles per full scan
print(fx * 40 / 100 / 10)          # 40 m/s crossing at 100 m, 10 fps: 413 px per frame
```

---

## 10. Decisions to Discuss

| Question | Why it matters | Options |
| :--- | :--- | :--- |
| Range or frame rate for the first camera? | §3.4: one camera cannot do both | Full res 10 fps (range) · binned 40 fps (tracking) · 32 mm binned (both, half the view) |
| What is the second camera for? | Stereo (metric range), wide + narrow, or long-range + fast | Decide after measuring K and tracking on real footage |
| Where does detection run in the field? | Power, weight, cost | Laptop (owned) · Orin Nano Super · Orin NX · Hailo |
| Is mains power available at the test site? | A laptop GPU draws far more than a 25 W Jetson | Site survey |
| Real-time budget for confirmation | 0.3 s is met at 40 fps, not at 10 fps | Keep 0.3 s, or accept ~0.5 s |
