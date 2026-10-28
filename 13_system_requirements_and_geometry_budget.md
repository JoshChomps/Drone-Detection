# System Requirements & Geometry Budget

The documents before this one choose *algorithms*. This one sets the **numbers the algorithms must meet**: how many pixels a drone covers, how far away it can be detected, how much warning that buys, and what the latency budget really is. The decisions in §6 and the reference scenarios in §6.1 turn these into a camera budget. Rerun them for the real site before buying hardware.

> [!NOTE]
> Added 2026-09-21. All tables below were computed from the formulas shown, not estimated. Rerun the snippet in §2.3 with your own sensor and lens.
>
> **Revised 2026-09-22.** Three corrections to the first version:
> * **§2.4 (new): elevation geometry.** Distant drones appear *near the horizon*, so a camera looking up at 15°+ cannot see low-flying drones until they are close.
> * **§3.1 (new): compute.** Tiling a full 4K frame costs ~80 ms on a T4, so the earlier 30-60 ms detection figure was unsupported.
> * **§4: minimum system.** The previous version implied the PTZ extends detection range. It cannot: the *cueing* camera sets detection range.
>
> **Decisions recorded 2026-09-22.** §6 is filled in (academic, detection only, all four threat tiers, 5 s timeline). §6.1 adds three reference site scenarios computed with `tools/geometry_budget.py`.

---

## 1. The Chain That Sets Warning Time

```
lens + sensor  ──►  pixels on target  ──►  max detection range  ──►  warning time  ──►  what you can do in time
  (focal length)     (≥ ~10 px needed)       (per drone size)          (range ÷ speed)     (alert, slew, human decision)
```

Detector choice, Jev, and every millisecond of inference sit *downstream* of this chain. A better model can lower the pixel threshold somewhat. It cannot create pixels the lens did not capture.

---

## 2. Pixels on Target: The Constraint That Dominates Everything

### 2.1 Formula
For an image $W_{\text{img}}$ pixels wide with horizontal field of view $\text{HFOV}$:

$$f_x = \frac{W_{\text{img}}/2}{\tan(\text{HFOV}/2)}, \qquad w_{\text{px}} = \frac{f_x \cdot S}{R}, \qquad R_{\max} = \frac{f_x \cdot S}{w_{\min}}$$

where $S$ is the drone's size (m), $R$ its slant range (m), and $w_{\min}$ the smallest box your detector reliably finds.

### 2.2 Tables (4K sensor, 3840 px wide)

**Pixels across a 0.35 m (Mavic-class) drone:**

| HFOV | $f_x$ (px) | 100 m | 200 m | 300 m | 500 m | 1,000 m |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 90° | 1,920 | 6.7 | 3.4 | 2.2 | 1.3 | 0.7 |
| 60° | 3,326 | 11.6 | 5.8 | 3.9 | 2.3 | 1.2 |
| 30° | 7,166 | 25.1 | 12.5 | 8.4 | 5.0 | 2.5 |
| 10° | 21,946 | 76.8 | 38.4 | 25.6 | 15.4 | 7.7 |
| 5° | 43,975 | 153.9 | 77.0 | 51.3 | 30.8 | 15.4 |

**Maximum detection range (m) at ≥ 10 px / ≥ 5 px:**

| HFOV | VFOV (16:9) | Small quad (0.25 m) | Mavic-class (0.35 m) | Shahed-class (2.5 m span) | Cameras for 360° azimuth |
| :--- | ---: | :--- | :--- | :--- | ---: |
| 90° | 58.7° | 48 / 96 | 67 / 134 | 480 / 960 | 4 |
| 60° | 36.0° | 83 / 166 | 116 / 233 | 831 / 1,663 | 6 |
| 30° | 17.1° | 179 / 358 | 251 / 502 | 1,791 / 3,583 | 12 |
| 10° | 5.6° | 549 / 1,097 | 768 / 1,536 | 5,486 / 10,973 | 36 |

Reading the tables:
* A single wide camera (90°) sees small drones **only within ~50-130 m**. That is about 1-3 s of warning against a 40 m/s FPV.
* The camera counts cover **one row** of azimuth only. A narrow lens also has a narrow *vertical* field of view, so covering more elevation needs more rows (see §2.4 for which elevations matter).
* Narrow lenses buy range but need many fixed cameras to cover the sky continuously. A PTZ on a search pattern covers it only intermittently, revisiting each direction every few seconds, which is too slow for fast threats. Hence the **fixed cueing cameras + narrow PTZ confirmation camera** design in [08](08_camera_angles_and_operational_pov_selection.md).
* **The 10-px threshold is a working assumption.** Measure recall binned by box width on your own footage ([10 §4.1](10_adverse_weather_datasets_and_benchmarks_2026.md#41-detection-coco-conventions)) and replace it with your measured value.
* These figures are *upper bounds*. They ignore lens blur, atmospheric haze ([10 §3](10_adverse_weather_datasets_and_benchmarks_2026.md#3-physical-models-of-weather-degradation)), compression artefacts and heat shimmer, all of which cost range. Slant range also includes altitude.

### 2.3 Recompute for your hardware
```python
import math

def f_px(hfov_deg, width_px=3840):
    return (width_px / 2) / math.tan(math.radians(hfov_deg / 2))

def max_range_m(hfov_deg, target_m, min_px=10, width_px=3840):
    return f_px(hfov_deg, width_px) * target_m / min_px

print(round(max_range_m(30, 0.35)))   # 251 m for a Mavic-class drone at 30° HFOV
```

### 2.4 Elevation: Where Distant Drones Actually Appear
A drone $\Delta h$ metres above the camera, at horizontal distance $d$, appears at elevation $\text{el} = \arctan(\Delta h / d)$:

| Height above camera | 50 m away | 100 m | 200 m | 500 m | Last distance above 15° elevation |
| :--- | ---: | ---: | ---: | ---: | :--- |
| 5 m (low FPV) | 5.7° | 2.9° | 1.4° | 0.6° | **19 m** (0.5 s at 40 m/s) |
| 30 m | 31.0° | 16.7° | 8.5° | 3.4° | 112 m (2.8 s) |
| 100 m (survey / recon height) | 63.4° | 45.0° | 26.6° | 11.3° | 373 m (9.3 s) |

This changes how Case 1 ([08](08_camera_angles_and_operational_pov_selection.md)) should be set up:
* **Distant drones sit within a few degrees of the horizon.** A camera looking "up at 15°, 85°" sees mostly *nearby* drones, and a low-flying FPV only in its last half-second.
* **So the long-range camera should look low.** Tilt it until its lower edge sits just above the local skyline (trees, buildings). A 60° 4K camera with its lower edge at 2° covers elevations of about 2°, 38°. That region is still mostly sky, and a fixed camera can **mask everything below the skyline** once, keeping most of Case 1's clean-background advantage.
* **Drones flying *below* the skyline** (in front of trees or buildings) are the Case 2 problem. **In scope since 2026-10-22 (user):** a tilted, masked camera still has trees in frame (a ragged treeline, branches above the mask, a drone crossing the treeline), so the system must work with trees behind or next to the drone, and the field test scores it as its own condition (experiments/field_test.py, 14 §5).

---

## 3. Latency Budget

**Warning time** = detection range ÷ closing speed:

| Detection range | 15 m/s (survey quad) | 40 m/s (fast FPV) | 50 m/s |
| :--- | ---: | ---: | ---: |
| 100 m | 6.7 s | 2.5 s | 2.0 s |
| 250 m | 16.7 s | 6.2 s | 5.0 s |
| 500 m | 33.3 s | 12.5 s | 10.0 s |
| 1,000 m | 66.7 s | 25.0 s | 20.0 s |

**Work backwards from the response timeline**, not from a round number. The drone must still be outside the protected zone when the response completes:

$$R_{\text{detect}} \;\geq\; v \cdot \big(t_{\text{confirm}} + t_{\text{identify}} + t_{\text{decide}} + t_{\text{respond}}\big)$$

*Example:* 0.3 s to confirm the track, 2 s to slew the PTZ and look, 5 s for a human decision and 3 s for the response to take effect is ~10 s. Against a 40 m/s FPV that needs about **410 m** of detection range. At 10 px on a 0.35 m drone that means $f_x \geq$ 11,800 px, i.e. **HFOV ≤ ~18.5°** on a 4K sensor. Halving the timeline needs ~205 m, i.e. HFOV ≤ ~36°.

Where the time goes (targets, to be measured on the Jetson):

| Stage | Budget | Notes |
| :--- | :--- | :--- |
| Capture + decode | ≤ 1 frame (33 ms at 30 FPS) | Hardware decode on Jetson |
| Detection | ~80 ms per *full* 4K frame on a T4; less with motion cueing | See §3.1. Measure on the device |
| Tracking + IMM-EKF | < 5 ms | CPU |
| Track confirmation | 3-10 frames (0.1-0.33 s) | The largest *compute-side* term. Trades false alarms against delay |
| Rules layer | < 1 ms | Always on time |
| Jev advice | 70-500 ms, async | Never on the critical path ([12 §3](12_typesafe_jev_system_one_c2_threat_engine.md#3-architecture-rules-set-the-floor-jev-can-raise-it-a-human-decides)) |
| **PTZ slew + confirmation look** | **1-3 s** | Mechanical. Larger than every compute stage combined |
| **Human decision** | **seconds** | Policy, not engineering |

**Conclusion:** at these ranges, each 100 m of extra detection range (2.5 s at 40 m/s) is worth far more than shaving inference milliseconds.

### 3.1 Compute: What Full-Resolution Detection Costs
Tiling a 3840×2160 frame with 20% overlap takes **32 tiles of 640 px** (or 8 tiles of 1280 px, which is the same compute). Overlap means about **13.1 MP** is processed for an 8.3 MP frame. With YOLO26s (20.9 GFLOPs per 640 image, 2.5 ms on a T4 per the [Ultralytics table](https://docs.ultralytics.com/models/yolo26/)):

| Scheme | Tiles per second at 30 FPS | Compute | T4 batch-1 estimate |
| :--- | ---: | ---: | :--- |
| Full frame, every frame | 960 | ~20 TFLOP/s | ~80 ms/frame, i.e. **~12 FPS** |
| Motion-cued: crops around 5 tracks/motion blobs at 30 Hz + full scan at 2 Hz | 214 | ~4.5 TFLOP/s | ~4.5× cheaper |
| Same, 10 crops | 364 | ~7.6 TFLOP/s | ~2.6× cheaper |

Implications:
* **Full-frame tiling at 30 FPS does not fit** even a datacentre T4, let alone a Jetson. Downscaling the frame instead throws away exactly the pixels §2 says you need.
* **Use motion cueing.** Case 1's stationary camera makes cheap full-resolution background subtraction possible (MOG2 or frame differencing, GPU-accelerated). It proposes regions, and the CNN runs only on crops around proposals and existing tracks. A periodic full scan catches targets the motion stage missed (e.g. hovering drones).
* Batching improves throughput over the batch-1 figures above. The T4 numbers are a reference point, not a Jetson prediction.

---

## 4. Minimum Viable System (Case 1)

**Detection range is set by the cueing camera, not the PTZ.** A PTZ can only look where it is told to, so it improves *identification* of tracks the cueing camera already found. It does not detect anything earlier.

| Component | Starting choice | Why |
| :--- | :--- | :--- |
| Cueing camera(s) | 4K, fixed, lower edge ~2° above the skyline (§2.4), covering the expected approach sector(s). HFOV from §6.1: **≤ ~25°** for scenario A (sized by the FPV). 30-60° only suffices if fast FPVs are out of scope | Mavic-class detection to ~115 m (60°), ~250 m (30°) or ~300 m (25°). Narrower lenses need more cameras per sector, and a wider upper row covers high, close drones |
| Confirmation camera | PTZ, ~5-10° at full zoom | Identification (drone vs bird, payload) of cued tracks. At 5-10° it resolves Mavic-class drones beyond 500 m, so identification is never the range bottleneck |
| Compute | NVIDIA Jetson Orin-class | TensorRT deployment path ([05 §4](05_ai_technical_architectures.md#4-edge-deployment-nvidia-jetson--tensorrt)) |
| Detector | Motion-cued YOLO26s crops + periodic full SAHI scan (§3.1) | [07 §4](07_deep_learning_architectures_cnn_transformer_mamba.md#4-recommended-detector-architecture) |
| Tracker / filter | BoT-SORT (camera-motion compensation for the PTZ) + IMM-EKF | [09](09_monocular_depth_frame_tracking_and_kinematics.md) |
| Decision | Rules floor + Jev advisory + human | [12](12_typesafe_jev_system_one_c2_threat_engine.md) |

**What this MVP can and cannot defend against, honestly:**
* **Can:** survey/recon quads at typical heights (tens to 100+ m) approaching through the covered sector, with several seconds to tens of seconds of warning.
* **Cannot:** fast, low FPVs approaching below the skyline or from uncovered sectors (under ~1-3 s of warning). Those need longer-range cueing: more and narrower cameras, horizon-level (Case 2) coverage, or a non-optical cue such as radar or acoustics ([01](01_research_papers_compendium.md)).

---

## 5. Legal & Operational Scope

* **Detection** with passive cameras is generally lawful to build and test on your own property. Privacy law still applies to recording people.
* **Mitigation** (RF jamming, GNSS spoofing, cyber takeover, kinetic defeat) is **restricted almost everywhere**. In the US it is limited to specific federal agencies under statutory authority, and jamming is prohibited for others by communications law. Most other countries have similar rules. Keep this project's effector side **simulated** unless you are working under an organisation with explicit legal authority.
* **Licences:** the repository is **private with no licence** until publication (decided 2026-09-23). AGPL obligations start only when the code is *distributed* or offered to others over a network, so private development is unaffected.
  * At publication, any released code that incorporates Ultralytics (YOLO26, AGPL-3.0) must be released under AGPL.
  * To keep parts closed after publishing, swap in an Apache-2.0 detector behind `cuas/detector.py`. Checked on GitHub 2026-09-23: RT-DETR, D-FINE, RF-DETR N/S/M/L (its XL/2XL models are under PML 1.0) and YOLOX. SAHI is MIT. YOLOv10 (AGPL) and YOLOv9 (GPL) don't avoid it.
  * Check the university's IP policy before any public disclosure of new methods.
  * Dataset licences apply per dataset. Halmstad is CC0.
* **Canada:** Transport Canada's operation categories are Basic (250 g-25 kg, pilot certificate, visual line of sight), Advanced, and Level 1 Complex (beyond visual line of sight). Sub-250 g drones need no certificate. The categories page shows **no blanket academic exemption**, so confirm what the project's flight authorisation actually covers (e.g. an SFOC). Staying in visual line of sight limits test ranges to what the pilot can see, so place a pilot or spotter out along the flight path for long-range trials.
* **Third-party cloud:** Jev sends track data off-site ([12 §4](12_typesafe_jev_system_one_c2_threat_engine.md#4-reference-implementation)).

*This is an engineering summary, not legal advice.*

---

## 6. Decisions

Decided 2026-09-22, extended 2026-09-23 (sensing, cameras, compute, licensing, jurisdiction). The site is not chosen yet, so the geometry is worked for three reference scenarios (§6.1). Rerun them for the real site once it is known.

| Decision | Why it matters | Current value |
| :--- | :--- | :--- |
| Intended use / licensing | AGPL, dataset licences, legal scope | **Academic research** with a supervising professor, **detection only**, in **Canada** at a remote site with approved testers. Effectors stay simulated. **Private, no licence until publication**; open source later (§5 covers the AGPL consequence) |
| Sensing | What the project can claim | **Vision only, no thermal.** The research question is how far a good visible camera plus software can go. **Night is in scope (user, 2026-10-22; was out of scope before):** a visible-light or low-light camera at night (lit drones as moving points of light; unlit drones only where there is ambient light). Thermal stays out |
| Primary capability | Where effort goes | **"Drone eyes":** detect drones and reject birds, aircraft and non-drone decoys ([14](14_build_log_and_experiments.md)) |
| Threat set (sizes, speeds, **flight heights**) | Sets pixels-on-target, warning time, and whether low FPVs are in scope | **All four tiers** as an analysis set: Mini-class 0.25 m / 15 m/s / 30 m, Mavic-class 0.35 m / 15 m/s / 100 m, FPV 0.30 m / 40 m/s / 30 m (and 5 m), Shahed-class 2.5 m / 50 m/s / 100 m. Working assumptions. Low FPVs below the skyline stay out of optical reach, and Shahed-class is analysis only (it can't be flight-tested at an academic site) |
| Required warning time | Sets detection range → lens | **5 s detection-only timeline**: confirm 0.3 s + PTZ identify ~2 s + operator alerted and looking ~3 s. No effector step. The requirement is "identified and the operator alerted before the drone reaches the zone edge". Real-time *tracking* (30 FPS, < 5 ms filter) is a separate, already-met requirement |
| Site and protected zone geometry | Sets required coverage and slant ranges | *Not chosen.* Reference scenarios A/B/C in §6.1; the MVP targets **A (test field)** |
| Approach sectors and skyline height | Sets how many cueing cameras and their tilt (§2.4) | Scenario values (90° / 180° / 360°; skyline 2°, mast 5 m) until a site is surveyed |
| Camera + lens(es) | Everything in §2 | **1-2 cameras** for cost; research needs one sector, not full coverage.<br>Candidate: **Raspberry Pi HQ camera** (IMX477, 4056×3040, 1.55 µm pixels, C/CS mount) with a 12-25 mm lens. A **16 mm lens gives 22° HFOV** and 10 px on a Mavic at ~360 m, Mini ~260 m. Full resolution runs at only ~10 fps; the 2028-wide mode reaches 40 fps but halves every range.<br>**Second camera: open.** A stereo pair gives metric range, e.g. ±0.4 m at 200 m with a 5 m baseline vs ±11 m monocular. Wide + narrow gives coverage instead.<br>(Scenario A's 4-camera row in §6.1 is the full-coverage deployment figure) |
| Measured $w_{\min}$ for your detector | Replaces the 10-px assumption | **First measurement (E1, 2026-09-24): 16 px** at recall ≥ 0.8, YOLO26s trained on Halmstad only. Optimistic, because the small sizes are simulated by downscaling. That cuts ranges to 62% of the 10-px figures (16 mm Pi HQ: Mavic ~226 m). **E2 (training on simulated distance): 8 px at conf 0.4, 4 px at conf 0.25** on the same optimistic test ([14](14_build_log_and_experiments.md), [15](15_detector_training_methods_and_findings.md)). **Working value: 8 px**, pending real footage (Mavic ~452 m with a 16 mm Pi HQ lens). **Caveat (E11, 2026-09-25): in-domain only.** On a different dataset (Anti-UAV300) the drone class fails at every size, so the range claim needs multi-dataset training (E12) and own-footage validation. **E12 (2026-09-25): 8 px on held-out Anti-UAV test videos** (real Full HD footage shrunk to ~7 px; recall 0.87 pooled) after training on Halmstad + Anti-UAV. The 8-px working value now holds on two datasets, but still on simulated distance and on datasets seen in training. Untested on an unseen dataset (E15) and on own footage. Precision there is only 0.78, from clutter false alarms |
| Compute platform | Latency budget | **Laptop RTX 4060 (8 GB)** for training and replay. **Google Colab T4** as a fallback trainer. **Raspberry Pi 5** as the capture node: capture, motion cueing and recording. Ultralytics measures YOLO26n at ~67 ms per 640 image on a Pi 5 (NCNN), too slow for full-frame detection. Jetson Orin optional later. **Field hardware, tiles-per-second demand vs capacity, and glass-to-track latency are analysed in [18](18_hardware_optimisation_and_realtime_latency.md)** (2026-09-28): the 0.3 s confirmation budget is met at 40 fps but not at full-resolution 10 fps |

### 6.1 Reference Scenarios

Computed with [`tools/geometry_budget.py`](tools/geometry_budget.py) on a 4K sensor with $w_{\min}$ = 10 px, skyline 2° and a 5 m mast. The model:
* **Needed distance** = speed × timeline + zone radius (horizontal). Slant range adds the height above the mast.
* **Max HFOV** is the widest lens that still puts 10 px on the target at that slant range.
* **The long-range row** uses the narrowest lens any threat needs (excluding threats below the skyline at that distance). It is tilted with its lower edge at the skyline.
* **Warning** is the time from the row's first detection to the zone edge. The row detects a target only where it has ≥ 10 px *and* sits inside the row's elevations.
* **In row** is how long the target stays inside the row before it climbs above it.

| Scenario | Zone radius | Sector | Timeline | Long-range row |
| :--- | ---: | ---: | ---: | :--- |
| **A: test field** (MVP) | 50 m | 90° | 5 s | 4 cameras at ≤ 25.8°, elevations 2.0°, 16.7° |
| **B: perimeter** | 150 m | 180° | 10 s | 16 cameras at ≤ 11.9°, elevations 2.0°, 8.7° |
| **C: 360° rooftop** | 100 m | 360° | 5 s | 17 cameras at ≤ 21.7°, elevations 2.0°, 14.3° |

Per threat (warning / time in row, in seconds, from that scenario's row):

| Threat | A | B | C |
| :--- | :--- | :--- | :--- |
| Mini-class, 30 m | 10.5 / 8.3 | 20.5 / 19.7 | 10.0 / 10.1 |
| Mavic-class, 100 m | **not seen by the row**: needs an upper row ≤ 46° | 32.4 / **1.1** | **not seen**: upper row ≤ 37° |
| FPV, 30 m | 5.0 / 4.2 (sized for it) | 10.0 / 9.7 | 5.0 / 5.0 |
| FPV, 5 m (at mast height) | out of optical scope | out of optical scope | out of optical scope |
| Shahed-class, 100 m | 40.8 / 35.5 | 51.4 / 42.0 (capped by the skyline) | 48.1 / 42.7 |

What this shows:
* **The fast FPV sets the lens.** In scenario A, one row of four ~25° cameras gives every above-skyline threat at least the 5 s it needs.
* **High, close drones need a second, wider row** (or the PTZ on a search pattern). A Mavic at 100 m is above a narrow row by the time it has enough pixels. In B it is visible for only ~1 s, too short to rely on.
* **360° coverage or longer timelines multiply the camera count** (B and C need 16-17 cameras). That is the practical case for adding a non-optical cue (acoustic or radar) rather than more cameras.
* **Far targets are capped by the skyline, not by pixels.** A Shahed at 100 m is below a 2° skyline beyond ~2.7 km, however many pixels it would have.

Rerun for your own site, or a what-if:
```bash
python tools/geometry_budget.py --scenario A
python tools/geometry_budget.py --scenario B --timeline_s 5 --min_px 8
python tools/geometry_budget.py --site my_site.json   # keys as in SCENARIOS; optional "threats" list
```
