# Research Deliberations, Recommendations & User Pushback Log

A record of the design questions raised, where the user pushed back, and how each was resolved. The log is kept so later decisions can see *why* earlier ones were made, including the ones that turned out wrong.

> [!NOTE]
> **Revised 2026-09-21.** Facts in this file were re-verified. Corrections: YOLO26 latency figures are T4 (not Jetson); `yolo26*-depth` models are trained on indoor data; "MambaSOD" is salient-object detection; "AeroGS" and "GauMamba" could not be found and were removed; Gamba's authors are corrected. Two new log entries were added (rows 5-6).

---

## 1. Pushback & Resolution Log

| # | Date | Topic | Initial position | User pushback | Resolution |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | 2026-09-19 | **YOLO26** | Treated "YOLO26" as speculative or a layer index | Pointed to the [official docs](https://docs.ultralytics.com/models/yolo26/) | **User correct.** YOLO26 is the current Ultralytics family ([arXiv:2606.03748](https://arxiv.org/abs/2606.03748)). [07](07_deep_learning_architectures_cnn_transformer_mamba.md) was only fixed on 2026-09-21 |
| 2 | 2026-09-19 | **Frameworks** | PyTorch as the only option | Asked for alternatives | Alternatives evaluated (§3). PyTorch stays primary because the whole detector/export stack depends on it |
| 3 | 2026-09-19 | **Gaussian splatting** | 2D boxes + pinhole priors only | Asked about integrating 3DGS | Originally answered "yes, validated". **Revised:** feasible for site digital twins and synthetic data, *not* for tracking few-pixel drones (§4) |
| 4 | 2026-09-19 | **SSM / Mamba** | Suggested SSMs as a frontier | Asked for literature verification | Originally cited MambaSOD as small-object evidence. **Revised:** MambaSOD is RGB-D *salient*-object detection. Real SSM trackers exist (TrackingMiM), but there is no verified drone-detection result (§5) |
| 5 | 2026-09-21 | **Citation integrity** | Compendium presented as 45+ verified papers | Asked for an expert review | Audit found most arXiv IDs/DOIs wrong and 10 papers nonexistent. All reference files were rebuilt from verified records ([01 Appendix A](01_research_papers_compendium.md#appendix-a-what-changed-in-the-2026-09-21-audit)) |
| 6 | 2026-09-21 | **Jev's role** | Review said Jev should not sit in the engagement loop | "Doesn't Jev optimize and make it stronger?" | **Both partly right.** Jev can strengthen the system as a calibrated *advisory* and triage layer. It should not be the component that authorises effectors. [12](12_typesafe_jev_system_one_c2_threat_engine.md) was restructured accordingly |

| 7 | 2026-09-28 | **Masks vs boxes** | Boxes only | "Should we mask over the drones, or is that too hard, unnecessary, jargon?" | **No segmentation network.** At 8-30 px a mask barely differs from its box; no dataset used has mask labels; a segmentation head adds compute per tile ([18](18_hardware_optimisation_and_realtime_latency.md)). The useful part of a mask (silhouette area and brightness over time, the wing-flap signal) comes almost free from Otsu thresholding inside the box against sky (`cuas.track_features.silhouette`). Revisit for close-range identification by a zoom camera |
| 8 | 2026-09-30 | **YOLOBirDrone's YOLO changes** ([01 §2.8](01_research_papers_compendium.md#28-yolobirdrone-dataset-for-bird-vs-drone-detection-and-classification-and-a-yolo-based-enhanced-learning-architecture)) | Keep stock YOLO26s | "Is it superior to what we have, or can we incorporate things?" | **Not superior on its evidence; nothing adopted now.** The full model ties plain YOLOv8 (mAP50-95 0.668 vs 0.661, single runs). Each module alone is at or below baseline. There are no small-target results and no code. Our measured bottleneck is data, not architecture: E11 → E12 moved drone recall 0.05 → 0.87 with the architecture unchanged, and E12's main error is clutter. **Deformable convolutions: no.** They need custom operators on TensorRT and may not exist on NPUs like Hailo (unverified), and slow CPU inference on the Pi. **Lightweight attention (ECA-style):** a possible later ablation, only after E14/E15, with ≥ 2 seeds and per-video intervals. **P2 head (E7):** still the first architecture change if real small targets are missed. **Follow-up the same day:** the user asked to implement all three anyway, so they were built as **E18** (`cuas/arch.py`, 14 §4). Each starts as the identity (no pretraining lost); tested and costed (full set 1.6-1.8× compute). Trained on Colab after E14, with a second-seed baseline so a gain can be told from noise |
| 9 | 2026-10-01 | **Gazebo / ROS simulation and synthetic footage** | Real data (Anti-UAV, DUT, Drone-vs-Bird) for the detector; replay of real video for the pipeline | "How would simulating in Gazebo and ROS, or creating footage in those, impact the project?" | **Not for detector training; possibly later for pipeline testing.** Our measured failures are domain shift and clutter (E11, E15) and unlabelled real drones. Gazebo's renderer is not photoreal at 8-20 px, where sensor noise, blur, compression and haze decide detection, so it would mostly teach the detector Gazebo. Simulated birds would be hand-scripted, so an M2 classifier trained or validated on them would learn the script (the same confound risk as 17 §6). **Useful for:** ground-truth trajectories to test tracking, IMM, TTC and range error, and end-to-end latency, though much of that is cheaper as synthetic trajectories in Python. A site digital twin (§4) is the better synthetic-image route. **ROS 2** is an integration choice for the field node, not research; revisit at field build. The sibling `../Drone_to_Drone` needs PX4 + Gazebo anyway (`gz-sim` active; Gazebo Classic archived, GitHub API 2026-10-01) |
| 10 | 2026-10-01 | **PX4 + Gazebo for testing and noisy data; "consider the novel"** | Row 9: not for detector training | "I was also thinking of PX4 and Gazebo for testing and creating noisy data to put it against"; asked for novel research to be taken seriously | **Row 9 stands, made usable.** PX4 SITL is worth having for its *flight dynamics* (hover jitter, turns, wind), not for Gazebo's rendered pixels: `cuas/sim.py` projects paths, including PX4 logs (`from_px4_csv`), through our camera model and draws targets at true pixel size with blur, shake and sensor noise. Within a day it found three real defects (a flapping bird read as 0.2 s to impact; hoverers never confirmed; TTC biased long at a few pixels). **Noisy data for the detector comes from degrading real frames** (E21: 8-px recall 0.93 → 0.08), which is the measured gap. **Novel research, checked at arXiv (abstracts only):** temporal input (YOLOMG, dual-interval cues), a foundation backbone (DINOv3 / DEIMv2), NWD, SPD-Conv and a learned track classifier are ranked in [19 §5](19_design_review_parameters_and_novel_directions.md#5-novel-directions-ranked-against-this-projects-measured-failures) by whether their mechanism matches a failure measured here. The 2025 Drone-vs-Bird winner used scale, augmentation and frame-to-frame consistency, not a new architecture |
| 11 | 2026-10-02 | **Gazebo / PX4 again: "can they simulate footage/environments, can it be used to test or improve our process?"; varied terrain data; seed dependency** | Rows 9-10: PX4 for dynamics, not Gazebo for pixels | The user asked a third time, now with the E13/E18 results in hand, and asked for varied terrain and less seed dependency | **Rows 9-10 stand, with one addition.** Simulation is valuable for *testing* (range and velocity truth for the tracker, time to collision, latency) and not for *training pixels* (published rendered-data gains use Unreal Engine and are supplements). The addition is E32: PX4 flight paths driving **real drone crops pasted on real target-free background video**, which gives range truth and varied terrain with real appearance ([19 §6](19_design_review_parameters_and_novel_directions.md#6-simulation-px4-gazebo-and-cuassim)). Varied terrain: own target-free footage first (it also gives the false-alarm rate per hour), then pasted crops, then the DUT train split ([19 §3.9](19_design_review_parameters_and_novel_directions.md#39-varied-terrain-and-drone-data)). Seed dependency: per-model thresholds, fixed schedule, and averaging the seeds' weights into one model, E31 ([19 §3.8](19_design_review_parameters_and_novel_directions.md#38-reducing-the-seed-dependency)) |
| 12 | 2026-10-05 | **Night tracking of the user's own drones with no or limited lights** (drone-club exercises, mountain areas; future work) | Night is out of scope (13 §6): a visible camera sees a drone at night only if it is lit. Canadian rules require lights at night (CAR 901.39, read 2026-10-05) and white anti-collision lights flashing 40-100 per minute beyond visual line of sight (901.38.1). Current models on Anti-UAV's 18 night test videos: AP 0.89-0.93 but recall at 10 false drones per 1000 frames ~0.6 (day ~0.9): other lights are the false drones | "Track our drones at night with no or limited lighting, by observing movements in the sky and the differences of those positions (abnormal colour shifts), or a simulated heat map from colour pixels." Not necessarily Canada; daytime first | **Deferred until daytime is done (user).** Assessment for then: (1) **Motion between frames is the right core**, and is what `cuas.motion` already does; it needs the drone to differ from the sky at all: a silhouette against twilight, moonlight or skyglow, or stars it covers. A mountain sky is darker, so there is less silhouette. (2) **Colour carries little at night:** the light is roughly 10^5 (full moon) to 10^8 (starlight) times dimmer than day, so colour shifts are mostly sensor noise. (3) **A heat map computed from colour pixels adds no thermal information:** silicon cameras (including NoIR) do not see the long-wave infrared that warm motors and batteries emit; any such map is a recolouring of brightness. Its valid version is a **temporal accumulation map** (integrating faint changes along candidate paths, track-before-detect), which gains signal over many frames. (4) Ranked options: low-light sensor + fast lens + motion cueing + temporal integration; near-IR illumination (short range); a **thermal camera**, the real answer for unlit drones (outside this project's vision-only scope; the pipeline carries over to a one-channel image). (5) The club drones' own flight logs give position truth, which is also what the daytime work lacks (E32, TTC, real `w_min`). Local night-flight rules still apply wherever they fly |

### Process lesson
Rows 1 and 3-4 show two opposite failure modes: **dismissing** something new from stale memory (YOLO26), then **over-accepting** claims after pushback (3DGS, Mamba) without checking feasibility. The fix is the same both times: check the primary source, then state the limits for *this* geometry (small, distant targets).

---

## 2. Ultralytics YOLO26: Verified Facts & C-UAS Relevance

Checked against the [docs page](https://docs.ultralytics.com/models/yolo26/) and [arXiv:2606.03748](https://arxiv.org/abs/2606.03748):

| Feature | What it is | Why it matters here |
| :--- | :--- | :--- |
| **STAL** (Small-Target-Aware Label Assignment) | Guarantees positive label assignments for small objects, which earlier assigners could leave with none | Directly targets the few-pixel drone failure mode |
| **NMS-free head** (`nms=False`) | Optional one-to-one head. **The default path still uses NMS** for slightly higher accuracy (e.g. YOLO26n 40.9 vs 40.1 mAP end-to-end) | Simpler export and more predictable latency; measure the accuracy cost on small targets |
| **DFL removed** | Lighter head, unconstrained box regression | Simpler, faster export |
| **MuSGD** | Hybrid Muon + SGD optimizer adapted from LLM training | Faster convergence on custom data *(benefit on drone data unmeasured)* |
| **Speed** | 40.9-57.5 COCO mAP at 1.7-11.8 ms on **NVIDIA T4 + TensorRT 10** | Jetson numbers must be measured on the device |
| **`yolo26*-depth`** | Monocular depth heads (added 2026-08-21), trained on **NYU Depth V2 (indoor)** | **Not** a ranging tool for drones against sky ([09 §1.2](09_monocular_depth_frame_tracking_and_kinematics.md#12-method-b-monocular-metric-depth-foundation-models)) |
| **Licence** | **AGPL-3.0** | Closed or commercial deployment needs an Ultralytics Enterprise licence |
| **YOLO27** | Announced preview; no weights or date | Re-evaluate when released |

---

## 3. Alternatives to PyTorch

| Framework | Paradigm | Strengths | Limitations | Verdict here |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch 2.x** | Eager + `torch.compile` | Ultralytics, SAHI, trackers, gsplat and the ONNX→TensorRT path are all native | Heavier than needed on microcontrollers | **Primary** |
| **JAX (+ Flax / Equinox)** | Functional transforms (`jit`, `vmap`, `grad`) on XLA | Excellent for differentiable physics and simulation | Dynamic shapes (variable box counts, tiling) are awkward; no YOLO ecosystem. The mainstream 3DGS tooling (e.g. [gsplat](https://github.com/nerfstudio-project/gsplat)) is PyTorch/CUDA, not JAX | Only for simulation/physics side projects |
| **Mojo / MAX (Modular)** | Python-superset language + inference engine | Promising inference performance | Young ecosystem; no drone-detection models *(speed-up claims unverified)* | Watch |
| **Keras 3** | Multi-backend API | Write once, run on JAX/PyTorch/TF | Little control over custom detection heads | Teaching only |
| **Burn** (Rust) | Native Rust DL with multiple backends | Memory safety, no GC, embeddable | Small model zoo; vision necks must be hand-written | Possible for a Rust-only edge binary later |
| **Candle** (Rust, Hugging Face) | Minimal Rust inference | Small binaries | Inference-focused | Possible for small edge models |
| **tinygrad** | Minimal tensor library with its own compiler backends | Transparent, hackable | Small ecosystem, few pretrained vision weights | Educational |

---

## 4. 3D Gaussian Splatting (3DGS): Where It Can and Cannot Help

**Hard constraint:** 3DGS pose tracking fits a 3D model to image evidence. At 165 m a 0.35 m drone is **4-15 pixels wide** ([09 §1.1](09_monocular_depth_frame_tracking_and_kinematics.md#11-method-a-geometric-bounding-box-scale-prior-pinhole-formulation)). There is not enough signal to recover orientation, so the verified pose-tracking papers all work at close range.

| Paradigm | Verified papers | Feasible here? |
| :--- | :--- | :--- |
| **6-DoF pose tracking of the target** | GSGTrack ([arXiv:2412.02267](https://arxiv.org/abs/2412.02267), robotic manipulation), POGS ([arXiv:2503.05189](https://arxiv.org/abs/2503.05189), hand/robot manipulation), 6DGS ([arXiv:2407.15484](https://arxiv.org/abs/2407.15484), 6D pose from one image given a pre-built 3DGS model), PhysPose ([arXiv:2503.23587](https://arxiv.org/abs/2503.23587), pose refinement with physics) | **No** for Case 1. Maybe in an interceptor's final seconds (Case 3), when the target fills hundreds of pixels |
| **Site digital twin → synthetic training data** | DroneSplat ([arXiv:2503.16964](https://arxiv.org/abs/2503.16964), reconstruction from drone imagery), UAVTwin ([arXiv:2504.02158](https://arxiv.org/abs/2504.02158), neural digital twins for UAVs) | **Yes, the best fit.** Reconstruct your own site, then render drones and birds into it under synthetic fog and lighting ([10 §3](10_adverse_weather_datasets_and_benchmarks_2026.md#3-physical-models-of-weather-degradation)). Tooling: gsplat (PyTorch) |
| **Single-image 3D asset generation** | Gamba ([arXiv:2403.18795](https://arxiv.org/abs/2403.18795)), MVGamba ([arXiv:2406.06367](https://arxiv.org/abs/2406.06367)) | Only as a way to create drone/bird 3D assets for synthetic data; not for ranging |

*Removed:* "AeroGS (CVPR 2024)", which could not be found.

---

## 5. State-Space Models (Mamba) + Vision: What the Literature Supports

* **Gamba** (Shen, Wu, Yi, Zhou, Zhang, Yan, Wang; [arXiv:2403.18795](https://arxiv.org/abs/2403.18795)): a Mamba backbone predicts 3D Gaussians from one object-centric image with linear scaling in token count. Real, but it is a 3D-asset generator.
* **MVGamba** ([arXiv:2406.06367](https://arxiv.org/abs/2406.06367)): multi-view 3D generation as state-space sequence modelling.
* **TrackingMiM** ([arXiv:2507.01535](https://arxiv.org/abs/2507.01535)): real-time Mamba tracker for video *from* UAV platforms. The best evidence that SSM trackers can run in real time.
* **MambaSOD** ([arXiv:2410.15015](https://arxiv.org/abs/2410.15015)): **RGB-D salient object detection.** Not evidence for small-drone detection.
* *Removed:* "GauMamba (2025)" and "TrackingMamba", which could not be found under those names.

**Bottom line:** no verified paper shows an SSM backbone beating a tuned CNN for *detecting small drones* on an edge device. It remains a research ablation ([07 §4](07_deep_learning_architectures_cnn_transformer_mamba.md#4-recommended-detector-architecture)).

---

## 6. Current Project Recommendations

1. **Detector:** YOLO26s with STAL, run on motion-cued crops plus a periodic full SAHI scan ([13 §3.1](13_system_requirements_and_geometry_budget.md#31-compute-what-full-resolution-detection-costs)). Try `nms=False` and measure its small-target cost. Check the AGPL licence against your intended use.
2. **Framework:** PyTorch 2.x, WSL2 on the Windows workstation, TensorRT on the Jetson.
3. **Geometry:** Case 1 (ground-to-air) with the long-range camera tilted low ([13 §2.4](13_system_requirements_and_geometry_budget.md#24-elevation-where-distant-drones-actually-appear)), and the lens chosen from the pixels-on-target budget in [13](13_system_requirements_and_geometry_budget.md). Low, fast FPVs are out of the MVP's reach.
4. **Data:** synthetic pre-training (SimD3, DroneGround if granted) → real fine-tuning (Anti-UAV RGB, Drone-vs-Bird, UETT4K) → your own held-out site footage ([03 §2](03_datasets_and_benchmarks_catalog.md#2-what-to-train-on-for-case-1-recommended-order)).
5. **Kinematics:** IMM-EKF on bearing + box size, with TTC as the primary threat cue; treat range as coarse ([09 §3.4](09_monocular_depth_frame_tracking_and_kinematics.md#34-what-is-actually-observable-from-one-camera)).
6. **Decision layer:** deterministic rules gate every action; Jev advises and ranks ([12](12_typesafe_jev_system_one_c2_threat_engine.md)).
7. **3DGS:** only as a site digital twin for synthetic data, after the baseline works.
