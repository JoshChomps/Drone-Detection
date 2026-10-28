# Open-Source Repositories & Toolkits for Drone Defense

**29 repositories** (plus one GitHub topic index) covering vision, RF, radar, acoustic and simulation work relevant to counter-UAS.

> [!NOTE]
> **Verification status (2026-09-21).** Every link below was confirmed to exist via the GitHub API on that date. Star counts and "last push" are snapshots. Descriptions are based on each repo's own README or description. Claims we could not check are marked *(unverified)*. Two links from the previous version were dead (`m-al-sad/DroneRF`, `helsing-ai/edth-copenhagen-drone-acoustics`) and have been replaced.
>
> **Maturity legend:** 🟢 maintained, widely used · 🟡 research code (paper companion, unmaintained or small) · ⚪ hobby / hackathon scale · ⛔ archived.

---

## 📑 Categories

1. [Computer Vision & Tracking](#1-computer-vision--tracking)
2. [Radio Frequency (RF)](#2-radio-frequency-rf)
3. [Radar & Micro-Doppler](#3-radar--micro-doppler)
4. [Acoustic](#4-acoustic)
5. [Simulation & Autonomy](#5-simulation--autonomy)

---

## 1. Computer Vision & Tracking

| Repository | Maturity | What it is | Pairs with |
| :--- | :--- | :--- | :--- |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | 🟢 ~62k★ | YOLO26 training, export (ONNX/TensorRT) and tracking (ByteTrack, BoT-SORT built in). **AGPL-3.0.** | [01 §8.1](01_research_papers_compendium.md#81-ultralytics-yolo26-unified-real-time-end-to-end-vision-models) |
| [obss/sahi](https://github.com/obss/sahi) | 🟢 ~5.5k★ | Sliced/tiled inference for small objects in large frames; framework-agnostic | [05 §2.1](05_ai_technical_architectures.md#21-optical--thermal-detectors) |
| [ZhaoJ9014/Anti-UAV](https://github.com/ZhaoJ9014/Anti-UAV) | 🟢 ~850★ | Official Anti-UAV benchmark repo: dataset links, evaluation code (own IoU + visibility accuracy), Anti-UAV410 Jittor tooling | [01 §2.1](01_research_papers_compendium.md#21-anti-uav-a-large-scale-benchmark-for-vision-based-uav-tracking) |
| [HwangBo94/Anti-UAV410](https://github.com/HwangBo94/Anti-UAV410) | 🟡 ~190★ | Anti-UAV410 thermal benchmark and SiamDT tracker | [01 §2.2](01_research_papers_compendium.md#22-anti-uav410-a-thermal-infrared-benchmark-and-customized-scheme-for-tracking-drones-in-the-wild) |
| [Jake-WU/Det-Fly](https://github.com/Jake-WU/Det-Fly) | 🟡 ~100★ | Det-Fly air-to-air dataset (13k+ images). Dataset release, little code | [01 §2.4](01_research_papers_compendium.md#24-air-to-air-visual-detection-of-micro-uavs-an-experimental-evaluation-of-deep-learning-det-fly) |
| [KostadinovShalon/UAVDetectionTrackingBenchmark](https://github.com/KostadinovShalon/UAVDetectionTrackingBenchmark) | 🟡 ~140★ | Code for Isaac-Medina et al., *UAV Visual Detection and Tracking using Deep Neural Networks: A Performance Benchmark* ([arXiv:2103.13933](https://arxiv.org/abs/2103.13933)); common evaluation across MAV-VID, Drone-vs-Bird and Anti-UAV | - |
| [wosdetc/challenge](https://github.com/wosdetc/challenge) | 🟡 ~70★ | Official Drone-vs-Bird challenge annotations and instructions ([2025 site](https://wosdetc2025.wordpress.com/)) | [03](03_datasets_and_benchmarks_catalog.md) |
| [yjwong1999/IJCNN2025-DvB](https://github.com/yjwong1999/IJCNN2025-DvB) | ⚪ ~15★ | WRN-YOLO: YOLO with a Wide ResNet backbone (via `timm`) for the 8th Drone-vs-Bird challenge. *Not* an RT-DETR/GAN system as previously listed | [01 §2.7](01_research_papers_compendium.md#27-wrn-yolo-an-improved-yolo-for-drone-detection-using-wide-resnet) |
| [MagriniGabriele/NeRDD](https://github.com/MagriniGabriele/NeRDD) | 🟡 ~30★ | NeRDD event+RGB dataset (3.5 h, 115 videos) | [01 §8.4](01_research_papers_compendium.md#84-neuromorphic-drone-detection-an-event-rgb-multimodal-approach-nerdd) |
| [miccunifi/FRED](https://github.com/miccunifi/FRED) | 🟡 ~100★ | "Florence RGB-Event Drone dataset", the NeRDD authors' newer multimodal set *(contents unverified)* | - |

---

## 2. Radio Frequency (RF)

| Repository | Maturity | What it is |
| :--- | :--- | :--- |
| [kitoweeknd/RFUAV](https://github.com/kitoweeknd/RFUAV) | 🟢 ~440★ | Official RFUAV code: preprocessing, SNR tools, baselines. Data is on [Hugging Face](https://huggingface.co/datasets/kitofrank/RFUAV) as per-device `.rar` archives (see [03](03_datasets_and_benchmarks_catalog.md#rfuav)) |
| [Al-Sad/DroneRF](https://github.com/Al-Sad/DroneRF) | 🟡 ~190★ | MATLAB and Python code for the DroneRF paper (2019). Replaces the dead `m-al-sad/DroneRF` link |
| [tesorrells/RF-Drone-Detection](https://github.com/tesorrells/RF-Drone-Detection) | 🟡 ~225★ | Low-cost SDR drone detection (Python) *(GNU Radio / HackRF details unverified)* |
| [IQTLabs/RFClassification](https://github.com/IQTLabs/RFClassification) | ⛔ archived 2022 | RF device detection/classification experiments. Useful reference, not "production" |
| [r4d10n/rfml-moe-hub](https://github.com/r4d10n/rfml-moe-hub) | ⚪ 3★ | Personal "Drone RFML research hub". Not a benchmark; claims in its README are the author's own |

---

## 3. Radar & Micro-Doppler

| Repository | Maturity | What it is |
| :--- | :--- | :--- |
| [ZHOUYI1023/awesome-radar-perception](https://github.com/ZHOUYI1023/awesome-radar-perception) | 🟢 ~1.9k★ | Curated list of radar datasets, detection, tracking and fusion work. The best radar entry point |
| [diptiman-mohanta/Radar-Based-UAV-Classification](https://github.com/diptiman-mohanta/Radar-Based-UAV-Classification) | ⚪ ~20★ | MATLAB radar UAV classification. A **third-party** project, not the DIAT-µSAT authors' code |
| [Keerthi2134/doppler-radar](https://github.com/Keerthi2134/doppler-radar) | ⚪ ~20★ | Micro-Doppler target recognition, Smart India Hackathon (SIH1606) project |
| [UTSAVS26/Micro-Doppler-Based-Target-Classification](https://github.com/UTSAVS26/Micro-Doppler-Based-Target-Classification) | ⚪ ~10★ | Micro-Doppler classification demo. GitHub reports JavaScript as the main language *(ML stack unverified)* |

---

## 4. Acoustic

| Repository | Maturity | What it is |
| :--- | :--- | :--- |
| [saraalemadi/DroneAudioDataset](https://github.com/saraalemadi/DroneAudioDataset) | 🟡 ~120★ | Audio dataset behind Al-Emadi et al. ([01 §5.1](01_research_papers_compendium.md#51-audio-based-drone-detection-and-identification-using-deep-learning-techniques-with-dataset-enhancement-through-gans)) |
| [agamrossen/VolAnti](https://github.com/agamrossen/VolAnti) | 🟢 ~380★, active | Open-source acoustic drone detector, mainly written in **C** ("hears the propellers, not the radio"). There is no associated IEEE paper: the one previously cited does not exist |
| [kbhujbal/SudarshanChakra-acoustic_uav_threat_detection_CNN](https://github.com/kbhujbal/SudarshanChakra-acoustic_uav_threat_detection_CNN) | ⚪ ~50★ | Mel-spectrogram CNN detector ("production" is the repo's self-description) |
| [JoulesSpace/acoustic-drone-detection](https://github.com/JoulesSpace/acoustic-drone-detection) | ⚪ 7★ | Rust DSP + ML toolbox for acoustic detection *(embedded / power claims unverified)* |
| [AntoineNaccache/Drone_propellers_recognition](https://github.com/AntoineNaccache/Drone_propellers_recognition) | ⚪ 1★ | Lightweight CRNN for real-time drone audio classification |
| [norle/edth-copenhagen-drone-acoustics](https://github.com/norle/edth-copenhagen-drone-acoustics) | ⚪ fork | Participant copy of the EDTH Copenhagen 2025 drone-acoustics challenge (designed by Helsing). The original `helsing-ai/...` repo is no longer public |

---

## 5. Simulation & Autonomy

| Repository | Maturity | What it is |
| :--- | :--- | :--- |
| [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | 🟢 ~8k★ | NVIDIA's current GPU robot-learning framework, the successor to IsaacGymEnvs. Use this for RL pursuit/interception experiments |
| [NVIDIA-Omniverse/IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) | ⛔ archived | Legacy Isaac Gym environments. Kept only because older papers use it |
| [PX4/PX4-Autopilot](https://github.com/PX4/PX4-Autopilot) | 🟢 ~12.7k★ | Open autopilot for simulated/real interceptor or target drones (SITL with Gazebo) |
| [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) | 🟢 ~5.7k★ | CUDA Gaussian-splatting rasterizer (PyTorch). The practical choice if you build a site digital twin ([11 §4](11_research_deliberations_and_user_pushback_log.md#4-3d-gaussian-splatting-3dgs-where-it-can-and-cannot-help)) |
| [GitHub topic: drone-interception](https://github.com/topics/drone-interception) | - | Community index of pursuit/interception repos. Quality varies; vet each one individually |
