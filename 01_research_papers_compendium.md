# Research Papers Compendium: AI in Drone Detection & Defense (2019-2026)

A curated catalog of **32 verified papers** on detecting, tracking and countering drones, grouped by sensing modality.

> [!IMPORTANT]
> **Verification status (2026-09-21).** Every entry below was checked against arXiv, Crossref and/or Semantic Scholar: title, authors, venue and identifier all resolve to the paper described. "Key findings" are paraphrased from each paper's own abstract, not from secondary sources. Entries marked *(abstract not reviewed)* have confirmed metadata only.
>
> An earlier version of this file (2026-09-18) contained 32 entries whose identifiers were largely wrong: 11 of 16 arXiv IDs and all 19 DOIs pointed to unrelated papers or to nothing, and 10 papers could not be found at all. See [Appendix A](#appendix-a-what-changed-in-the-2026-09-21-audit). **Do not re-import citations from older copies of this file.**

Machine-readable versions: [papers_database.json](papers_database.json) is the source of truth, and [06_bibliography.bib](06_bibliography.bib) is generated from it (`python tools/build_bibliography.py`), so both share the same keys. `python tools/check_docs.py` fails if any verified paper is missing from this file.

---

## 📑 Table of Contents

1. [Surveys & C-UAS Taxonomies](#1-surveys--c-uas-taxonomies)
2. [Computer Vision: Electro-Optical (EO) & Thermal Infrared (IR)](#2-computer-vision-electro-optical-eo--thermal-infrared-ir)
3. [Radio Frequency (RF)](#3-radio-frequency-rf)
4. [Radar & Micro-Doppler](#4-radar--micro-doppler)
5. [Acoustic](#5-acoustic)
6. [Multi-Modal Sensor Fusion & Uncertainty](#6-multi-modal-sensor-fusion--uncertainty)
7. [Electronic Warfare & Navigation Security](#7-electronic-warfare--navigation-security)
8. [Frontier Detectors & Synthetic Data (2024-2026)](#8-frontier-detectors--synthetic-data-2024-2026)
9. [Transferable Techniques (Not Drone-Detection Papers)](#9-transferable-techniques-not-drone-detection-papers)
10. [Known Gaps](#10-known-gaps)

**"Project relevance"** lines judge each paper against this project's current design (fixed ground camera looking up: see [08](08_camera_angles_and_operational_pov_selection.md) and [13](13_system_requirements_and_geometry_budget.md)).

---

## 1. Surveys & C-UAS Taxonomies

### 1.1 Securing the Skies: A Comprehensive Survey on Anti-UAV Methods, Benchmarking, and Future Directions
* **Authors**: Yifei Dong, Fengyi Wu, Sanjian Zhang, … Zhi-Qi Cheng (11 authors)
* **Venue**: CVPR 2025 Workshops, pp. 6661-6675. **Best Paper Award**, CVPR 2025 Anti-UAV Workshop
* **Links**: [arXiv:2504.11967](https://arxiv.org/abs/2504.11967) · [DOI:10.1109/CVPRW67362.2025.00663](https://doi.org/10.1109/CVPRW67362.2025.00663)
* **Key findings**: Covers classification, detection and tracking across RGB, IR, audio, radar and RF, including diffusion-based data synthesis, multi-modal fusion, vision-language models, self-supervised learning and RL. Identifies gaps in real-time performance, stealth-target detection and swarms.
* **Project relevance**: The best single starting point for the literature.

### 1.2 Deep learning for unmanned aerial vehicles detection: A review
* **Authors**: Nader Al-lQubaydhi, Abdulrahman Alenezi, Turki Alanazi, … Salim Hariri (9 authors)
* **Venue**: *Computer Science Review*, vol. 51, 100614 (2024)
* **Links**: [DOI:10.1016/j.cosrev.2023.100614](https://doi.org/10.1016/j.cosrev.2023.100614)
* **Key findings**: Review of deep-learning UAV detection *(abstract not reviewed)*.

### 1.3 A Survey on Detection, Classification, and Tracking of UAVs using Radar and Communications Systems
* **Authors**: Wahab Ali Gulzar Khawaja, Martins Ezuma, Vasilii Semkin, Fatih Erden, Ismail Guvenc
* **Venue**: TechRxiv preprint, 2025 (not peer reviewed)
* **Links**: [DOI:10.36227/techrxiv.174181637.73215309/v1](https://doi.org/10.36227/techrxiv.174181637.73215309/v1)
* **Key findings**: Radar- and communications-system-based UAV sensing *(abstract not reviewed)*.

### 1.4 Counter-Unmanned Aircraft System(s) (C-UAS): State of the Art, Challenges, and Future Trends
* **Authors**: Jian Wang, Yongxin Liu, Houbing Song
* **Venue**: *IEEE Aerospace and Electronic Systems Magazine*, 36(3):4-29 (2021)
* **Links**: [DOI:10.1109/MAES.2020.3015537](https://doi.org/10.1109/MAES.2020.3015537) · [arXiv:2008.12461](https://arxiv.org/abs/2008.12461)
* **Key findings**: Defines C-UAS as systems that *lawfully and safely* disable, disrupt or seize control of UAS. Reviews detection, tracking, identification and mitigation.
* **Project relevance**: The "lawfully" framing matters. See the legal scope note in [13](13_system_requirements_and_geometry_budget.md#5-legal--operational-scope).

### 1.5 Defending Airports from UAS: A Survey on Cyber-Attacks and Counter-Drone Sensing Technologies
* **Authors**: Georgia Lykou, Dimitrios Moustakas, Dimitris Gritzalis
* **Venue**: *Sensors*, 20(12):3537 (2020)
* **Links**: [DOI:10.3390/s20123537](https://doi.org/10.3390/s20123537)
* **Key findings**: Surveys drone incidents near airports and the sensors used to prevent, detect, identify and mitigate them, from a critical-infrastructure perspective.

---

## 2. Computer Vision: Electro-Optical (EO) & Thermal Infrared (IR)

### 2.1 Anti-UAV: A Large-Scale Benchmark for Vision-Based UAV Tracking
* **Authors**: Nan Jiang, Kuiran Wang, Xiaoke Peng, … Zhenjun Han (12 authors)
* **Venue**: *IEEE Transactions on Multimedia*, 25:486-500 (2023)
* **Links**: [DOI:10.1109/TMM.2021.3128047](https://doi.org/10.1109/TMM.2021.3128047) · [arXiv:2101.08466](https://arxiv.org/abs/2101.08466) (preprint title: *A Large Multi-Modal Benchmark for UAV Tracking*) · [Code](https://github.com/ZhaoJ9014/Anti-UAV)
* **Key findings**: 300+ paired RGB/thermal videos with 580k+ manually annotated boxes, plus a dual-flow semantic consistency (DFSC) tracker.
* **Project relevance**: The standard RGB+IR tracking benchmark. Its metric is its own IoU-with-visibility score, not HOTA (see [10 §4](10_adverse_weather_datasets_and_benchmarks_2026.md#4-evaluation-protocols--metrics)).

### 2.2 Anti-UAV410: A Thermal Infrared Benchmark and Customized Scheme for Tracking Drones in the Wild
* **Authors**: Bo Huang, Jianan Li, Junjie Chen, Gang Wang, Jian Zhao, Tingfa Xu
* **Venue**: *IEEE TPAMI*, 46(5):2852-2865 (2024)
* **Links**: [DOI:10.1109/TPAMI.2023.3335338](https://doi.org/10.1109/TPAMI.2023.3335338) · [Code](https://github.com/HwangBo94/Anti-UAV410)
* **Key findings**: **Thermal-only** benchmark: 410 videos, 438k+ boxes, plus the SiamDT tracker for small UAVs in dynamic clutter.

### 2.3 CST Anti-UAV: A Thermal Infrared Benchmark for Tiny UAV Tracking in Complex Scenes
* **Authors**: Bin Xie, Congxuan Zhang, Fagan Wang, Peng Liu, Feng Lu, Zhen Chen, Weiming Hu
* **Venue**: ICCV 2025 Workshops, pp. 6216-6225
* **Links**: [arXiv:2507.23473](https://arxiv.org/abs/2507.23473) · [DOI:10.1109/ICCVW69036.2025.00647](https://doi.org/10.1109/ICCVW69036.2025.00647)
* **Key findings**: 220 thermal sequences with 240k+ boxes and per-frame attribute labels, dominated by **tiny** targets. The best of 20 trackers reaches only **35.92%** state accuracy, versus 67.69% on Anti-UAV410.
* **Project relevance**: A sober data point. Tiny-target tracking is far from solved, which is exactly the regime a wide-FOV Case 1 camera lives in.

### 2.4 Air-to-Air Visual Detection of Micro-UAVs: An Experimental Evaluation of Deep Learning (Det-Fly)
* **Authors**: Ye Zheng, Zhang Chen, Dailin Lv, Zhixing Li, Zhenzhong Lan, Shiyu Zhao
* **Venue**: *IEEE Robotics and Automation Letters*, 6(2):1020-1027 (2021)
* **Links**: [DOI:10.1109/LRA.2021.3056059](https://doi.org/10.1109/LRA.2021.3056059) · [Dataset](https://github.com/Jake-WU/Det-Fly)
* **Key findings**: Det-Fly has 13,000+ images of a target UAV shot from another flying UAV, across varied backgrounds, angles, distances and lighting. Eight deep-learning detectors are evaluated on it.
* **Project relevance**: Case 3 (air-to-air) only. Not the current focus.

### 2.5 UETT4K Anti-UAV: A Large Scale 4K Benchmark Dataset for Vision-Based Drone Detection in High-Resolution Imagery
* **Authors**: Mughees Sarwar Awan, Syed Azhar Ali Zaidi, Junaid Mir
* **Venue**: *IEEE Access*, 13:73553-73568 (2025)
* **Links**: [DOI:10.1109/ACCESS.2025.3562890](https://doi.org/10.1109/ACCESS.2025.3562890)
* **Key findings**: 33,601 hand-annotated **4K** images from real drone videos in diverse conditions. Eight YOLOv6v3 models are compared.
* **Project relevance**: High. 4K is the resolution this project needs for pixels-on-target (see [13 §2](13_system_requirements_and_geometry_budget.md#2-pixels-on-target-the-constraint-that-dominates-everything)).

### 2.6 Experimental study and evaluation of ground-to-air drone visual detection based on deep learning
* **Authors**: Jun Li, Ban Wang, Qitian Cui, Feng Tian, Maoying Zhou, Rougang Zhou
* **Venue**: *Signal, Image and Video Processing*, 19(12):1003 (2025)
* **Links**: [DOI:10.1007/s11760-025-04616-4](https://doi.org/10.1007/s11760-025-04616-4)
* **Key findings**: Ground-to-air detector evaluation *(abstract not reviewed)*.
* **Project relevance**: Same geometry as Case 1. Read this one first when implementation starts.

### 2.7 WRN-YOLO: An Improved YOLO for Drone Detection using Wide ResNet
* **Authors**: Yi Jie Wong, Wingates Voon, Mau-Luen Tham, Ban-Hoe Kwan, Yoong Choon Chang, Yan Chai Hum
* **Venue**: IJCNN 2025 (8th WOSDETC Drone-vs-Bird Detection Challenge)
* **Links**: [Code](https://github.com/yjwong1999/IJCNN2025-DvB) · [Challenge](https://wosdetc2025.wordpress.com/) · [Annotations](https://github.com/wosdetc/challenge)
* **Key findings**: Swaps a Wide ResNet backbone into YOLO (via `timm`) for the Drone-vs-Bird challenge *(abstract not reviewed; metadata from the authors' repository)*.

### 2.8 YOLOBirDrone: Dataset for Bird vs Drone Detection and Classification and a YOLO based enhanced learning architecture
* **Authors**: Dapinder Kaur, Neeraj Battish, Arnav Bhavsar, Shashi Poddar
* **Venue**: arXiv preprint (2026)
* **Links**: [arXiv:2601.08319](https://arxiv.org/abs/2601.08319) · [Repository](https://github.com/dapinderk-2408/YOLOBirDrone)
* **Key findings**: Introduces the BirDrone dataset: 8,428 still images with bird and drone boxes, taken in Chandigarh, India with a webcam and a phone camera. The study combines it with 3,067 frames from four Drone-vs-Bird videos (11,495 images; 13,881 drones, 15,867 birds). A YOLO variant with layer aggregation and attention modules reaches about 85% detection accuracy.
* **Project relevance** (checked 2026-09-29):
  * **Not a substitute for Drone-vs-Bird.** These are still images, so there are no tracks for M2's motion features and no test of the recording confound.
  * Most targets are medium or large: 1,129 of ~29,700 boxes are under 20×20 px. That is little for the small-target problem.
  * The repository (838 KB, no licence file) says the data is only **partially available**.
  * Still useful later: boxed birds *and* drones in the same image set could help the detector's bird/drone appearance. Check the release and its licence first. Requested by email 2026-09-30.
* **Architecture** (read 2026-09-30): YOLOv9 (variant not stated) with three changes. **AELAN** swaps the CSP convolutions for deformable convolutions (learned sampling offsets). **MPDA** is multi-kernel (3/5/7/9) features with spatial then channel attention (EPSA/ECA-style) at the high-resolution pyramid levels. **RMPDA** is the same with the order reversed at the deep levels. The head is unchanged (no P2).
* **How strong is the evidence** (checked against its tables):
  * **About a tie with plain YOLOv8.** Full model mAP50 0.948 / mAP50-95 0.668 against YOLOv8 at 0.947 / 0.661 (Table V). These are single runs, with no seeds or intervals.
  * **The ablation is not additive.** Each module alone leaves mAP50-95 at or below the YOLOv9 baseline (0.629-0.639 against 0.644). Only all three together reach 0.668. With one run per row, this is within what training noise can produce.
  * **Missing details.** Aggregate metrics only (no per-size or per-class results, so nothing on the < 20 px targets). Model sizes are not given. "Detection accuracy" is undefined. The split sentence is truncated. Timings (0.149 s/frame, on an unnamed "RTX" GPU) are 30-60× slower than normal YOLO inference, so they are not comparable.
  * **No YOLO26 comparison** (it has a small-target label assignment, STAL). **No model code** in the repository (only a README and ~30 sample images; checked with the GitHub API).
* **Verdict for this project:** not better than YOLO26s on the evidence given. Nothing is adopted now ([11 row 8](11_research_deliberations_and_user_pushback_log.md#1-pushback--resolution-log)). Deformable convolutions are the worst fit for edge hardware. The cheap attention blocks are a possible later ablation, run with our per-video intervals.

### 2.9 Video, motion and track-sequence methods (abstracts checked 2026-10-01)

Found in the 2026-10-01 design review ([19 §5](19_design_review_parameters_and_novel_directions.md#5-novel-directions-ranked-against-this-projects-measured-failures)). **Only the abstracts were read**, so the summaries repeat the authors' claims; none has been run here. The first three are air-to-air (a moving camera), unlike this project's fixed camera.

| Paper | What it does | Project relevance |
| :--- | :--- | :--- |
| **YOLOMG**: Guo, Lin, Zhao, [arXiv:2503.07115](https://arxiv.org/abs/2503.07115) (2025) | Feeds a YOLOv5-based detector the RGB frame **plus a pixel-level motion difference map**, fused in a bimodal module. New dataset ARD100 (100 videos, 202,467 frames) | The closest published form of "temporal input" (19 §5.1). Our registration already produces the aligned frames it needs |
| **TransVisDrone**: Sangam, Dave, Sultani, Shah, [arXiv:2210.08423](https://arxiv.org/abs/2210.08423) (2022) | CSPDarkNet-53 features into a VideoSwin spatio-temporal transformer; end to end over several frames | A heavier way to use time. A candidate only if a frame stack into the existing CNN fails |
| **Dogfight**: Ashraf, Sultani, Shah, [arXiv:2103.17242](https://arxiv.org/abs/2103.17242) (2021) | Two-stage segmentation with spatio-temporal attention for small, erratic drones | Evidence that motion cues carry small-drone detection; segmentation labels are not available to us |
| **GL-YOMO**: Liu, Plotegher, Roura, de Souza Junior, He, [arXiv:2411.02582](https://arxiv.org/abs/2411.02582) (2024) | YOLO plus multi-frame motion detection by template matching, for small UAVs at long range | Same shape as our pipeline (detector + motion stage); their dataset is their own |
| **Dual-interval motion cues**: Wang, Zhang, [arXiv:2605.22605](https://arxiv.org/abs/2605.22605) (2026) | Homography compensation, motion at **two** frame intervals, motion-guided attention in the pyramid | Two intervals cover slow and fast apparent motion: relevant to choosing the frame gap k of a stack |
| **Drone-vs-Bird 2025 winner**: Laroca, dos Santos, Menotti, [arXiv:2504.19347](https://arxiv.org/abs/2504.19347) (2025) | YOLOv11 with **multi-scale inference, copy-paste augmentation and frame-to-frame consistency post-processing**. First place, 8th WOSDETC challenge (IJCNN 2025) | The challenge was won with data, scale and temporal consistency, not a new architecture: the same levers our evidence points to |
| **Sequence models for drone vs bird**: Akyon, Akagunduz, Altinuc, Temizel, [arXiv:2207.10409](https://arxiv.org/abs/2207.10409) (2022) | Classifies detection **tracks** with a 3D CNN, an LSTM and a Transformer; R(2+1)D best; bird F1 up to +73% | A learned alternative to our hand-built track features, on Drone-vs-Bird data. Worth testing once those videos arrive |

### 2.10 Wider sweep: long-range data, ground-to-air detectors, challenges (checked 2026-10-01)

From the wider literature sweep of [19 §10](19_design_review_parameters_and_novel_directions.md#10-wider-literature-sweep-what-we-are-doing-wrong). Abstracts read unless noted; none run here.

| Paper | What it does | Project relevance |
| :--- | :--- | :--- |
| **LRDDv3**: Peterson et al., [arXiv:2605.25942](https://arxiv.org/abs/2605.25942) (ICRA 2026) | 102,532 4K RGB images of drones at 0-200 m **with range labels**, plus 6,031 bird and 722 airplane boxes. YOLOv11m: mAP50 0.543 at 640 input, 0.822 at 1920 | The only set found with real small drones, **range truth** and birds in the same scenes. **Access is on hold** (project page, 2026-10-01); see 03 and 19 §10 |
| **LRDDv2**: Rouhi et al., [arXiv:2508.03331](https://arxiv.org/abs/2508.03331) | 39,516 1080p images, most drones at 50 px or fewer, 8,000+ with range | Smaller predecessor; same use |
| **SDD-YOLO**: Chen et al., [arXiv:2603.25218](https://arxiv.org/abs/2603.25218) | YOLO26-style ground-to-air detector with a P2 head; 86.0% mAP50 (n model) on its own 30,000-image set | The closest published architecture to ours: ours plus a P2 head |
| **DroneDAR**: Peterson et al., [arXiv:2606.07756](https://arxiv.org/abs/2606.07756) | Range from one camera using the crop and box geometry | The learned version of 09 §1.1's range estimate; trained on LRDD-type data |
| **Drone-vs-Bird 2020 challenge**: Coluccia et al., *Sensors* 21(8):2824, [DOI:10.3390/s21082824](https://doi.org/10.3390/s21082824) | Three entries compared: very distant drones and moving cameras are hardest | The field's own statement that small size, not architecture, is the limit |
| **Static-background two-stage detector**: Seidaliyeva et al., *Sensors* 20(14):3856, [DOI:10.3390/s20143856](https://doi.org/10.3390/s20143856) | Background subtraction, then a CNN classifying drone / bird / background | Our motion-cueing design, published; its stated limit (a moving background) is the one we hit |
| **Trajectory classification**: Nesteruk et al., *Sensors* 26(3):755, [DOI:10.3390/s26030755](https://doi.org/10.3390/s26030755) | BiLSTM on coordinate sequences, trained on **simulated** tracks | Same idea as M2's features; shows simulated trajectories being used for training, with the risk 11 row 9 names |

---

## 3. Radio Frequency (RF)

### 3.1 RFUAV: A Benchmark Dataset for Unmanned Aerial Vehicle Detection and Identification
* **Authors**: Rui Shi, Xiaodong Yu, Shengming Wang, Yijia Zhang, Lu Xu, Peng Pan, Chunlai Ma
* **Venue**: arXiv preprint (2025)
* **Links**: [arXiv:2503.09033](https://arxiv.org/abs/2503.09033) · [Code](https://github.com/kitoweeknd/RFUAV) · [Data (Hugging Face)](https://huggingface.co/datasets/kitofrank/RFUAV)
* **Key findings**: About **1.3 TB** of raw RF from **37 UAVs**, recorded with a USRP. Includes tools to re-render data at different SNRs, an "RF drone fingerprint" feature sequence, and open evaluation tooling. The label set includes several RC *transmitters* (e.g. FlySky, FrSky, Futaba), not only airframes.

### 3.2 RF-based drone detection and identification using deep learning approaches (DroneRF)
* **Authors**: Mohammad F. Al-Sa'd, Abdulla Al-Ali, Amr Mohamed, Tamer Khattab, Aiman Erbad
* **Venue**: *Future Generation Computer Systems*, 100:86-97 (2019). Pre-2020, kept as the foundational open RF dataset
* **Links**: [DOI:10.1016/j.future.2019.05.007](https://doi.org/10.1016/j.future.2019.05.007) · [Code](https://github.com/Al-Sad/DroneRF) · [Data](http://dx.doi.org/10.17632/f4c2b4n755.1)
* **Key findings**: Accuracy falls sharply as the task gets finer: **99.7%** detection (2 classes), **84.5%** drone type (4 classes), **46.8%** type + flight mode (10 classes), under 10-fold cross-validation.

---

## 4. Radar & Micro-Doppler

### 4.1 DIAT-µSAT: Small Aerial Targets' Micro-Doppler Signatures and Their Classification Using CNN
* **Authors**: Harish C. Kumawat, Mainak Chakraborty, A. Arockia Bazil Raj, Sunita Vikrant Dhavale
* **Venue**: *IEEE Geoscience and Remote Sensing Letters*, 19:1-5 (2022)
* **Links**: [DOI:10.1109/LGRS.2021.3102039](https://doi.org/10.1109/LGRS.2021.3102039)
* **Key findings**: 4,849 micro-Doppler images from an X-band (10 GHz) CW radar. Classes are rotors with different blade counts, a quadcopter, a bionic bird, and rotor+bird together. Transfer-learning DCNNs reach **95%** (VGG16) and **97%** (VGG19).

### 4.2 Exploring Radar Micro-Doppler Signatures for Recognition of Drone Types
* **Authors**: Jun Yan, Huiping Hu, Jiangkun Gong, Deyong Kong, Deren Li
* **Venue**: *Drones*, 7(4):280 (2023)
* **Links**: [DOI:10.3390/drones7040280](https://doi.org/10.3390/drones7040280)
* **Key findings**: Compares puller- and lifting-blade micro-Doppler across fixed-wing, multirotor and hybrid VTOL drones. **Lifting blades produce stronger micro-Doppler** than puller blades in all three types.

---

## 5. Acoustic

### 5.1 Audio-Based Drone Detection and Identification Using Deep Learning Techniques with Dataset Enhancement through GANs
* **Authors**: Sara Al-Emadi, Abdulla Al-Ali, Abdulaziz Al-Ali
* **Venue**: *Sensors*, 21(15):4953 (2021)
* **Links**: [DOI:10.3390/s21154953](https://doi.org/10.3390/s21154953) · [Dataset](https://github.com/saraalemadi/DroneAudioDataset)
* **Key findings**: A hybrid acoustic dataset of recorded plus GAN-generated drone audio, used to compare CNN, RNN and CRNN classifiers. GAN augmentation improves detection of **unfamiliar** drones.

---

## 6. Multi-Modal Sensor Fusion & Uncertainty

### 6.1 A Tri-Modal Dataset and a Baseline System for Tracking Unmanned Aerial Vehicles (MM-UAV)
* **Authors**: Tianyang Xu, Jinjie Gu, Xuefeng Zhu, Xiaojun Wu, Josef Kittler
* **Venue**: arXiv preprint (Nov 2025)
* **Links**: [arXiv:2511.18344](https://arxiv.org/abs/2511.18344)
* **Key findings**: 1,321 synchronised **RGB + IR + event** sequences with 2.8M+ annotated frames across 30+ scenarios. The baseline adds offset-guided sensor alignment, adaptive dynamic fusion, and event-enhanced association.

### 6.2 Fusion of Cellular ISAC and Passive RF Sensing for UAV Detection and Tracking
* **Authors**: Cole Dickerson, Sean Kearney, Sultan Manjur, Ismail Guvenc, Sevgi Gurbuz, Ali Gurbuz, Ozgur Ozdemir, Mihail Sichitiu
* **Venue**: IEEE Asilomar Conference on Signals, Systems, and Computers (2025)
* **Links**: [arXiv:2512.14608](https://arxiv.org/abs/2512.14608) · [AERPAW dataset](https://aerpaw.org/dataset/multi-modal-rf-sensor-and-radar-dataset-for-uav-tracking/)
* **Key findings**: Real flights at the NSF AERPAW testbed show passive RF (Keysight N6841A) and Ku-band radar (Fortem TrueView R20) have complementary strengths. A plain **constant-velocity Kalman filter** fusing asynchronous 2D RF and 3D radar suppresses large single-sensor errors.

### 6.3 Evidential Deep Learning for Multi-Modal Anti-UAV Detection
* **Authors**: Dmitry Golovchits, Seyed Sahand Mohammadi Ziabari, Ali Mohammed Mansoor Alsahag
* **Venue**: arXiv preprint (Sep 2026)
* **Links**: [arXiv:2609.01742](https://arxiv.org/abs/2609.01742)
* **Key findings**: Ablation on AntiUAV600, TRIDENT and MM-UAV. The evidential *training objective* helps (+5.9 / +4.8 accuracy points). **Dempster-Shafer fusion does not beat simple probability averaging**, and temporal gating gives no real latency saving.
* **Project relevance**: A direct counter-example to the Dempster-Shafer stage drawn in [05 §3](05_ai_technical_architectures.md#3-multi-sensor-data-fusion-architecture). Start with averaging.

### 6.4 Uncertainty-Aware Multimodal Anti-UAV Detection via Evidential Fusion and Conflict-Discounted Belief Aggregation
* **Authors**: Sharanda Suttorp, Seyed Sahand Mohammadi Ziabari, Ali Mohammed Mansour Alsahag
* **Venue**: arXiv preprint (Aug 2026)
* **Links**: [arXiv:2608.29235](https://arxiv.org/abs/2608.29235)
* **Key findings**: RGB-thermal fusion beats either stream on Anti-UAV (accuracy 0.670 vs 0.604 IR / 0.598 RGB) at 38+ FPS. Uncertainty is well calibrated (ECE 0.057), but conflict-discounted fusion is **indistinguishable from plain averaging**.
* **Project relevance**: Shows how to *measure* calibration (ECE), which is needed before trusting any "calibrated" score, including Jev's ([12](12_typesafe_jev_system_one_c2_threat_engine.md)).

---

## 7. Electronic Warfare & Navigation Security

Both verified papers here protect *friendly* UAVs. No verified paper on AI-driven hostile-drone interception or spoofing was found (see [§10](#10-known-gaps)).

### 7.1 Meta-Reinforcement Learning in Time-Varying UAV Communications: Adaptive Anti-Jamming Channel Selection
* **Authors**: L. Hu, Y. Shao, Y. Qian, F. Du, J. Li, Y. Lin, Z. Wang
* **Venue**: *Radioengineering*, 33(3):417-431 (2024)
* **Links**: [DOI:10.13164/re.2024.0417](https://doi.org/10.13164/re.2024.0417)
* **Key findings**: Models channel selection for a multi-UAV network facing a Markov jammer as a partially observable stochastic game, solved with meta-mean-field Q-learning for fast adaptation.

### 7.2 Deep Learning-Based GNSS Spoofing Attack Detection for Quadrotor UAV
* **Authors**: Yupeng Zhu, Zetao Huang, Quanqi Zhang, Zhuoyu Li, Chengwei Wu
* **Venue**: Chinese Control Conference (CCC), 2025, pp. 9331-9336
* **Links**: [DOI:10.23919/CCC64809.2025.11178696](https://doi.org/10.23919/CCC64809.2025.11178696)
* **Key findings**: A DNN predicts state-estimation residuals from onboard physical measurements, and deviations flag GNSS spoofing. Validated in simulation.

---

## 8. Frontier Detectors & Synthetic Data (2024-2026)

### 8.1 Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models
* **Authors**: Glenn Jocher, Jing Qiu, Mengyu Liu, Shuai Lyu, Fatih Cagatay Akyon, Muhammet Esat Kalfaoglu
* **Venue**: arXiv preprint (June 2026)
* **Links**: [arXiv:2606.03748](https://arxiv.org/abs/2606.03748) · [Docs](https://docs.ultralytics.com/models/yolo26/)
* **Key findings**:
  * Optional NMS-free one-to-one head (`nms=False`); the default path still uses NMS.
  * No Distribution Focal Loss.
  * MuSGD optimizer (Muon + SGD) and Progressive Loss.
  * **STAL**, which guarantees positive label coverage for small objects.
  * 40.9-57.5 COCO mAP at 1.7-11.8 ms, measured on **NVIDIA T4 + TensorRT** (not Jetson).
* **Project relevance**: The chosen detector. STAL targets exactly the tiny-object failure mode. **Licence: AGPL-3.0.** Closed or commercial deployment needs an Ultralytics Enterprise licence.

### 8.2 DroneGround: Open-Vocabulary Drone Payload Characterization Using Synthetic Data and Grounded Vision-Language Models
* **Authors**: Ami Pandat, Rajasekhar Punna, Gopika Vinod, Rohit Shukla
* **Venue**: Accepted at RVS-SE Workshop, BMVC 2026 (arXiv preprint, Sep 2026)
* **Links**: [arXiv:2609.07780](https://arxiv.org/abs/2609.07780)
* **Key findings**:
  * **~100,000 synthetic images** (Unreal Engine 5 + Cosys-AirSim): 15 platforms, 6 payload categories, 5 bird species, 7 environments, 5 weather conditions.
  * YOLO26s localises the drone and a LoRA-tuned PaliGemma VLM describes its payload.
  * F1 rises from 82.5% to 96.3%, and reaches 80.4% on unseen payloads (vs 42.7%).
* **Project relevance**: A useful synthetic pre-training source. Two caveats: it is **entirely synthetic**, and data is **released on request only**.

### 8.3 SimD3: A Synthetic drone Dataset with Payload and Bird Distractor Modeling for Robust Detection
* **Authors**: Ami Pandat, Kanyala Muvva, Punna Rajasekhar, Gopika Vinod, Rohit Shukla
* **Venue**: arXiv preprint (Jan 2026)
* **Links**: [arXiv:2601.14742](https://arxiv.org/abs/2601.14742)
* **Key findings**: UE5 synthetic drones with payloads and bird distractors, under controlled weather and lighting, captured by a 360° six-camera rig. A YOLOv5m+C3b variant beats baseline YOLOv5m in-domain and on unseen real benchmarks.

### 8.4 Neuromorphic Drone Detection: an Event-RGB Multimodal Approach (NeRDD)
* **Authors**: Gabriele Magrini, Federico Becattini, Pietro Pala, Alberto Del Bimbo, Antonio Porta
* **Venue**: ECCV 2024 Workshops (NeVi)
* **Links**: [arXiv:2409.16099](https://arxiv.org/abs/2409.16099) · [Dataset](https://github.com/MagriniGabriele/NeRDD)
* **Key findings**: Fuses event and RGB streams for drone detection. Releases NeRDD: **3.5+ hours** of synchronised, annotated event+RGB recordings in 115 videos at 30 FPS. Event cameras handle fast motion and low light, but lose information on static targets.

### 8.5 Beyond Clear Skies: Synthetic Seasonal and Weather Variations for Real-World Drone Detection
* **Authors**: Tamara R. Lenhard, Andreas Weinmann, Tobias Koch
* **Venue**: arXiv preprint (Aug 2026)
* **Links**: [arXiv:2608.16191](https://arxiv.org/abs/2608.16191)
* **Key findings** (abstract only): extends SynDroneVision with 55,187 game-engine images of three urban environments under rain, snow and fog. Added to synthetic drone-detection training, it reduces missed detections and false alarms on real benchmarks under adverse conditions.
* **Project relevance**: support for rendered data as a *supplement* for conditions that are hard to film. It does not change [11 row 9](11_research_deliberations_and_user_pushback_log.md#1-pushback--resolution-log): our first gap is real-sensor noise, blur and compression at a few pixels (E21), which degrading real frames reproduces directly.

### 8.6 Event-based Tiny Object Detection: A Benchmark Dataset and Baseline (EV-UAV)
* **Authors**: Nuo Chen, Chao Xiao, Yimian Dai, Shiman He, Miao Li, Wei An
* **Venue**: arXiv preprint (2025)
* **Links**: [arXiv:2506.23575](https://arxiv.org/abs/2506.23575)
* **Key findings** (abstract only): 147 event-camera sequences of UAVs averaging 6.8 × 5.4 px, with 2.3 million event-level annotations. Small moving targets trace continuous curves in the event point cloud.
* **Project relevance**: a different sensor that answers the frame-rate limit of [18](18_hardware_optimisation_and_realtime_latency.md) directly (microsecond timing, no motion blur). Outside the current two-camera plan; noted as the alternative if 10 fps at full resolution proves too slow.

---

## 9. Transferable Techniques (Not Drone-Detection Papers)

These papers are often cited in drone-defense discussions (including earlier drafts of this repo) but **study a different task**. Their techniques may transfer; their accuracy numbers do not.

| Paper | What it actually does | What might transfer |
| :--- | :--- | :--- |
| **HA-YOLO**: Wang et al., *Automation* 7(1):3 (2025), [DOI:10.3390/automation7010003](https://doi.org/10.3390/automation7010003) | Detects ground objects in fog, in imagery taken *from* UAVs (YOLO11 + wavelet conv + CEHSA attention) | Fog-robust neck design for Case 1 bad-weather work |
| **Gamba**: Shen et al., [arXiv:2403.18795](https://arxiv.org/abs/2403.18795) | Single-image 3D Gaussian-splat reconstruction of object-centric images (Objaverse) with a Mamba backbone | Nothing for ranging distant drones; possibly asset generation for synthetic data |
| **GSGTrack**: Chen et al., [arXiv:2412.02267](https://arxiv.org/abs/2412.02267) | 6-DoF pose tracking of *close-range* objects for robotic manipulation | Only relevant if a drone fills hundreds of pixels (terminal phase of an interceptor) |
| **TrackingMiM**: Liu et al., [arXiv:2507.01535](https://arxiv.org/abs/2507.01535) | Real-time single-object tracking in video *from* UAV cameras (Mamba-in-Mamba) | Evidence that SSM trackers can run in real time |
| **MambaSOD**: Zhan et al., *Neurocomputing*, [arXiv:2410.15015](https://arxiv.org/abs/2410.15015) | RGB-D **salient** object detection ("SOD" ≠ small-object detection) | Nothing. Listed so it is not misread again |
| **NWD**: Wang et al., [arXiv:2110.13389](https://arxiv.org/abs/2110.13389) | Models boxes as 2-D Gaussians and replaces IoU with their normalised Wasserstein distance for tiny objects | A loss / assignment / matching measure for targets under ~8 px, where IoU jumps with one pixel of error |
| **SPD-Conv**: Sunkara & Luo, [arXiv:2208.03641](https://arxiv.org/abs/2208.03641) | Replaces strided convolution and pooling with space-to-depth plus a plain convolution | Keeps fine detail in the first downsampling stages; the one non-standard operator with a mechanism that fits few-pixel targets |
| **LP-FT**: Kumar et al., [arXiv:2202.10054](https://arxiv.org/abs/2202.10054) | Shows full fine-tuning can distort pretrained features and lose out-of-distribution accuracy; probe first, then fine-tune (classification) | Our failure is out-of-distribution (E15). Freezing or slowing the backbone is one training argument |
| **DINOv3**: Siméoni et al., [arXiv:2508.10104](https://arxiv.org/abs/2508.10104) | Self-supervised foundation model with stable dense features (Gram anchoring) | A backbone pretrained far more broadly than COCO, for the same out-of-distribution failure |
| **DEIMv2**: Huang et al., [arXiv:2509.20787](https://arxiv.org/abs/2509.20787) | Real-time DETR-style detectors on DINOv3 features; 9.71 M parameters at COCO AP 50.9 (S) | The concrete detector to try behind `cuas/detector.py` if a foundation backbone is tested; check its licence and its stride for 8-px targets first |
| **RF-DETR**: Robinson et al., [arXiv:2511.09554](https://arxiv.org/abs/2511.09554) | Detection transformer tuned to a dataset by weight-sharing architecture search | The Apache-2.0 alternative already listed in [13 §5](13_system_requirements_and_geometry_budget.md#5-legal--operational-scope) |
| **XS-VID**: Guo et al., [arXiv:2407.18137](https://arxiv.org/abs/2407.18137) | Aerial video dataset with objects under 12×12 px, and a detector with temporal motion features | Benchmark and method for very small *video* objects; filmed from the air |
| **SAHI**: Akyon et al., [arXiv:2202.06934](https://arxiv.org/abs/2202.06934) | Runs any detector on overlapping slices, and fine-tunes on slices; AP gains of 12.7-14.5% on VisDrone and xView | The published form of our tiling (`cuas/tiling.py`); confirms tiles for training as well as inference |
| **Scale Match**: Yu et al., [arXiv:1912.10664](https://arxiv.org/abs/1912.10664) | Aligns the object-size distribution of the pretraining data with the target data for tiny objects | The reason zoom-out mosaics worked (E2): COCO pretraining has few 8-px objects |
| **Small-object augmentation**: Kisantal et al., [arXiv:1902.07296](https://arxiv.org/abs/1902.07296) | Oversamples images with small objects and copy-pastes small objects; +7.1% relative on small objects (COCO) | Copy-paste of shrunk drone crops onto real backgrounds, instead of whole-frame mosaics (19 §9) |
| **TOOD (Task Alignment Learning)**: Feng et al., [arXiv:2108.07755](https://arxiv.org/abs/2108.07755) | Assigns training samples by a joint classification-localisation metric | The scheme Ultralytics' assigner descends from; 19 §9.2 counts what it gives a small drone |
| **SOD4SB challenge**: Kondo et al., [arXiv:2307.09143](https://arxiv.org/abs/2307.09143) | 39,070 4K images, 137,121 birds, 95% under 32 px. Baseline AP50 46.4 public → 15.4 private; winner 30.3 with copy-paste, NWD loss, slicing, test-time augmentation, ensembling | A large open source of small birds; and proof that the public-to-private collapse we saw on DUT is the field's normal |
| **Small-bird YOLOv7**: Shigematsu, [arXiv:2401.01018](https://arxiv.org/abs/2401.01018) | Higher input resolution, multi-scale and flip inference, weighted boxes fusion | Resolution and multi-scale inference, again, not architecture |
| **SMOT4SB challenge**: Kondo et al., [arXiv:2507.12832](https://arxiv.org/abs/2507.12832) | 211 videos of tiny birds; SO-HOTA scores tracks by **Dot Distance**, not IoU | Bird tracks for M2's negative class; the field's verdict that IoU is the wrong match for tiny objects |
| **YOLOv8-SMOT**: Yu et al., [arXiv:2507.12087](https://arxiv.org/abs/2507.12087) | Slice training; appearance-free association with expanded boxes and a distance penalty | Agrees with our tracker's choice: motion only, never box overlap alone |
| **RFLA**: Xu et al., [arXiv:2208.08738](https://arxiv.org/abs/2208.08738) | Assigns labels by receptive-field distance so tiny boxes get positives | The problem 19 §9.2 measured (5.3 positives against 10) |
| **DQ-DETR**: Huang et al., [arXiv:2404.03507](https://arxiv.org/abs/2404.03507) | DETR with a count-driven number of queries; 30.2 mAP on AI-TOD-V2 | The DETR-family reference for tiny objects if E25 goes ahead |
| **QueryDet**: Yang et al., [arXiv:2103.09136](https://arxiv.org/abs/2103.09136) | Coarse features say where to compute the high-resolution head | The in-network version of our motion-cued crops |
| **SODA survey**: Cheng et al., [arXiv:2207.14096](https://arxiv.org/abs/2207.14096) | Survey and two benchmarks for small objects | Background reading; names poor appearance and noisy features as the limit |
| **DNA-Net**: Li et al., [arXiv:2106.00487](https://arxiv.org/abs/2106.00487) | Keeps infrared point targets alive through deep layers by dense cross-level interaction | Same mechanism as a P2 head or SPD-Conv, from the point-target literature |
| **Background Recalibration Loss**: Zhang et al., [arXiv:2002.05274](https://arxiv.org/abs/2002.05274) | Down-weights "background" loss where an unlabeled object is likely | Our datasets label one target per frame; the others are trained as background (19 §10) |
| **SparseDet**: Suri et al., [arXiv:2201.04620](https://arxiv.org/abs/2201.04620) | Self-supervised loss on unlabeled regions instead of noisy pseudo-labels | Same problem, alternative remedy |
| **Robustness benchmark**: Michaelis et al., [arXiv:1907.07484](https://arxiv.org/abs/1907.07484) | Detectors fall to 30-60% on corrupted images; stylised training helps | E21 reproduced this on our detector; the remedy is training-time |
| **PhysAug**: Xu et al., [arXiv:2412.11807](https://arxiv.org/abs/2412.11807) | Frequency-domain augmentation modelled on atmospheric optics | A candidate augmentation for E22 beyond noise, blur and JPEG |
| **OA-DG**: Lee et al., [arXiv:2312.12133](https://arxiv.org/abs/2312.12133) | Object-aware mixing augmentation and loss for unseen domains | Same aim as E22 with object semantics protected |
| **Box-size confidence bias**: Gilg et al., [arXiv:2112.01901](https://arxiv.org/abs/2112.01901) | Detector confidence depends on box size; calibrate per size | Why one threshold does not serve all sizes (14 §5: scores top out near 0.5) |
| **Multivariate calibration**: Küppers et al., [arXiv:2004.13546](https://arxiv.org/abs/2004.13546) | Calibrates confidence using box position and scale | Same, with position |
| **Model soups**: Wortsman et al., ICML 2022, [arXiv:2203.05482](https://arxiv.org/abs/2203.05482) | Averages the weights of several models fine-tuned from one pretrained model: one model, no extra inference cost, often more accurate and more robust than any member (image classification) | E31: average our seeds into one model, to remove the seed lottery from the deployed weights (19 §3.8) |
| **Open-vocabulary detectors on aerial images**: Tsourveloudis, [arXiv:2601.22164](https://arxiv.org/abs/2601.22164) | Zero-shot detectors fail: best F1 27.6%, others under 4% recall | A foundation model must be fine-tuned to help; it will not work zero-shot |

---

## 10. Known Gaps

The audit removed several topic areas because no real paper could be found behind the claims. They remain open questions for this compendium:

* **AI-guided kinetic interception of hostile drones** (e.g. DRL pursuit policies with field-trial results).
* **Adversarial robustness of RF drone classifiers.**
* **Ultra-low-power (spiking / neuromorphic) acoustic detection.**
* **Mixture-of-experts RF fingerprinting in congested spectrum.**

When adding papers, use `python tools/fetch_arxiv_papers.py` to find candidates. Then follow the verification steps in the [README](README.md#adding-or-verifying-a-reference): confirm at the source, mark the entry verified in the JSON, regenerate the `.bib`, and cite it here.

---

## Appendix A: What Changed in the 2026-09-21 Audit

**Removed: no matching paper exists** (identifier dead or unrelated, and no title match in Crossref, arXiv or Semantic Scholar). Several had invented authors derived from GitHub usernames (e.g. `JoulesSpace` → "J. Joules, K. Space"; `r4d10n` → "S. Radionov"):

| Old § | Claimed title (abridged) | Claimed statistic that should not be reused |
| :--- | :--- | :--- |
| 3.3 | Mixture-of-Experts for Robust Drone RF Fingerprinting | 94.2% accuracy in congested spectrum |
| 3.4 | Adversarial Robustness of DL Models for Drone RF ID | 96% restored accuracy |
| 4.1 | Micro-Doppler Radar Classification of Small UAVs vs. Birds | 98.9% discrimination |
| 4.3 | Transformer-Based Micro-Doppler … Small Training Samples | - |
| 5.2 | VolAnti … DoA Estimation (IEEE TIM) | ±3.2° DoA (the VolAnti *repository* is real: see [02](02_github_repositories_and_codebases.md)) |
| 5.3 | Spiking Neural Networks … Acoustic Drone Detection | 92.5% at 15 mW |
| 6.2 | Multi-Modal Sensor Fusion for Counter-UAS (*Information Fusion*) | 99.8% Pd, 94% fewer false alarms |
| 7.1 | DRL for Adaptive Anti-Jamming (IEEE JSAC) | 4.2× faster adaptation |
| 7.2 | Autonomous Aerial Interception … DRL (IEEE T-RO) | 88.5% interception success |
| 7.3 | Deep Learning-Driven GNSS Spoofing and Detection | - |

**Corrected: real work, wrong metadata.** Anti-UAV (was "TPAMI", wrong ID); Anti-UAV410 (was "TIP", author "H. Bo" is Bo Huang); Det-Fly (wrong authors/ID); RFUAV (was "Zhou & Frank, IEEE TMC 2024"); DroneRF (was "Sensors 2021"; 84.5% was mislabelled as flight-mode accuracy); DIAT-µSAT (authors were taken from a third-party GitHub repo); DroneAudioDataset paper; MM-UAV (wrong ID, venue unconfirmed); Gamba and GSGTrack (wrong authors/ID); the C-UAS survey (title belonged to Wang et al., authors to Lykou et al.); the review in *Computer Science Review* (wrong authors, DOI off by two); the radar/comms survey (wrong authors; it is a TechRxiv preprint).

**Dropped as off-topic.** Old §1.5, "AI Methods for UAV Cybersecurity and Counter-UAS" (Khan et al., *Sensors* 2026). The closest real paper is Kacem & Benjamin, *Drones* 10(6):400 (2026, [DOI:10.3390/drones10060400](https://doi.org/10.3390/drones10060400)). It covers protecting UAVs from cyber-attack, not countering hostile drones.

**New since the audit:** CST Anti-UAV, UETT4K, the ground-to-air evaluation (SIVP 2025), the AERPAW RF+radar fusion paper, the two 2026 evidential-fusion studies, SimD3, and the §7 replacements (anti-jamming meta-RL, GNSS spoofing detection).

**Relabelled: real paper, different task.** PVswin-YOLOv8s (pedestrian/vehicle detection from UAVs; removed), UAVScenes (scene understanding from UAVs; removed), HA-YOLO and MambaSOD (moved to §9), Drone-vs-Bird 2025 (the cited repo is WRN-YOLO, not an RT-DETR/GAN system).
