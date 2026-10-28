# Benchmark Datasets for Drone Detection & Defense

Training and evaluating a counter-UAS model needs real, annotated data from the sensor and camera geometry you will actually deploy. This catalog lists the datasets we have verified, what they really contain, and how to load them.

> [!NOTE]
> **Verification status (2026-09-21).** Sizes, modalities and access routes were checked against each paper's abstract, repository or dataset page. Earlier versions of this file had wrong modality claims (Anti-UAV410 is thermal-only), wrong sizes (NeRDD, AERPAW), and ingestion code that could not run (see [§4](#4-loading-the-data)).

---

## 1. Master Comparison Table

**Fit** = fit to this project's Case 1 design (fixed ground camera looking up, EO first). ★★★ high · ★★ partial · ★ low.

| Dataset | Year | Modality | Scale (as published) | Real / Synthetic | Access | Fit |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **Anti-UAV** | 2021/23 | Paired RGB + thermal video | 300+ video pairs, 580k+ boxes | Real | [anti-uav.github.io](https://anti-uav.github.io/) · [repo](https://github.com/ZhaoJ9014/Anti-UAV) | ★★★ |
| **Anti-UAV410** | 2023/24 | **Thermal only** | 410 videos, 438k+ boxes | Real | [repo](https://github.com/HwangBo94/Anti-UAV410) | ★★ |
| **CST Anti-UAV** | 2025 | Thermal, **tiny** targets | 220 sequences, 240k+ boxes | Real | [arXiv:2507.23473](https://arxiv.org/abs/2507.23473) | ★★ |
| **Drone-vs-Bird** (WOSDETC) | 2017-2025 | Long-range RGB video, drones + birds | Growing per challenge edition | Real | [annotations](https://github.com/wosdetc/challenge) · [2025 site](https://wosdetc2025.wordpress.com/); videos via organisers | ★★★ |
| **LRDDv3** (Drexel) | 2026 | **4K** RGB video frames + paired IR; **range labels**; drones, birds, airplanes | 102,532 RGB images (5.7 h, 128 clips), 29,630 IR; range 0-200 m | Real | [project page](https://research.coe.drexel.edu/ece/imaple/lrddv3/) · [arXiv:2605.25942](https://arxiv.org/abs/2605.25942) · **access on hold** (2026-10-01) | ★★★ if obtainable |
| **LRDDv2** (Drexel) | 2025 | 1080p RGB, range for 8,000+ images | 39,516 images, most drones ≤ 50 px | Real | [project page](https://research.coe.drexel.edu/ece/imaple/lrddv2/) · [arXiv:2508.03331](https://arxiv.org/abs/2508.03331) · **access on hold** (2026-10-01) | ★★★ if obtainable |
| **SOD4SB** (MVA 2023) | 2023 | 4K RGB images of small **birds** | 39,070 images, 137,121 birds, 95% under 32 px | Real | [arXiv:2307.09143](https://arxiv.org/abs/2307.09143) | ★★ (bird negatives) |
| **SMOT4SB** (MVA 2025) | 2025 | RGB **video** of small birds from UAVs, tracked | 211 sequences, 108,192 frames | Real | [arXiv:2507.12832](https://arxiv.org/abs/2507.12832) | ★★ (bird tracks for M2; moving camera) |
| **UETT4K Anti-UAV** | 2025 | **4K** RGB images | 33,601 images | Real | See [paper](https://doi.org/10.1109/ACCESS.2025.3562890) | ★★★ |
| **Halmstad drone dataset** (Svanström et al.) | 2021 | Visible + IR video, audio; drone / bird / airplane / helicopter labels | 285 visible + 365 IR videos (visible 640×512, 89k frames) | Real | [repo](https://github.com/DroneDetectionThesis/Drone-detection-dataset) (CC0, 357 MB) | ★★ (used for E1; **no drones under ~20 px**, see [14 §5](14_build_log_and_experiments.md#5-findings-so-far)) |
| **Det-Fly** | 2021 | Air-to-air RGB | 13,000+ images | Real | [repo](https://github.com/Jake-WU/Det-Fly) | ★ (Case 3) |
| **NeRDD** | 2024 | Event camera + RGB, synchronised | 3.5+ h, 115 videos, 30 FPS | Real | [repo](https://github.com/MagriniGabriele/NeRDD) | ★★ |
| **MM-UAV** | 2025 | RGB + thermal + event | 1,321 sequences, 2.8M+ frames | Real | [arXiv:2511.18344](https://arxiv.org/abs/2511.18344) | ★★ |
| **DroneGround** | 2026 | RGB + payload/bird labels, 5 weather conditions | ~100,000 images | **Synthetic** (UE5 + Cosys-AirSim) | **On request only** ([arXiv:2609.07780](https://arxiv.org/abs/2609.07780)) | ★★ (pre-training) |
| **SimD3** | 2026 | RGB, payloads + bird distractors | Large-scale (see paper) | **Synthetic** (UE5) | [arXiv:2601.14742](https://arxiv.org/abs/2601.14742) | ★★ (pre-training) |
| **RFUAV** | 2025 | Raw RF (USRP) | ~1.3 TB, 37 devices | Real | [Hugging Face](https://huggingface.co/datasets/kitofrank/RFUAV) · [repo](https://github.com/kitoweeknd/RFUAV) | ★ (RF) |
| **DroneRF** | 2019 | Raw RF, 2.4 GHz band | 3 drone types + background, 5 flight modes | Real | [Mendeley Data](http://dx.doi.org/10.17632/f4c2b4n755.1) · [code](https://github.com/Al-Sad/DroneRF) | ★ (RF) |
| **DIAT-µSAT** | 2022 | X-band CW radar micro-Doppler images | 4,849 images, 6 classes | Real | See [paper](https://doi.org/10.1109/LGRS.2021.3102039) | ★ (radar) |
| **DroneAudioDataset** | 2019/21 | Acoustic clips | Recorded + GAN-augmented | Real + generated | [repo](https://github.com/saraalemadi/DroneAudioDataset) | ★ (acoustic) |
| **AERPAW Dataset-28** | 2025 | Passive RF position estimates + Ku-band radar tracks + UAV ground truth | 33 flights (+3 set-up flights) | Real | [dataset page](https://aerpaw.org/dataset/multi-modal-rf-sensor-and-radar-dataset-for-uav-tracking/) | ★ (fusion) |

**Removed from the previous version:** *UAVScenes* is a dataset for scene understanding *from* a UAV (built on MARS-LVIG), not for detecting drones. *UAV-CDT* could not be found.

---

## 2. What to Train On for Case 1 (Recommended Order)

1. **Pre-train** on synthetic data (DroneGround if granted, SimD3) to get bird distractors and weather variety cheaply.
2. **Fine-tune** on real ground-to-air RGB: Anti-UAV (RGB stream), Drone-vs-Bird, UETT4K.
3. **Hold out** a set recorded with *your own camera and lens* at your site. Public benchmarks never match your focal length, and focal length sets pixels-on-target (see [13](13_system_requirements_and_geometry_budget.md)). Report results on this set, not only on public ones.
4. **Stress-test** tiny targets on CST Anti-UAV (thermal) once a thermal channel is added.

### 2.1 Multi-Dataset Strategy (2026-09-24)

**Why more than one dataset.** E1 memorised its ~230 training videos after epoch 12 ([14 §5](14_build_log_and_experiments.md#5-findings-so-far)), so variety of cameras, sites, skies and drones is the main lever left. And Halmstad has no real small drones, so E2's small-target gain has only been shown on *simulated* distance.

**Access status (checked 2026-09-24):**

| Dataset | Visible data | Access | Labels | Role |
| :--- | :--- | :--- | :--- | :--- |
| Halmstad | 640×512 video | Free, CC0 | Drone + bird/airplane/helicopter | **In use** (E1, E2) |
| **Anti-UAV300** | **Full HD RGB** + IR video, 20 fps, 318 sequences (160/67/91) | Google Drive zip (6.0 GB; `gdown` hits the quota, the browser works); repo MIT. The 410/600 versions are **IR only** | One drone per frame + explicit visibility flag | **Large drones only** (median width 109-122 px; 43 of ~280k boxes < 16 px). Burned-in on-screen text + reticle (inpaint it). Used for E11: cross-dataset test at normal size, plus simulated distance |
| DUT Anti-UAV | 10,000 detection images | Google Drive; repo Apache-2.0 | Drones only | Training variety. Measured 2026-10-08 (16 #86): train 5,200 images / 5,243 drones, val 2,600 / 2,621, test 2,200 / 2,245; drone width p10 / median / p90 = 24 / 43 / 169 px in train (1% under 16 px), so it adds scenes and drone models, not small targets |
| **Drone-vs-Bird** | Long-range RGB video | **Email request + data usage agreement** (research use) | Drones only (verified 2026-09-30: birds are unlabelled distractors); training set only, test set withheld | Closest to the bird problem at range. **Request early; it takes time**. Reader: `cuas/datasets/dvb.py`. **Not the same as** Kaggle's "Drone vs Bird: Aerial Object Classification Dataset" (muhammadsaoodsarwar/drone-vs-bird; checked via the Kaggle API 2026-09-29): 1.7 GB of images in `drones/` and `birds/` folders, classification only, no boxes, no video, source not stated. Without video it gives no tracks, so it can't test M2 |
| **BirDrone** ([01 §2.8](01_research_papers_compendium.md#28-yolobirdrone-dataset-for-bird-vs-drone-detection-and-classification-and-a-yolo-based-enhanced-learning-architecture)) | Still RGB images, bird and drone boxes | [GitHub](https://github.com/dapinderk-2408/YOLOBirDrone): **partially available**, no licence file (checked 2026-09-29); requested by email 2026-09-30 | Birds and drones both boxed (8,428 images; plus 3,067 Drone-vs-Bird frames in the paper) | Mostly medium/large targets, still images (no tracks). Possible later detector data once fully released with a licence; not a Drone-vs-Bird substitute |
| SimD3 / DroneGround | Synthetic | See §1 | Drones + birds + payloads | Rare conditions later |
| Own site footage | Pi HQ camera | Own | Own | **Final test set, always** |

**Rules for combining datasets:**
1. **One class map** (drone 0, bird 1, airplane 2, helicopter 3). Datasets that label *only drones* still contain birds and aircraft, **unlabelled**. Never take background examples from them, or the model learns birds as "nothing" (the Halmstad label-tail lesson, [14 §5](14_build_log_and_experiments.md#5-findings-so-far)).
   * **Refinement (E12, 2026-09-25):** positive-only tiles have a cost. E12 never saw Anti-UAV's clutter as "nothing" and called lamps, buildings, cranes and pylons "drone" (precision 0.78 on its test). For a single-target dataset, a region with no drone and no bird-like object *is* a safe drone-negative.
   * The safe way to add such regions is **hard-negative mining**: take the model's false alarms on *training* sequences, check a sample by eye, and drop anything that could be an unlabelled drone or bird (E14, [14 §4](14_build_log_and_experiments.md#4-experiments)).
2. **Split by video within each dataset**, and tag every image with its source dataset and video, so results can be reported per dataset and bootstrapped by video.
3. **Report per dataset, and hold one whole dataset out** as a cross-dataset test. "Trained on A+B, tested on unseen C" is the honest generalisation measure, and a publishable result in itself.
4. **Tile high-resolution frames, never downscale** (Anti-UAV's Full HD frames tile into 640-px crops at true pixel size).
5. **Balance sampling** so the largest dataset doesn't dominate. Record each dataset's licence (Drone-vs-Bird: research only, no redistribution).
6. Label styles differ (box tightness, one-target-per-frame). The IoU 0.3 matching in `cuas/eval.py` tolerates loose boxes. Still compare *within* a dataset first.

### 2.2 Dataset Sweep (2026-10-10)

Drone-vs-Bird videos and LRDD are not obtainable for now, so this is a wide search for replacements (web search, the GitHub, Zenodo, Hugging Face and figshare APIs). "Checked" means the repository or record was read on 2026-10-10; **no file was downloaded, so the Google Drive, OneDrive and Baidu links are not confirmed live.** Nothing here is in the training data yet.

**Purpose (user, 2026-10-10): the project identifies drones; every other class is there only so it is not mistaken for a drone.** Datasets are judged by that. No dataset is taken in to recognise people, vehicles or buildings, and none to identify crewed aircraft as a goal.

**Open and worth getting**

| Dataset | What it is | Access and licence (checked) | Use here |
| :--- | :--- | :--- | :--- |
| **Multi-view drone tracking datasets** (Li et al., IROS 2020, [arXiv:2003.04784](https://arxiv.org/abs/2003.04784)) | Five outdoor flights of a hexacopter (2 to 10 min), each filmed by 4 to 7 consumer cameras (phones, compacts, GoPro); **3D trajectory from RTK** (dataset 5: total station + IMU, three drones); 2D labels for datasets 1-4; camera locations for 3 and 5; calibration for every camera | [repo](https://github.com/CenekAlbl/drone-tracking-datasets), videos in the repo as split zips; CC BY-NC-SA 4.0 | **The replacement for Drone-vs-Bird as the unseen video set, with range truth:** real `w_min` by distance, TTC and track decision on video never trained on. One drone model, one site per flight |
| **UAVDB** ([arXiv:2409.06490](https://arxiv.org/abs/2409.06490)) | Boxes (and SAM2 masks) for the set above, made from the trajectory points: 10,763 / 2,720 / 4,578 images | [Zenodo 14534940](https://zenodo.org/records/14534940) (6.5 GB, CC BY 4.0; a newer version exists), [repo](https://github.com/wish44165/UAVDB) | Ready labels for the same videos. Its boxes are machine-made: check a sample by eye |
| **DUT Anti-UAV, tracking subset** | 20 videos, visible light, manual boxes | Same [repo](https://github.com/wangdongdut/DUT-Anti-UAV) as the detection set already used (Google Drive / Baidu; Apache-2.0) | Video in DUT-like scenes for E23 and the track decision. Same dataset family as DUT test: not a second unseen set |
| **MMAUD** (ICRA 2024, [arXiv:2402.03706](https://arxiv.org/abs/2402.03706)) | Rooftop rig: two synchronised cameras, two lidars, radar, audio; DJI Mavic 2, Mavic 3, Phantom 4, Avata, M300; **position truth from a Leica total station at 5 Hz**; flights within about 30 m (V1) and 100 m (V2) | [repo](https://github.com/ntu-aris/MMAUD), OneDrive, 11-20 GB per sequence as rosbag or folder; MIT | Five named drone models with range truth at short range. Needs a rosbag or folder reader |
| **Real World Object Detection Dataset for Quadcopter UAV** (Pawełczyk and Wojtyra, IEEE Access 2020) | 51,446 train + 5,375 test images, 640×480, frames from a listed set of YouTube videos: many scenes, drone types, times of day | [repo](https://github.com/Maciullo/DroneDetectionDataset) (Google Drive mirror, about 5 GB; labels MIT), [Hugging Face copy](https://huggingface.co/datasets/pathikg/drone-detection-dataset) | **Scene variety, the measured weakness** ([15 §6.19](15_detector_training_methods_and_findings.md#619-e24-and-e27-a-frozen-backbone-and-an-objects365-start-dut-by-background-2026-10-06)). Low resolution; most labels were proposed by a detector and corrected |
| **AirBirds** (ACCV 2022, [arXiv:2304.11662](https://arxiv.org/abs/2304.11662)) | 118,312 frames at 1920×1080 from airport cameras, 409,967 bird boxes, most under 10 px | [Hugging Face](https://huggingface.co/datasets/auniquesun/AirBirds), YOLO labels; CC BY-NC-SA 4.0 | The thin "not drone" side: small real birds on sky, at true pixel size |
| **FBD-SV-2024** ([arXiv:2409.00317](https://arxiv.org/abs/2409.00317)) | 483 surveillance **video clips**, 28,694 frames, 28,366 flying birds | [figshare](https://figshare.com/articles/dataset/FBD-SV-2024/27377595) (1.79 GB of video, CC BY 4.0), [scripts](https://github.com/Ziwei89/FBD-SV-2024_github) | Bird tracks for the track classifier (video, so wing-flap and kinematics features apply) |
| **MAV-VID** | 64 videos of single drones from ground cameras, phones and other drones (about 40,000 frames) | [Bitbucket](https://bitbucket.org/alejodosr/mav-vid-dataset) per the [benchmark repo](https://github.com/KostadinovShalon/UAVDetectionTrackingBenchmark) (the Kaggle page is gone) | More drone video; mixed viewpoints, large targets |
| **UAVSwarm** | 72 sequences, 12,598 images, 3 to 23 drones per frame, 13 scenes | [repo](https://github.com/UAVSwarm/UAVSwarm-dataset) (no licence file) | Several drones at once, which no current set has |

**Use with care**

* **Seraphim** ([Hugging Face](https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset), CC BY 4.0): 83,483 images merged from 23 Kaggle and Roboflow sets, one class. Every image is resized and padded to 640×640 (small targets are no longer at true pixel size), and its card says it mixes real photos with marketing pictures and synthetic images. **Check its source list for DUT or Anti-UAV images before any use**, or the unseen test is contaminated.
* **Synthetic:** Cranfield synthetic drone set ([Hugging Face](https://huggingface.co/datasets/mazqtpopx/cranfield-synthetic-drone-detection), CC BY 4.0, [arXiv:2411.09077](https://arxiv.org/abs/2411.09077)); SynDroneVision ([Zenodo 13360116](https://zenodo.org/records/13360116): that record holds labels only). Low priority: simulation is for testing here, not training pixels ([19 §6](19_design_review_parameters_and_novel_directions.md)).
* **Air-to-air (moving camera, another problem):** ARD100 (100 videos, 202,467 frames, tiny drones; [repo](https://github.com/Irisky123/YOLOMG), **Baidu only**), NPS-Drones, FL-Drones, ARD-MAV, Det-Fly.
* **RGB + event:** GAREUD ([repo](https://github.com/coconut-xia/GAREUD)), SkyEV, MM-UAV. RGB halves usable; not read in detail.

**Found but gated, unreleased or unverified**

* LRDD v1-v3: access form, on hold (§3). DroneSOD-30K ([arXiv:2603.25218](https://arxiv.org/abs/2603.25218)): the [repo](https://github.com/shiner-chen/sdd-yolo) holds code only. VisioDECT (20,924 images, six drone models; said to be on IEEE DataPort, Zenodo and Kaggle), USC Drone, TIB-Net (2,850 images of far drones, [repo](https://github.com/kyn0v/TIB-Net)), RCSD-UAV, UAV-CB, BirdRecorder (kites near wind turbines, [arXiv:2508.18136](https://arxiv.org/abs/2508.18136)), the Danish and Japanese wind-farm bird videos: named in papers, download not confirmed.
* The 4th Anti-UAV Challenge release on Zenodo (15103888) is restricted, and thermal.

**Left out**

* TartanAviation (airport cameras with ADS-B type and range labels, 3.3 TB), KI4Flight (light aircraft with distances), the Airborne Object Tracking set (aircraft encounters filmed from an aircraft, over 11 TB): aircraft-identification sets. Halmstad already supplies airplane and helicopter examples as "not drone"; these would add size, not drone accuracy.
* Any set whose labels are people, vehicles or buildings.

### 2.3 Pulled Datasets: Sizing and Caveats (2026-10-10)

All under `D:\cuas_data\datasets\<folder>\` (C: has little room; pull log in `_logs\`). **Status 2026-10-10: seven sets are on disk (about 108 GB); MAV-VID and MMAUD are not.** Frame sizes and box widths marked *measured* were read from the files (labels and image headers, archives not unpacked); the rest is from each source's paper or card. Widths are box widths in native pixels: 5th / 50th / 95th percentile.

| Folder | Source | Native frame size | Target size (as stated) | Labels | Caveats that change sizing or scoring |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `multiview_drone_tracking` | [repo](https://github.com/CenekAlbl/drone-tracking-datasets), 35 GB clone; videos are split zip archives that need 7-Zip (not installed here) | Per camera, *measured* through UAVDB: 1920×1080 (17 cameras), 3840×2160 (2), 1440×1080 (2); frame rates 25 to 59.94 fps and **not synchronised** | Not measurable from its own labels (points) | One drone position per frame and camera (x, y; datasets 1-4); RTK 3D track; calibration per camera; camera locations for datasets 3 and 5 only | **Labels are points, not boxes: pixel width must come from range and that camera's calibration, never from a box.** Each camera has its own focal length: never pool pixel sizes across cameras. Range needs the camera position (given for 3 and 5; to be solved for 1, 2, 4). One hexacopter (dataset 5: three drones, snow). Rolling shutter. CC BY-NC-SA |
| `uavdb` | [Zenodo 14534940](https://zenodo.org/records/14534940), 6.5 GB, one zip | Same videos, every 10th frame; 18,061 images over 21 cameras | *Measured:* 18 / 22 / 56 px; 64% in 16-32 px, 33% in 32-64 px, **none under 18 px** | YOLO, one drone in every image, no empty images | **Box widths are quantised (18, 22, 28, 32, 42, 52 ... px) with a floor of 18 px: the box-growing method cannot return a smaller box, so a 6 px drone is also labelled 18 px. Never bin by these widths, never feed them to TTC or `w_min`.** Mean IoU 0.46 against human labels (its paper): score by centre distance. Its train / val / test split is by camera inside the same flights (d1c0 train, d1c1 val, d1c3 test ...): use the whole set as one test here. The two 4K cameras (d3c2, d4c3) must be tiled, not shrunk |
| `dut_tracking` | [repo](https://github.com/wangdongdut/DUT-Anti-UAV) (Google Drive), 3.6 GB | *Measured:* 15 videos 1920×1080, 5 videos 1280×720; 24,804 frames, 83 to 2,635 per video | *Measured:* 48 / 68 / 179 px (quartiles 48 and 113): **large targets, almost none under 32 px** | One box per frame as x, y, w, h | **Absent target is written as a negative box (w = −100): 2,586 of the 24,804 frames, in 9 videos (*measured*); drop them before any statistic, and count them as "no drone" frames only after a look.** Not a small-target set: says nothing about `w_min`. Same family as DUT test and E39's tiles: check for shared footage before calling anything unseen. A second drone or a bird in frame is unlabelled |
| `quadcopter_youtube` | [Hugging Face copy](https://huggingface.co/datasets/pathikg/drone-detection-dataset) of [the original](https://github.com/Maciullo/DroneDetectionDataset), 4.7 GB of parquet | *Measured:* **all 54,071 images are 640×480, downscaled from the source videos** | *Measured:* 12 / 46 / 358 px; 10% under 16 px, 21% over 128 px | COCO boxes; 51,446 train + 2,625 test in this copy (the original test set has 5,375); 1 image has no box, 1,255 have two or more | **Pixel sizes are not native: keep it out of every `w_min` and size-binned score.** Do not tile or upscale; pad 640×480 to 640×640. A fifth of the drones are close-ups over 128 px (product-style shots), far from our use: consider a size cap. Many frames per video: split by video, but this copy has only running image ids, so the video of each frame must come from the original file names. Most labels were proposed by a detector and corrected. Burned-in text is likely (E36) |
| `airbirds` | [Hugging Face](https://huggingface.co/datasets/auniquesun/AirBirds); **partial pull: all 12 label zips, `images9.zip` (8,312 images) and one test zip, 27 GB of 176 GB** | *Measured:* 1920×1080 PNG | *Measured over all 118,312 label files:* 3 / 5 / 20 px; **77% under 8 px**, 16% in 8-16 px | YOLO, one class (bird); 29,063 label files are empty | **Most birds are below our own `w_min` (8 px): as training targets they teach 3-5 px dots as "bird", and the same dots are what a far drone looks like. Use them first as hard negatives for the drone class and for testing false drones; decide separately whether sub-8 px birds are trained as a class.** One airport, fixed cameras (file names carry date and camera): sample sparsely, split by camera and day. Aircraft are in view and unlabelled: an empty label file is "no bird", not "no object". Non-commercial |
| `fbd_sv_2024` | [figshare](https://figshare.com/articles/dataset/FBD-SV-2024/27377595), 1.8 GB | 1280×720, 25 fps (paper) | *Measured (2026-10-11, the per-frame label set):* 13 / 34 / 78 px; 9% in 8-16 px, 37% in 16-32 px, 41% in 32-64 px, none under 8 px (the labels are `.rar` archives inside the zip; Windows `tar` opens them) | Per-frame boxes with track ids and a difficulty level | Clips are short (about 2.4 s on average): one or two track-classifier windows each. Random split by clip, so scenes repeat across its train and test. 720p surveillance compression |
| `mav_vid` (**not pulled:** four clones failed at the same point, "pack has bad object at offset 611"; the archive link and the file API refuse anonymous access) | [Bitbucket](https://bitbucket.org/alejodosr/mav-vid-dataset), 11.5 GB | Measure (mixed sources) | Large for the most part; measure | YOLO boxes, one drone per frame, train / val folders | Mixed viewpoints (ground, handheld, from another drone): tag each video, use the ground-camera ones. Internet footage |
| `uavswarm` | [repo](https://github.com/UAVSwarm/UAVSwarm-dataset), 1.8 GB | *Measured:* **mixed and odd sizes (640×352, 812×428, 1279×606, 1919×1079, 1919×870 ...): screen captures or resized video, not camera frames** | *Measured (train):* 13 / 21 / 67 px, 63% in 16-32 px | COCO-style json, 36 train videos with 6,844 images and 49,661 boxes (median 6 drones per image); **the test json (5,754 images) has no boxes** | Not native pixels for most videos: keep out of size-binned scores. Only the train half is usable. No licence file. Partly internet video of drone shows: check for night scenes and light-show drones (out of scope) |
| `mmaud` | **Not pulled: the OneDrive links refuse scripted downloads** | Two synchronised cameras; measure | Drone within about 30 m (V1) or 100 m (V2): large | Leica position truth at 5 Hz | To get in a browser (list below). CC BY-NC-SA per its page, MIT on the repo: treat as non-commercial |

**Multi-view set, distances (dataset 3, 2026-10-11).** Camera-to-drone distance from its camera positions and RTK track, **assuming both files share one coordinate frame (not verified by reprojection)**: median 40 to 88 m per camera, 5th to 95th percentile 18 to 138 m over the six cameras. With each camera's calibrated focal length that is about 17 to 23 px per metre of drone at the median distance on the 1080p cameras (38 on the 4K one). **So this is a short-range set: the hexacopter is roughly 10 to 40 px wide, and it does not test distant, few-pixel drones.** It gives range truth and unseen video; long range still needs zoom-out mosaics or own footage. Datasets 1, 2 and 4 have no camera positions.

**Unpacking (2026-10-11).** The multi-view videos are split zips; `python tools/unpack_split_zip.py <repo> <out>` unpacks them without 7-Zip (each file's CRC checked; it reads the zip directory itself because `zipfile` refuses zip64 archives that span parts). **All 27 videos unpacked (33.9 GB) and open in OpenCV:** datasets 1-2 about 3 min per camera, 3-4 about 9-11 min, 5 about 9-16 min; 21 to 59.94 fps; one row per frame in each label file of datasets 1-4 (dataset 5 has no 2D labels). **Of 310,737 frame rows, 180,317 carry a drone position (dataset 1: 60%, 2: 82%, 3: 63%, 4: 49%); the rest are `0 0`, which its README defines as "not distinguishable from the background", so the drone may still be in the picture: those frames are neither positives nor drone-free negatives.** Some label files start with a header line. Videos go to `multiview_drone_tracking/videos/`, the other sets to `<folder>/unpacked/`; the archives are kept. The two `.git` folders of the clones were removed (34 GB; source commits are in `SOURCE_COMMIT.txt`).

**Rules these add to §2.1:** (7) record the native frame size and whether a set was resized before it reaches us; a resized set never enters a pixel-size result. (8) Machine-made boxes are scored by centre distance. (9) A set that labels one target per frame, or one class only, never supplies background examples.

**Places a person has to look (login, browser or an email):**

* **MMAUD:** open [its page](https://ntu-aris.github.io/MMAUD/) in a browser and download one "Folder Data" zip (DJI Mavic3, 11.1 GB, is the smallest) and its ground-truth bag into `D:\cuas_data\datasets\mmaud\`.
* **MAV-VID:** the Bitbucket repository will not clone with Git for Windows here (same error every time, probably a file over 4 GB). Try the repository's Downloads or Source page in a browser, or `git clone` inside WSL, into `D:\cuas_data\datasets\mav_vid\`. Lower priority than the others.
* **Roboflow Universe and Kaggle:** many small drone and bird sets of uneven quality; note the source and licence of each, and whether it already contains DUT or Anti-UAV images.
* **ARD100:** Baidu account only ([repo](https://github.com/Irisky123/YOLOMG)).
* **VisioDECT** (20,924 images, six drone models): said to be on IEEE DataPort, Zenodo and Kaggle; download not confirmed.
* **Emails:** the TIB-Net authors (2,850 images of far drones), the wind-farm bird video groups, and Drexel again for LRDD.
* **The rest of AirBirds** (149 GB more) only if the first part proves useful.
* **Own footage** ([20](20_field_sessions_own_footage.md)): still the best addition; the multi-view set is a template for filming with position truth.

---

### 2.4 Test-Footage Sweep: Night and Trees (2026-10-22)

Searched for the field test (14 §5, 16 #93) after the user put night and tree backgrounds in scope. The full catalogue, with the sets that need a login or form and the dead ends (thermal-only, synthetic, cameras on drones), is `D:\cuas_data\datasets\_search_2026-10-22.md`; each folder has a `SOURCE.txt`. About 85 GB, all on `D:\cuas_data\datasets\`. Nothing here is in any training set: keep it that way until a decision says otherwise.

| Folder | What | Conditions | Labels | Caveats |
| :--- | :--- | :--- | :--- | :--- |
| `gareud` | GAREUD-Real, 15 ground-to-air recordings (RGB part), 137,000 frames, 1024×576, 30 fps, CC BY-NC 4.0 | **Night 10-100 lux** (R_026, R_027), **dusk** with street lamps (R_003, R_021-R_025), bare trees (R_002, R_003), reeds and woodland, snow, glare | `labels.txt` `ts cls xc yc w h`; a timestamp alone = no drone | Drones mostly 20-100 px; night frames very noisy. Field-test set `gareud` |
| `fred` | FRED, whole test split (47) + dusk train sequences 30-33, 35, 36; 1280×720, 30 fps, Apache-2.0 | Dusk (by clock time in frame names), courtyards with trees and buildings, multi-drone | `coordinates_rgb.txt` `t: x1, y1, x2, y2, id, model` | Close range, near-horizontal views; test 21 is black (no-signal check). Same lab as NeRDD. Field-test set `fred` |
| `nerdd` | NeRDD, 18 sequences with trees behind the drone (15 GB of a 176 GB zip), 1280×720 | Daytime, foliage backgrounds | `ev_rgb_coordinates.txt` | No loader yet |
| `uav_cb` | UAV-CB (CVPR 2026), 3,393 visible stills (+ thermal), CC BY-NC 4.0 | Vegetation 889, building 1,191, cloud 590, powerline 555 | LabelMe JSON | Still images: a detector test, not a field test |
| `usc_drone` | USC/MCL drone, 9 campus sequences (19,138 frames, 15 fps) + 25 internet videos | Trees, buildings | `groundtruth_rect.txt`, sparse | 9 of the 30 clips its card lists |
| `gimbaled_uav` | Gimbaled UAV tracking, 2448×2048 on a moving gimbal + RTK, CC0 | Sky, horizon trees, range truth | Tracker outputs, no hand boxes | Download was still running on 2026-10-22 |

## 3. Dataset Notes

### Anti-UAV and Anti-UAV410
* Anti-UAV pairs RGB with thermal video. Anti-UAV410 and CST Anti-UAV are **thermal only**.
* Each sequence has a JSON label file with two parallel lists: `exist` (per-frame visibility flag) and `gt_rect` (`[x, y, w, h]` per frame).
* The official score is the challenge's own accuracy: mean IoU on frames where the target is visible, plus credit for correctly reporting "absent" when it is not (see the Anti-UAV README). Anti-UAV410 tooling reports **SA (state accuracy)**. Neither uses HOTA.

### Drone-vs-Bird
* Targets are often at the limit of resolution, with birds as the dominant confuser. That makes it the closest public proxy for Case 1's false-alarm problem.
* Annotations are public on GitHub. The videos are distributed by the challenge organisers (check the repo for current terms).
* **Checked 2026-09-30** (the organisers' reply to our request, plus the 77 annotation files parsed with `cuas.datasets.dvb`):
  * **Only the training set is given.** The challenge test videos and their annotations are withheld. We split the training videos by video into our own train/val/test. Our numbers are therefore **not comparable** with published challenge results, which use the hidden test set.
  * The training annotations are the public GitHub files, not part of the email delivery. Copy in `C:\cuas_data\dvb\annotations`.
  * 77 videos, 106,485 frames (175-4,612 per video, median 925), 119,259 boxes. Frames start at 0; every frame has a line (9,086 with no object).
  * **Only drones are annotated.** Birds are unlabelled distractors ("parrot" in file names is the Parrot drone brand).
  * **Real small targets:** widths 4-1,403 px, median 25. 29% of boxes are under 16 px, 62% under 32 px. This is the real-footage check that the zoom-out mosaics (optimistic) could not give.

### LRDDv2 / LRDDv3 (found 2026-10-01; **access on hold**)
* **Why it matters here:** it is the only set found with **real small drones, range truth and birds in the same scenes**. That would cover three open questions: `w_min` on real distant drones (E19 is waiting on the Drone-vs-Bird videos), a check of time to collision and range against truth (19 §2 #16), and drones and birds filmed together for the track classifier's confound test.
* **Availability (both project pages, checked 2026-10-01):** "Dataset access is currently on hold while we coordinate dataset license and access requirements." The pages name the CDLA licence, no redistribution, and access "based on the US export regulations". The LRDDv3 paper says CC BY 4.0; the page is the newer statement. **Action: ask the Drexel iMaPLe lab by email, as was done for Drone-vs-Bird, stating academic use in Canada.**
* **From the LRDDv3 paper** (read through an automated summary; verify on receipt): 3840×2160, 5 FPS, DJI Mini 3 and Autel EVO II, most images at 0-50 m, boxes down to 12 px of area, 93,652 drone / 6,031 bird / 722 airplane boxes, 20,000 test images from 34 clips. The page speaks of the range "between the target drone and the camera drone", so at least part of it is filmed from another drone, not from the ground.
* **Their baseline shows what resolution is worth:** YOLOv11m, mAP50 0.543 at 640 input against 0.822 at 1920. Our tiles keep full resolution.
* **Unverified:** the annotation format, the size, whether clips can be split by video, and how much is from a fixed ground camera. 5 FPS is too slow for flap features but enough for size-against-range and for time to collision.

### RFUAV
* The Hugging Face repo holds **37 per-device `.rar` archives** of raw recordings. The devices are **6 drone systems** (DJI Mini 3, Mini 4 Pro, Mavic 3 Pro, Avata 2, FPV Combo; Autel EVO Nano) and **31 RC transmitters / ground-control links** (FlySky, FrSky, Futaba, JR Propo, Jumper, RadioMaster, Radiolink, SIYI, Skydroid, WFLY, Yunzhuo, Herelink, Devention).
* The repo also contains a pre-rendered spectrogram image set (`ImageSet-AllDrones-MatlabPipeline/`, ~19k JPEGs), model weights and a 5-drone validation set.
* Consequence: RFUAV mostly teaches a model to fingerprint the **control link**. It says nothing about drones flying radio-silent or on fibre (see [04](04_cuas_defense_systems_and_industry.md)).

### DroneRF
* Classes: background, 3 drone types, and flight modes (off, on and connected, hovering, flying, video recording).
* The paper's accuracies (99.7% / 84.5% / 46.8% for 2 / 4 / 10 classes) are the right baselines to beat.

### AERPAW Dataset-28
* Contains **position estimates** from Keysight N6841A RF sensors and a Fortem TrueView R20 radar, plus UAV ground truth. It does **not** contain raw RF or radar signals.
* Good for testing track-level fusion and Kalman filters (see [01 §6.2](01_research_papers_compendium.md#62-fusion-of-cellular-isac-and-passive-rf-sensing-for-uav-detection-and-tracking)), not for training detectors.

---

## 4. Loading the Data

### Anti-UAV / Anti-UAV410 labels
```python
import json
from pathlib import Path

seq = Path("Anti-UAV410/test/20190925_101846_1_1")   # one sequence folder
labels = json.loads((seq / "IR_label.json").read_text())

boxes, visible = labels["gt_rect"], labels["exist"]   # parallel per-frame lists
assert len(boxes) == len(visible)
print(f"{len(boxes)} frames, target visible in {sum(visible)}")
print("first visible box [x, y, w, h]:", next((b for b, v in zip(boxes, visible) if v), None))
```
The folder name is an example. `IR_label.json` and its `gt_rect`/`exist` keys are what the official Anti-UAV410 evaluation code reads. For the RGB stream of the original Anti-UAV release, check the label file name in the release you download.

### RFUAV (one device at a time)
The dataset is not in a `datasets`-loadable format, so `load_dataset(..., streaming=True)` will not work. Download a single device archive instead:

```python
from huggingface_hub import hf_hub_download   # pip install huggingface_hub

path = hf_hub_download(repo_id="kitofrank/RFUAV", repo_type="dataset",
                       filename="DJI MINI3.rar")
print("downloaded to", path)   # extract with a RAR tool, then use the RFUAV repo's loaders
```
Each archive is large. Check free disk space, and pick devices that match your threat list before pulling the full ~1.3 TB.
