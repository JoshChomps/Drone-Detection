# Adverse Weather Drone Detection, Datasets & Benchmarks (2024-2026)

Fog, rain, low light and snow are where optical counter-UAS breaks first. This file covers the newest vision benchmarks, the physics of image degradation (used to synthesise training data), and the metrics to report.

> [!NOTE]
> **Revised 2026-09-21.** Corrections: DroneGround is **fully synthetic** and released on request (it was described as removing the need for synthetic data); Anti-UAV410 is **thermal-only**; NeRDD is 3.5 h / 115 videos; the Anti-UAV challenge does **not** use HOTA; "UAV-CDT" could not be found and was removed. Full dataset details are in [03](03_datasets_and_benchmarks_catalog.md).

---

## 1. 2024-2026 Vision Benchmarks at a Glance

| Dataset | Year | Modality | Scale | Weather / conditions | Caveat |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DroneGround** | 2026 | Synthetic RGB (UE5 + Cosys-AirSim) | ~100k images | **5 weather conditions**, 7 environments, bird confounders | Synthetic; data on request only |
| **SimD3** | 2026 | Synthetic RGB (UE5, 360° six-camera rig) | Large-scale | Controlled weather and lighting, bird distractors | Synthetic |
| **CST Anti-UAV** | 2025 | Thermal | 220 seqs, 240k+ boxes | Complex scenes, **tiny** targets | Best tracker: 35.92% SA |
| **MM-UAV** | 2025 | RGB + thermal + event | 1,321 seqs, 2.8M+ frames | 30+ challenging scenarios incl. low light | Preprint |
| **NeRDD** | 2024 | Event + RGB | 3.5+ h, 115 videos | Fast motion, difficult illumination | Events fade on static targets |
| **Anti-UAV410** | 2023/24 | **Thermal only** | 410 videos, 438k+ boxes | Day/night thermal clutter | - |
| **Drone-vs-Bird** | 2017-2025 | Long-range RGB | Grows per edition | Real outdoor conditions; birds as confusers | Videos via organisers |

**Where real bad-weather data comes from:** none of the *real* datasets above is organised by weather class. Real fog/rain/snow footage at your site, plus physics-based synthesis (§3) and synthetic sets (DroneGround, SimD3), is the realistic path.

---

## 2. Benchmark Notes

### 2.1 DroneGround (2026)
* ~100,000 synthetic images: 15 drone platforms, 6 payload categories, 5 bird species, 7 virtual environments, 5 weather conditions. Built for **payload characterisation** (YOLO26s localisation + LoRA-tuned PaliGemma).
* Useful as **pre-training / augmentation** data for weather robustness. Always validate on real footage, because the paper itself frames synthetic-to-real shift as the core challenge.

### 2.2 NeRDD (2024) and why event cameras matter
* A frame camera integrates light over its exposure time. A drone crossing the field of view at speed smears across many pixels at 1/60 s.
* An event camera reports per-pixel log-brightness changes ($\Delta \ln I > \theta$) asynchronously at microsecond timing and with very high dynamic range. That suits fast motion and glare.
* The trade-off: **no events when nothing changes**. A hovering drone's body goes quiet against a static sky; only its spinning propellers keep flickering, and only while they span enough pixels to be resolved. NeRDD fuses both streams for exactly this reason.

### 2.3 Anti-UAV challenge series
* RGB + thermal tracking (Anti-UAV), then thermal-only (Anti-UAV410).
* Scored with the challenge's **own accuracy measure**: per-frame IoU when the target is visible, plus credit for correctly reporting absence, averaged over frames (see the [official README](https://github.com/ZhaoJ9014/Anti-UAV)). Anti-UAV410 tooling reports **state accuracy (SA)**. Report these when comparing to published numbers, and HOTA/IDF1 (§4.2) for multi-target work.

---

## 3. Physical Models of Weather Degradation

Used to synthesise degraded training frames before (or alongside) collecting real bad-weather data:

```
[Clear frame J(x)]
      |
      v
 FOG  (Koschmieder):   I(x) = J(x)·t(x) + A∞·(1 − t(x)),   t(x) = exp(−β·d(x))
      |
      v
 RAIN (streaks + mist): I(x) = I_fog(x) + Σ_k R_k(x)   (directional streak layers)
      |
      v
 LOW LIGHT (Poisson-Gaussian): I(x) = g·Poisson(J(x)/g) + N(0, σ_read²)
      |
      v
[Degraded training frame]
```

### 3.1 Fog (Koschmieder's law)
$$I(x) = J(x)\,t(x) + A_\infty\,(1 - t(x)), \qquad t(x) = e^{-\beta d(x)}$$
* $J$: clear-scene radiance; $A_\infty$: airlight (fog colour); $\beta$: extinction coefficient; $d$: distance.
* Meteorological visibility is $V \approx 3.912/\beta$. So $\beta = 0.05\,\text{m}^{-1}$ means $V \approx 78$ m (dense fog). A drone 100 m away then keeps $e^{-5} \approx 0.67\%$ of its contrast. In practice that is undetectable in RGB, which is the physical argument for **thermal** (longer wavelengths scatter less in haze, though thick fog still attenuates LWIR).

### 3.2 Rain
1. **Streaks**: semi-transparent linear streaks, modelled as directional motion-blur layers aligned with the wind.
2. **Mist**: suspended droplets add fog-like airlight (reuse §3.1).

### 3.3 Low light
Auto-gain raises the sensor gain $g$, amplifying shot noise:
$$I(x) \sim g\cdot\mathcal{P}\!\left(\frac{J(x)}{g}\right) + \mathcal{N}(0, \sigma_{\text{read}}^2)$$
A few-pixel drone sinks into this noise floor quickly. That is another reason to add thermal for night operation.

---

## 4. Evaluation Protocols & Metrics

### 4.1 Detection (COCO conventions)
* **mAP@50** and **mAP@50:95** (IoU 0.50 to 0.95 in steps of 0.05).
* **$AP_S$**: AP on boxes smaller than $32 \times 32$ px. **The most informative number for distant drones.** Also report recall at your operating threshold, binned by box size (e.g. <8, 8-16, 16-32 px), to connect results to [13](13_system_requirements_and_geometry_budget.md).

### 4.2 Tracking
* **HOTA**: $\text{HOTA}_\alpha = \sqrt{\text{DetA}_\alpha \cdot \text{AssA}_\alpha}$, balancing detection and association.
* **IDF1**: identity consistency over time.
* **Anti-UAV accuracy / SA**: see §2.3; needed for comparison with the Anti-UAV literature.

### 4.3 Kinematics
* **Position / velocity / acceleration RMSE** against ground truth (drone flight logs or RTK-GPS).
* **Normalised estimation error squared (NEES)**: checks whether the filter's covariance is honest. This matters more than RMSE whenever downstream logic uses the uncertainty.

### 4.4 System-level
* **Time to first stable track** (ms from first visible frame).
* **Detection range at fixed false-alarm rate**: the metric that actually sets warning time ([13 §3](13_system_requirements_and_geometry_budget.md#3-latency-budget)).
* **False alarms per hour** (bird season, not just clear days).
