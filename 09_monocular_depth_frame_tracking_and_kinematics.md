# Monocular Depth Estimation, Frame Tracking & Kinematics (Position, Velocity, Acceleration)

This research document details the mathematical and algorithmic foundations for estimating the 3D spatial position, velocity, and acceleration vectors of an airborne drone from 2D monocular camera streams.

> [!NOTE]
> **Revised 2026-09-21.** The maths was checked numerically: the variance of the finite-difference acceleration, the process-noise matrix and the measurement Jacobian are all correct. Fixes: the Q matrix was mislabelled "DWSA" (it is the continuous white-noise-jerk / CWPA model); process-noise units are corrected; the depth-error example now states its lens assumption; sections on **size-prior bias** and **what is actually observable at range** were added.
>
> **Revised 2026-09-22.** Fixes:
> * The time-to-collision formula had a sign error.
> * "Six orders of magnitude" is replaced with the exact figure.
> * The IMM now includes its Markov switching and mixing steps.
> * The camera-frame state is flagged as fixed-camera-only.
> * The ByteTrack and OC-SORT descriptions are corrected.

---

## 1. The Monocular Depth Recovery Problem

A standard 2D camera performs a projective transformation from 3D Euclidean space $\mathbb{R}^3$ to the 2D image plane $\mathbb{R}^2$, discarding absolute depth $Z_c$. For an airborne drone, recovering metric depth requires exploiting physical constraints, geometric priors, or specialized foundation models.

```
       Camera Focal Center (0,0,0)
                 \
                  \  Projective Ray
                   \
              [Image Plane]  Pixel (u, v), Bounding Box Width w
                     \
                      \
                   [Target Drone in 3D Space] Position (X, Y, Z), Metric Width W
```

### 1.1 Method A: Geometric Bounding-Box Scale Prior (Pinhole Formulation)
If the object classification head recognizes the target platform class (e.g., DJI Mini $W \approx 0.24\,\text{m}$, DJI Mavic $W \approx 0.35\,\text{m}$, or generic tactical quadcopter $W \approx 0.40\,\text{m}$), depth $Z_c$ along the optical axis is derived from pinhole optics:

$$Z_c = \frac{f_x \cdot W_{\text{metric}}}{w_{\text{pixels}}}$$

where:
* $f_x$: Camera focal length in horizontal pixel units ($f_x = \frac{F_{\text{mm}} \cdot W_{\text{sensor\_px}}}{\text{SensorWidth}_{\text{mm}}}$).
* $W_{\text{metric}}$: Known or assumed physical width/diagonal of the drone airframe (meters).
* $w_{\text{pixels}}$: Predicted 2D bounding box width in pixels.

Once $Z_c$ is determined, 3D Euclidean camera coordinates are reconstructed via camera intrinsics:
$$X_c = \frac{(u - c_x) \cdot Z_c}{f_x}, \quad Y_c = \frac{(v - c_y) \cdot Z_c}{f_y}$$

#### Sensitivity & Error Propagation Analysis:
Differentiating $Z_c$ with respect to bounding box pixel measurement $w$:
$$\frac{d Z_c}{d w} = -\frac{f_x W_{\text{metric}}}{w^2} = -\frac{Z_c^2}{f_x W_{\text{metric}}}$$

$$\sigma_Z \approx \left( \frac{Z_c^2}{f_x W_{\text{metric}}} \right) \sigma_w$$

> [!WARNING]
> **Depth variance scales quadratically.** Depth uncertainty $\sigma_Z$ grows with the **square of the distance** ($Z_c^2$). For a 0.35 m drone and **1 px of box-width jitter**:
>
> | Lens (4K sensor, 3840 px wide) | Z = 20 m | Z = 165 m | Z = 200 m |
> | :--- | :--- | :--- | :--- |
> | 90° HFOV ($f_x$ ≈ 1,920 px) | 34 px wide, σ_Z ≈ 0.6 m | 4.1 px, **σ_Z ≈ 40 m** | 3.4 px, σ_Z ≈ 60 m |
> | $f_x$ ≈ 3,800 px (the lens implied by the previous text) | 67 px, σ_Z ≈ 0.3 m | 8.1 px, σ_Z ≈ 20 m | 6.7 px, σ_Z ≈ 30 m |
> | 30° HFOV ($f_x$ ≈ 7,166 px) | 125 px, σ_Z ≈ 0.2 m | 15 px, σ_Z ≈ 11 m | 12.5 px, σ_Z ≈ 16 m |
>
> *Mitigation*: filter box size in an EKF rather than trusting single frames. Filtering only averages **noise**, though. It cannot remove **bias** (next section).

#### Size-prior bias (not removable by filtering)
$Z_c$ is proportional to the *assumed* $W_{\text{metric}}$. If the classifier calls a 0.24 m DJI Mini a 0.35 m Mavic, every range estimate is **46% too far**, consistently. Box width also changes with the drone's yaw and bank angle and with motion blur. Treat pinhole range as a coarse estimate with a class-dependent **systematic** error. Use it to gate and prioritise, not as a precise measurement. A second sensor (radar, laser rangefinder, or a second camera for triangulation) is the only way to get metric range at distance.

---

### 1.2 Method B: Monocular Metric Depth Foundation Models
Modern foundation models offer zero-shot depth estimation directly from single frames:

1. **Depth Anything v2 (2024-2025)**: State-of-the-art relative and metric monocular depth estimation trained on massive synthetic and real datasets. Strong boundary delineation, but open-sky regions often produce arbitrary background depth values.
2. **Metric3D v2 (2024-2025)**: Solves the metric scale problem by training across diverse camera intrinsics, predicting metric distance in meters.
3. **UniDepth (CVPR 2024)**: Jointly predicts camera intrinsics and metric 3D scene point clouds without requiring known focal length.

*Research Takeaway*: Monocular depth foundation models excel in textured indoor or driving environments with ground planes. For high-angle sky backgrounds where pixels are uniform, **Method A (Geometric Prior) combined with an EKF** remains computationally lighter and more reliable. The same applies to the `yolo26*-depth` checkpoints: they are trained on **NYU Depth V2 (indoor scenes)**, so they give no dependable range for a few-pixel drone against sky.

---

### 1.3 Method C: Looming Cues & Optical Expansion (Time-to-Collision)
When estimating incoming strike velocity without knowing metric scale $W_{\text{metric}}$, use **optical expansion rate**:

$$\tau = -\frac{Z_c}{\dot{Z}_c} = \frac{s}{\dot{s}}$$

where $s(t) = \sqrt{w \cdot h}$ is the bounding box pixel scale, and $\dot{s}(t)$ is its temporal rate of expansion. The two forms are equal because $s \propto 1/Z_c$ implies $\dot{s}/s = -\dot{Z}_c/Z_c$. For an approaching target $\dot{Z}_c < 0$ and $\dot{s} > 0$, so $\tau > 0$. The time-to-collision (TTC) $\tau$ can be estimated directly from image pixels, even if true metric distance is unknown. **This is the most robust threat cue a single fixed camera has:** it does not depend on the size prior, so it is immune to the bias above. It assumes a roughly constant closing speed along the line of sight, and it is noisy while the box is only a few pixels wide.

---

**As implemented and measured (2026-10-01, `cuas/track_features.py`):** for a constant closing speed, $1/w$ is a straight line in time that crosses zero at the moment of impact, so $\tau$ is read from a least-squares line through $1/w$ over the last 2 s, with no lag correction. It is reported only when the slope is 4 standard errors from zero and the boxes span 0.5 s: detector box widths jitter 3-6%, and without those tests 75% of non-closing tracks read $\tau < 10$ s. Two limits remain: $\tau$ reads long while the target is a few pixels wide (blur adds a constant to the width), and on real tracks of distant aircraft and birds it still reads under 3 s once or twice a minute, because box-size error is correlated in time ([19 §2](19_design_review_parameters_and_novel_directions.md#2-code-review-findings-closer-evaluation-fixes)).

## 2. Frame Identification & Multi-Object Spatio-Temporal Tracking

Detecting a drone in isolated frames is insufficient for kinematic estimation. The pipeline requires a **Multi-Object Tracker (MOT)** to link detections across frames into continuous trajectories.

```
Frame (t-1)                     Frame (t)                      Frame (t+1)
+-------------------+           +-------------------+           +-------------------+
|  Det #1 [Track 7] | --------> |  Det #1 [Track 7] | --------> |  Det #1 [Track 7] |
+-------------------+           +-------------------+           +-------------------+
                                         |
                                         v
                         Kalman Prediction & Hungarian Matching
```

### 2.1 State-of-the-Art Tracking Algorithms

1. **ByteTrack**:
   * *Mechanism*: Instead of discarding low-confidence detections (which standard SORT does), ByteTrack uses a two-stage association strategy. High-score boxes are matched first; then unmatched tracks are matched against low-score boxes.
   * *Why it matters for C-UAS*: Distant or occluded drones frequently flicker between low ($0.2$) and high ($0.7$) detector confidence. Keeping low-score boxes in the second matching stage preserves track continuity through motion blur and cloud crossings, with fewer identity switches than score-thresholded SORT.
2. **BoT-SORT (Robust Multi-Object Tracking)**:
   * *Mechanism*: Combines motion cues, Deep Re-ID appearance embeddings, and **Camera Motion Compensation (CMC)** via optical flow / ORB feature matching.
   * *Suitability*: Essential when the camera is mounted on a moving PTZ or vehicle mast where camera vibration would otherwise corrupt target velocity estimates.
3. **OC-SORT (Observation-Centric SORT)**:
   * *Mechanism*: Keeps the Kalman filter but corrects it with observations. After an occlusion, it re-fits the lost stretch of the track along a virtual trajectory between the last and the new observation (observation-centric re-update). It also adds a motion-direction consistency term, computed from observations, to the association cost. Designed for non-linear motion and occlusion, which suits manoeuvring drones.

---

## 3. Kinematic State Estimation: Position, Velocity & Acceleration

### 3.1 The Double-Differentiation Fallacy
Never compute acceleration by taking finite differences of noisy position predictions:

$$v_k \approx \frac{p_k - p_{k-1}}{\Delta t}, \quad a_k \approx \frac{p_k - 2p_{k-1} + p_{k-2}}{\Delta t^2}$$

Because pixel detection noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is present in each frame:
$$\text{Var}(a_k) \approx \frac{6 \sigma^2}{\Delta t^4}$$

At $30\,\text{FPS}$ ($\Delta t = 0.0333\,\text{s}$), $6/\Delta t^4 \approx 4.9 \times 10^6$. So **1 px of detector jitter becomes ~2,200 px/s² of acceleration noise** (standard deviation). That swamps any real manoeuvre, which is why acceleration must come from a filter with a motion model, not from differencing.

### 3.2 The Extended Kalman Filter (EKF) with a Discretised Motion Model
Define the 9-dimensional state vector $\mathbf{x}_k \in \mathbb{R}^9$:

$$\mathbf{x}_k = \begin{bmatrix} X & Y & Z & \dot{X} & \dot{Y} & \dot{Z} & \ddot{X} & \ddot{Y} & \ddot{Z} \end{bmatrix}^T$$

#### Nearly-Constant-Acceleration Transition Model (continuous white-noise jerk, "CWPA" in Bar-Shalom et al.):
$$\mathbf{x}_k = \mathbf{F} \mathbf{x}_{k-1} + \mathbf{w}_{k-1}$$

$$\mathbf{F} = \begin{bmatrix} 
\mathbf{I}_3 & \Delta t \mathbf{I}_3 & \frac{1}{2} \Delta t^2 \mathbf{I}_3 \\ 
\mathbf{0}_3 & \mathbf{I}_3 & \Delta t \mathbf{I}_3 \\ 
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{I}_3 
\end{bmatrix}, \quad 
\mathbf{Q} = q \begin{bmatrix} 
\frac{\Delta t^5}{20} \mathbf{I}_3 & \frac{\Delta t^4}{8} \mathbf{I}_3 & \frac{\Delta t^3}{6} \mathbf{I}_3 \\ 
\frac{\Delta t^4}{8} \mathbf{I}_3 & \frac{\Delta t^3}{3} \mathbf{I}_3 & \frac{\Delta t^2}{2} \mathbf{I}_3 \\ 
\frac{\Delta t^3}{6} \mathbf{I}_3 & \frac{\Delta t^2}{2} \mathbf{I}_3 & \Delta t \mathbf{I}_3 
\end{bmatrix}$$

where $q$ is the power spectral density of the white-noise **jerk**, in $\text{m}^2/\text{s}^5$, reflecting target manoeuvrability. This $\mathbf{Q}$ is the exact discretisation of the continuous model (checked numerically with Van Loan's method on 2026-09-21). The *discrete* Wiener-process-acceleration alternative uses $\mathbf{Q} = \boldsymbol{\Gamma}\boldsymbol{\Gamma}^T\sigma^2$ with $\boldsymbol{\Gamma} = [\tfrac{1}{2}\Delta t^2,\ \Delta t,\ 1]^T$ per axis. Both are valid; don't mix their parameter values.

#### Non-Linear Measurement Function $h(\mathbf{x}_k)$:
The detector outputs 2D bounding box center and scale: $\mathbf{z}_k = [u, v, w_{\text{pixels}}, h_{\text{pixels}}]^T$:

$$h(\mathbf{x}_k) = \begin{bmatrix} 
f_x \frac{X_k}{Z_k} + c_x \\ 
f_y \frac{Y_k}{Z_k} + c_y \\ 
\frac{f_x W_{\text{metric}}}{Z_k} \\ 
\frac{f_y H_{\text{metric}}}{Z_k} 
\end{bmatrix}$$

#### Measurement Jacobian Matrix $\mathbf{H}_k = \left. \frac{\partial h}{\partial \mathbf{x}} \right|_{\mathbf{x}_{k|k-1}}$:
$$\mathbf{H}_k = \begin{bmatrix} 
\frac{f_x}{Z} & 0 & -\frac{f_x X}{Z^2} & 0 & 0 & 0 & 0 & 0 & 0 \\ 
0 & \frac{f_y}{Z} & -\frac{f_y Y}{Z^2} & 0 & 0 & 0 & 0 & 0 & 0 \\ 
0 & 0 & -\frac{f_x W_{\text{metric}}}{Z^2} & 0 & 0 & 0 & 0 & 0 & 0 \\ 
0 & 0 & -\frac{f_y H_{\text{metric}}}{Z^2} & 0 & 0 & 0 & 0 & 0 & 0 
\end{bmatrix}$$

This formulation keeps the state in the **camera frame**, which is only valid for a fixed camera. For a PTZ, keep the state in a fixed world frame (e.g. local east-north-up) and put the camera's current pan/tilt rotation inside $h(\mathbf{x})$. Otherwise every slew looks like target motion.

---

### 3.3 Interacting Multiple Model (IMM) Filter
Quadcopter drones alternate between two distinct flight regimes:
1. **Steady Cruise / Hover**: Low process noise, approximately Constant Velocity (CV).
2. **Evasive Break Maneuvers**: High-g turns ($>3g$), aggressive roll/pitch tilting, high process noise.

A single Kalman filter with fixed process noise $q$ will either be too sluggish to follow sudden dives (if $q$ is low) or produce noisy, oscillating velocity estimates during hover (if $q$ is high).

An **IMM Filter** runs two parallel Kalman filters simultaneously:
* **Model 1 ($\mathcal{M}_1$)**: Nearly-constant velocity. White-noise *acceleration* with PSD $q_1$ in $\text{m}^2/\text{s}^3$ (starting point: 0.5).
* **Model 2 ($\mathcal{M}_2$)**: Nearly-constant acceleration (the CWPA model above). White-noise *jerk* with PSD $q_2$ in $\text{m}^2/\text{s}^5$ (starting point: 25; tune on real tracks). A **Singer** model (exponentially correlated acceleration) is a third, different option.

Both models must share a state vector for IMM mixing. Run the CV model on the same 9-dimensional state, with its acceleration block pinned near zero (zero rows in $\mathbf{F}$ plus a small $\mathbf{Q}$ on acceleration), so mixed estimates and covariances stay the same size.

Each IMM cycle has three steps:

1. **Predict the model probabilities** with a Markov switching matrix $\Pi$, where $\pi_{ij}$ is the chance of switching from model $i$ to model $j$ per step (start with $\pi_{ii} \approx 0.95$, $0.98$):
   $$\mu_{k|k-1}^j = \sum_i \pi_{ij}\,\mu_{k-1}^i$$
2. **Mix** each filter's starting estimate and covariance, weighted by the conditional probabilities $\pi_{ij}\mu_{k-1}^i / \mu_{k|k-1}^j$. Then run both filters' predict and update steps.
3. **Update the model probabilities** from each filter's innovation $\mathbf{y}^j_k$ and its covariance $\mathbf{S}^j_k$:
   $$\mu_k^j = \frac{\mu_{k|k-1}^j \cdot \mathcal{N}(\mathbf{y}_k^j; \mathbf{0}, \mathbf{S}_k^j)}{\sum_i \mu_{k|k-1}^i \cdot \mathcal{N}(\mathbf{y}_k^i; \mathbf{0}, \mathbf{S}_k^i)}$$

$\Pi$ sets how fast the IMM reacts to a manoeuvre, so tune it alongside $q_1$ and $q_2$.

**Implemented (2026-09-29)** in image coordinates as `cuas.track.IMMFilter` (state x, y, vx, vy, ax, ay; CV + CA; `Tracker(motion="imm")`). For a fixed camera, pixels are bearing. The world-frame EKF above is still needed for a PTZ or a moving camera.

The final estimated position, velocity, and acceleration are the probability-weighted mixture of both models. That gives faster response to manoeuvres than a low-$q$ filter and smoother hover estimates than a high-$q$ filter. It is a trade-off to tune, not a free lunch.

### 3.4 What Is Actually Observable from One Camera
* **Bearing (azimuth/elevation) and its rates:** well observed, at sub-pixel precision after filtering. This is the camera's real strength.
* **Time-to-collision:** observable from expansion rate, independent of size priors (§1.3).
* **Range, and radial velocity/acceleration:** weakly observed and biased (§1.1). At 165 m on a wide 4K lens, 1 px of jitter is ±40 m of range, so a reported radial *acceleration* is filter output driven by the model, not a measurement. Downstream logic, including any decision layer ([12](12_typesafe_jev_system_one_c2_threat_engine.md)), should consume **bearing, bearing rate, TTC and the filter covariance**, not bare point estimates of range and acceleration.
