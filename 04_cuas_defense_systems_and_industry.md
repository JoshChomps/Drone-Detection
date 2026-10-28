# C-UAS Defense Systems & Industry Platforms (2020-2026)

This document provides an operational and architectural analysis of fielded military and commercial Counter-Unmanned Aircraft Systems (C-UAS), defense technology manufacturers, electronic warfare suites, kinetic interceptors, and battlefield lessons learned from modern conflict zones.

> [!NOTE]
> **Source status (2026-09-21).** Product descriptions come from vendor marketing and press coverage. None of them are independent performance data. The US procurement section was updated: the Joint Counter-small UAS Office (JCO) was **replaced by JIATF-401 in August 2025**. Specific figures that could not be confirmed are marked *(unverified)*.
>
> **Revised 2026-09-22.** Fixes:
> * Interceptor costs corrected: Patriot interceptors cost several million dollars, not $1M.
> * Frequency hopping is now distinguished from band switching.
> * D-Fend's "undisturbed" claim is labelled as a vendor claim, with its limits stated.

---

## Modern Military & Industrial C-UAS Platforms

### 1. Anduril Industries
*   **Flagship Technologies**:
    *   **Lattice OS**: AI Command and Control (C2) operating system. Ingests raw telemetry from radars, optical/infrared cameras, and RF sensors across wide geographical areas. Uses computer vision and sensor fusion models to correlate tracks, prioritize incoming threats, and automatically propose engagement solutions.
    *   **Anvil & Anvil-M**: Autonomous kinetic interceptor drones: quadcopters designed to ram hostile drones. The **Anvil-M** variant is described as carrying a munition for heavier targets *(speed and warhead details unverified)*.
    *   **Pulsar**: Software-defined electronic warfare (EW) suite using edge AI to rapidly identify, adapt to, and jam adversary frequency-hopping communication and video datalinks.
    *   **Sentry Towers**: Autonomous, solar-powered surveillance towers with onboard edge compute, 360-degree radar, and thermal PTZ cameras.

### 2. DroneShield
*   **Flagship Technologies**:
    *   **DroneSentry-C2**: Cloud and on-premise command-and-control platform powered by deep learning. Provides real-time automated threat classification and track aggregation.
    *   **DroneSentry**: Fully integrated stationary multi-sensor perimeter defense platform combining 3D radar, passive RF direction-finding (RfOne), and high-resolution optical/thermal cameras.
    *   **RfPatrol**: Ultra-lightweight body-worn passive RF detector used by dismounted infantry and special operations. Detects drone control uplinks and downlinks passively with zero detectable electromagnetic emission.
    *   **DroneGun Tactical & DroneGun MK4**: Directional soft-kill RF jamming rifles capable of simultaneously neutralizing GNSS (GPS, GLONASS, BeiDou, Galileo) and 433 MHz, 915 MHz, 2.4 GHz, and 5.8 GHz control frequencies.

### 3. Fortem Technologies
*   **Flagship Technologies**:
    *   **SkyDome Manager**: AI-driven digital airspace platform that tracks hundreds of aerial objects simultaneously, filtering out birds and authorized flights to highlight hostile incursions.
    *   **TrueView Radar (R20)**: Compact electronically scanned Ku-band radar for small, slow, low-RCS UAVs. An R20 is used in the public AERPAW RF+radar dataset ([03](03_datasets_and_benchmarks_catalog.md#aerpaw-dataset-28)) *(weight and other variants unverified)*.
    *   **DroneHunter F700**: Autonomous air-to-air drone hunter. Guided by TrueView radar and onboard visual AI to pursue hostile drones and fire a compressed gas net-gun with an integrated parachute, capturing the rogue drone mid-air without ground collateral damage.

### 4. Dedrone (acquired by Axon, 2024)
*   **Flagship Technologies**:
    *   **DedroneTracker.AI**: Multi-sensor fusion software. Vendor claims identification of drone make/model and pilot location *(unverified)*.
    *   **Dedrone RF sensors**: Passive RF sensors that scan ISM bands for drone control and video links.
    *   **DedroneDefender**: Lightweight handheld directional smart jammer designed for tactical civilian and military operations.

### 5. D-Fend Solutions
*   **Flagship Technology**:
    *   **EnforceAir**: Non-jamming, non-kinetic RF cyber-takeover system. Passively monitors and automatically identifies hostile drone protocols (e.g. DJI, Autel, proprietary telemetry). When authorized, it transmits protocol-level commands that take over the drone's datalink and land it in a designated zone. The vendor claims surrounding communications are unaffected. This only works against drones whose protocols it supports, and not at all against RF-silent or fibre-guided drones.

### 6. Rafael Advanced Defense Systems & Leonardo
*   **Rafael Drone Dome**: Military-grade, multi-layered air defense system. Employs 360-degree AESA radar, SIGINT/COMINT sensors, high-performance electro-optical sensors, and an optional High-Power Laser Directed Energy Weapon (DEW) effector to destroy drones at stand-off distances.
*   **Leonardo Falcon Shield**: Modular expeditionary C-UAS combining passive RF sensing, electro-optics and electronic attack *(specific operator deployments unverified)*.

---

## 🎖️ Defense Procurement & Joint Interagency Frameworks

### 1. US DoD: from the JCO to JIATF-401
*   **JIATF-401 (Joint Interagency Task Force 401)**: In August 2025 the Secretary of Defense directed the Army to stand up JIATF-401, which **replaces the Joint Counter-small UAS Office (JCO)**. It reports to the Deputy Secretary of Defense and centralises C-UAS requirements, test, acquisition and interagency coordination, with authority to approve up to $50M per initiative ([DefenseScoop](https://defensescoop.com/2025/08/28/hegseth-army-new-counter-drone-task-force-jiatf-401/), [IDGA explainer](https://www.idga.org/command-and-control/articles/jiatf-401-explained-how-the-pentagons-new-counter-uas-task-force-is-changing-drone-defense)). A 2026 C-sUAS quick-reference guide is published on [media.defense.gov](https://media.defense.gov/2026/Sep/08/2003992822/-1/-1/1/JOINT-INTERAGENCY-TASK-FORCE-401-2026-COUNTER-SMALL-UNMANNED-AIRCRAFT-SYSTEMS-QUICK-REFERENCE-GUIDE.PDF).
*   **FAAD C2 (Forward Area Air Defense Command and Control)**: The Army C2 system the JCO designated as the interim common C2 for counter-small-UAS. It fuses radars, optical sensors and effectors into a common operational picture.
*   **JCO demonstrations (2021-2025)**: The JCO ran periodic field demonstrations of low-collateral interceptors, high-power microwave and other defeat options. Their results fed JIATF-401 *(per-demonstration topics unverified)*.

### 2. NATO C-UAS Work
*   NATO runs counter-UAS working groups and technical interoperability exercises aimed at common sensor-to-effector interfaces across allies. *(The previous version's claim of a NATO "TRL 8/9 mandate for AI perception models" could not be verified and was removed.)*

---

## ⚔️ Battlefield Realities & Tactical Evolution (2020-2026)

The conflicts in Ukraine and the Middle East have permanently transformed the nature of drone defense, demonstrating that peace-time civilian counter-drone assumptions fail under high-intensity warfare.

```
+---------------------------------------------------------------------------------------------------+
|                            EVOLUTION OF DRONE COMBAT DYNAMICS                                     |
+------------------------------------+--------------------------------------------------------------+
| Phase / Era                        | Dominant Threats & Counter-UAS Tactics                       |
+------------------------------------+--------------------------------------------------------------+
| 2020 - 2022                        | Commercial Reconnaissance (DJI Mavic)                       |
| (Early War Period)                 | * Threat: Artillery spotting, grenade drops.                 |
|                                    | * Defense: RF detection (e.g. DJI AeroScope, since retired), |
|                                    |   handheld RF jammers, directional antenna jamming.          |
+------------------------------------+--------------------------------------------------------------+
| 2022 - 2024                        | Mass Loitering Munitions & FPV Emergence                     |
| (FPV & Shahed Revolution)          | * Threat: Shahed-136 delta-wings, Lancet loitering munitions,|
|                                    |   high-speed 7-inch and 10-inch kamikaze FPVs.               |
|                                    | * Defense: Wideband trench jammers, EW domes, mobile search- |
|                                    |   light anti-aircraft gun trucks (Gepard, Zu-23-2).          |
+------------------------------------+--------------------------------------------------------------+
| 2024 - 2026                        | Autonomous Terminal Guidance & Fiber-Optics                  |
| (Modern Contested Era)             | * Threat: Autonomous visual AI terminal lock, fiber-optic    |
|                                    |   tethered quads (immune to link jamming), swarms.           |
|                                    | * Defense: Distributed acoustic sensor networks, autonomous  |
|                                    |   kinetic interceptor drones, automated shotguns / CIWS,     |
|                                    |   high-power directed energy lasers & microwaves.            |
+------------------------------------+--------------------------------------------------------------+
```

### Key Tactical Insights for AI Defense Systems:

1.  **The Cost-Exchange Ratio Dilemma**:
    *   Surface-to-air interceptors cost from around a million dollars (NASAMS-class missiles) to several million (Patriot). Firing them at one-way attack drones costing tens of thousands of dollars (Shahed-class; public estimates vary widely), let alone ~$500 FPVs, exhausts both budgets and stockpiles.
    *   Modern C-UAS systems demand **asymmetric cost solutions**: automated guns, high-power microwaves, and interceptor drones reported to cost in the low thousands of dollars (drone-on-drone ramming).

2.  **The Collapse of RF-Only Jamming**:
    *   Hostile drones increasingly use **frequency hopping** within a band and **switch bands** (from a few hundred MHz up to 5.8 GHz) to escape fixed jammers. Some fly in complete **radio silence**, using inertial navigation and terrain or visual matching.
    *   **Fibre-optic-tethered drones** (unspooling ~10 km or more of fibre) carry control and video over the fibre, so there is no link to detect or jam.
    *   *Result*: Modern C-UAS AI must prioritize **Radar Micro-Doppler, Optical/Thermal Vision, and Acoustic sensing**, which cannot be defeated by RF silence.

3.  **Terminal AI Target Locking**:
    *   Press and open-source reporting describes FPVs and loitering munitions fitted with low-cost edge computers running lightweight detectors. Once the pilot points the drone at a target, the onboard model locks on and completes the terminal dive even if the link is jammed *(specific hardware unverified)*.
    *   *Defense Countermeasure*: Counter-UAS must achieve intercept *before* the drone enters its terminal dive phase, requiring long-range multi-sensor early warning networks.

4.  **Why this matters for this project**:
    *   Points 2 and 3 are the strongest argument for **passive EO/IR detection** (this project's focus): it works when the target is RF-silent, fibre-guided or autonomous.
    *   They also bound what a camera-only system can do. Warning time is set by **detection range**, and range is set by optics (see [13](13_system_requirements_and_geometry_budget.md)).
