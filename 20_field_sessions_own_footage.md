# 20: Field Sessions: Filming Our Own Drones (living plan)

> **Started 2026-10-05.** A standing plan for filming the user's drone-club drones, with their flight logs, from a fixed camera. Update the session log (§9) after every session. Daytime first; night is a later session type (§8, [11 row 12](11_research_deliberations_and_user_pushback_log.md#1-pushback--resolution-log)).

---

## 1. Why: What No Dataset Here Has

| Gap | Where it was found | What a session gives | Feeds |
| :--- | :--- | :--- | :--- |
| **Range and speed truth** | Time to collision has never been checked against a true range ([19 §2](19_design_review_parameters_and_novel_directions.md#2-code-review-findings-closer-evaluation-fixes) #16); box width reads +25% under 8 px ([19 §11.3](19_design_review_parameters_and_novel_directions.md#113-e35-the-scores-that-were-missing)) | The drone's logged position at every frame | TTC and range error; E32 with real flights |
| **Real distant drones** | `w_min` ≈ 8 px was measured on shrunken images ([13 §7](13_system_requirements_and_geometry_budget.md)) | The same drone at known distances down to where it vanishes | `w_min` on real targets (as E19) |
| **False declarations per hour** | E20 has 5.9 min of non-drone video, an upper bound of 30/h ([15 §6.9](15_detector_training_methods_and_findings.md#69-e20-the-system-on-whole-videos-2026-10-01)) | Hours of empty sky from the deployment-style camera | E20 / E26 on own footage; a threshold set on site |
| **An unscored test set** | The Anti-UAV test and DUT have been looked at after every run ([19 §11.1](19_design_review_parameters_and_novel_directions.md#111-the-validation-design)) | Footage no model has been tuned on | The final test; frees DUT's train split for training |
| **Varied terrain** | DUT transfer stays low; our two datasets' backgrounds are exhausted (16 #75) | Own skies, ridges, treelines | E22c-style pasting onto own backgrounds (train sessions only) |

---

## 2. Before the Session (checklist)

**Rules.** Fly under the club's rules and local regulations (in Canada: registration, pilot certificate, visual line of sight, distance from people). Nothing in this plan needs a drone flown at or over a person: every approach aims at a point well to the side of the camera.

**Tell the analysis side beforehand, if possible:**
* the camera (phone model or Pi HQ + lens), the video mode (resolution, frame rate);
* each drone: model, flight controller (DJI, PX4, ArduPilot, Betaflight), whether it has GPS, colour;
* so the distance ladder (§5) is computed for your exact camera and drones, and a log reader is ready.

**Bring and record:**
* [ ] Tripod; the camera must not move during a clip (re-shoot the sync and calibration steps if it is moved).
* [ ] Tape measure: each drone's **span** (tip to tip across the propellers, in metres) and the camera lens height above ground.
* [ ] Camera position: phone GPS at the tripod, read for a minute (note latitude, longitude, altitude).
* [ ] Two far landmarks visible in the frame (a peak, a mast) and their names: they fix the camera's heading against a map.
* [ ] A phone showing a GPS-synced clock with seconds (for time sync).
* [ ] Flight logging switched on for every drone, at the highest rate it offers.
* [ ] Optional: a printed checkerboard (calibration of the lens; 20 s of video tilting it in front of the camera).

---

## 3. Camera Setup (every clip)

* **Fixed focal length, no zoom.** Digital zoom changes the scale silently.
* **Stabilisation OFF**, HDR off, beauty or scene modes off. Phone stabilisation crops and warps the image, which breaks the geometry.
* **Focus locked at infinity**, exposure locked on the sky if the camera allows it.
* **Highest bitrate and resolution at 30 fps** (60 fps if it keeps the same resolution). Note what was used.
* **Tilted low:** the lower edge of the frame just above the skyline, as the deployed camera will be ([13 §2.4](13_system_requirements_and_geometry_budget.md)). Keep the sun out of the frame.
* **Never trim, re-encode or export** the originals. Copy them as they are (§7).

---

## 4. Shot List, Session Type A (daytime)

Each battery starts with **S0**. Times are flying time; a full session is about 60-90 min of flights plus at least 20 min of empty sky.

| ID | What | How | Length | Answers |
| :-: | :--- | :--- | :--- | :--- |
| **S0** | **Sync and calibration** | Start recording. Hold the GPS clock in front of the lens for 5 s. Take off **in view** (lift-off is visible in both the video and the log). Then hover 5 s each at the frame centre and near the four corners, at about the 16-px distance (§5) | 2 min per battery | Ties video time to log time; the corner hovers let the camera's heading, tilt and roll be solved from the log |
| **S1** | **Distance ladder (hover)** | Hover 10 s at each distance of §5 (30, 16, 12, 8, 6, 4 px), then further out until the drone is invisible on the camera's screen; come back in the same way. Keep it above the skyline in the frame | 10 min | `w_min` on a real drone; box-width error by size |
| **S2** | **Approach and recede** | Fly straight from the farthest visible distance towards a point **at least 20 m to the side** of the camera, at a steady 3, then 6, then 10 m/s; stop at a safe distance; fly back out the same way | 10 min | Time to collision and range against truth |
| **S3** | **Crossing passes** | Fly left to right and back across the frame at near, middle and far distance, steady speed | 10 min | Tracking on real motion; motion blur on small targets |
| **S4** | **Manoeuvres** | Hover, then dash; zig-zag; orbit; climb and descend; sharp turns | 10 min | The tracker's manoeuvre model; the track features that separate drones from birds |
| **S5** | **Backgrounds** | Repeat a slow crossing with the drone against clear sky, cloud, the ridge or treeline edge, and hazy sky towards (not including) the sun | 10 min | False alarms and misses at the skyline |
| **S6** | **Empty sky** | Camera as set; **no drone in the air** (ask the club); note birds and aircraft seen, with times | ≥ 20 min per session, aim for hours in total across days and weather | False declarations per hour; a threshold chosen on own footage |
| **S7** | **Two drones** (if available) | Two drones crossing each other's path at similar distance | 5 min | Whether tracks keep their identity |

Land in view at the end of each battery (a second sync point).

---

## 5. Distances for a Given Pixel Size

A drone of span *S* metres covers *w* pixels at range *R* = *f*ₚₓ · *S* / *w*, where *f*ₚₓ = image width in pixels / (2 · tan(horizontal field of view / 2)). Worked out for two cameras:

| Camera | *f*ₚₓ | Span 0.25 m (Mini-class) at 30 / 16 / 12 / 8 / 6 / 4 px | Span 0.35 m (Mavic-class) at 30 / 16 / 12 / 8 / 6 / 4 px |
| :--- | ---: | :--- | :--- |
| Phone, 4K video (3840 px), ~70° field of view | ~2,740 | 23 / 43 / 57 / 86 / 114 / 171 m | 32 / 60 / 80 / 120 / 160 / 240 m |
| Pi HQ + 16 mm lens, full width (4056 px, 1.55 µm pixels) | ~10,320 | 86 / 161 / 215 / 323 / 430 / 645 m | 120 / 226 / 301 / 452 / 602 / 903 m |

* A phone puts the interesting sizes (4-12 px) inside 250 m, which is easy to keep in sight. The Pi HQ with a 16 mm lens puts them at 300-900 m.
* **Use the measured span of your drone and your camera's field of view;** the phone figure of 70° varies by model. Send them and the exact ladder comes back.
* The logged position gives the true distance afterwards, so hovering "about" at each step is enough.

---

## 6. Clip Log (one line per clip)

| Clip file | Start (clock) | Drone | Shot (S0-S7) | Distance / speed planned | Weather, wind, light | Camera mode | Notes (birds, aircraft, anything odd) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | | | | | | | |

Also once per session: camera position and lens height, the two landmarks, each drone's span, which log file belongs to which flight.

---

## 7. After the Session

**The user:** copy everything, untouched, to `C:\cuas_data\field\<YYYY-MM-DD>\` with three folders: `video\`, `logs\` (flight logs as the drone or app saved them), `notes\` (the clip log, photos of the setup). Data stays out of the project folder.

**The analysis side:**
1. A reader for the flight-log format (PX4 exists: `ulog2csv` → `cuas.sim.from_px4_csv`; DJI, ArduPilot and others are written when the first log arrives).
2. Time sync from S0 (clock and lift-off), checked against the landing.
3. The camera's pose from the S0 corner hovers; the lens from the checkerboard or from the hovers.
4. Every frame's drone box from the log through the camera model, **checked by eye** on contact sheets (the lesson of 16 #69 and #76).
5. Then: `w_min` on real drones (S1), box-width bias (S1), time to collision and range error (S2), tracking (S3, S4, S7), skyline false alarms (S5), false declarations per hour (S6).

---

## 8. Data Rules and Later Sessions

* **Each session is marked test or train before anything is scored** (in §9), and test sessions are never used to choose a recipe or a threshold ([19 §11.4](19_design_review_parameters_and_novel_directions.md#114-what-changes) #3). The first sessions are **test**: they are the unseen set the project lacks. Empty-sky clips from later train sessions can become backgrounds for pasted drones.
* **Session types, in order:**
  * **A, daytime** (§4), repeated over weather: clear, overcast, haze, wind.
  * **B, terrain:** mountain ridges and treelines at the skyline, the club's mountain sites.
  * **C, birds:** a site and season with birds, for the track classifier (real non-drone tracks in the same scenes, the confound test of 14 §5).
  * **D, dusk and night (future, after daytime is done):** the user's own drones with lights on, dimmed and off, at dusk and on dark skies; empty night sky with stars and aircraft. Its questions and the assessment of a colour-pixel "heat map" are in 11 row 12.

---

## 9. Session Log

| Date | Site | Drones | Camera | Shots done | Test or train | Files | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | | | | | | | |
