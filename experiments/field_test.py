"""
Field test: the FROZEN model over whole real videos of varied conditions, scored per clip and per condition.
The step from "scores on test images" to "what the program does on footage": the same code path as
`python -m cuas` (cuas.replay.run), whole videos, every condition the user wants covered (open sky, trees, buildings,
dusk, night), and the system-level numbers: is the drone declared, how soon, and how many false declarations per hour.

The model is a frozen bundle (C:\\cuas_data\\models\\<name>\\model.json: weights, track classifier, threshold,
evidence): every field result names the bundle it came from, so results stay comparable while training goes on.

    C:\\cuas_venv\\Scripts\\python experiments/field_test.py --set dut_tracking antiuav_test multiview
    C:\\cuas_venv\\Scripts\\python experiments/field_test.py --folder D:\\cuas_data\\field\\2026-11-01 --tag own
    C:\\cuas_venv\\Scripts\\python experiments/field_test.py --sheet dut_tracking     (one labelled frame per clip)
    C:\\cuas_venv\\Scripts\\python experiments/field_test.py --report                 (tables from saved results)

Finished clips are skipped (one JSON per clip), so a stopped run just continues. --render also writes an annotated
video per clip. Results: D:\\cuas_data\\field_test\\<model>\\.

Scoring (per clip; the drone's position comes from the set's labels, as a box or only a centre point):
  * a box MATCHES the drone if their centres are within max(12 px, half the larger side of either box);
  * frames: drone frames where a "drone" detection at the threshold matches (det recall), where a detection of ANY
    class does, and where a live track does (track cover);
  * tracks: a track is ON TARGET if it matches in at least half of its frames that have a known answer (drone
    visible, or labelled absent), OFF TARGET if not, UNVERIFIED if none of its frames has a known answer
    (multi-view "not distinguishable" frames, DUT's negative boxes);
  * declarations are judged by what the track followed in the 2 s up to the declaration (same on / off / unverified
    rule over that window): the drone is DECLARED if such a declaration is on target; delay = first declaration -
    first labelled drone frame. An off-target declaration is a FALSE declaration (per hour of video); an unverified
    one is counted apart. A track declared on the drone that is off target over its whole life DRIFTED (counted).
    Each track's per-frame status is saved, so the scoring can be changed and redone with --report alone. Every false or unverified declaration is saved as a crop for an audit by eye: a set labels one
    drone, so a second drone or a bird in frame is "false" here until someone looks.
Conditions: "night" is set automatically (median frame brightness under 50 of 255, the bimodal split of 15/E35);
backgrounds (sky / trees / buildings / ...) come from experiments/field_conditions.json, tagged by eye from --sheet.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.replay import open_source, run  # noqa: E402

DS = Path(r"D:\cuas_data\datasets")
OUT = Path(r"D:\cuas_data\field_test")
MODELS = Path(r"C:\cuas_data\models")
CONDITIONS = Path(__file__).with_name("field_conditions.json")
NIGHT = 50.0
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mts", ".webm"}


# --- sets: each yields clips {set, clip, source, fps, gt, preprocess}; gt maps frame -> (cx, cy, w, h) with w = h =
# None for a centre point, or "absent" (labelled: no drone), or is missing (unknown); gt None = no labels at all.

def set_dut_tracking():
    """DUT Anti-UAV tracking: 20 videos as JPEG sequences, one box per frame; w < 0 = target absent, which is NOT
    trusted as "no drone" (03: count it only after a look), so those frames are unknown. The frame rate is not
    published: 30 is assumed (it changes seconds and per-hour figures, not counts)."""
    root = DS / "dut_tracking" / "unpacked"
    for d in sorted((root / "Anti-UAV-Tracking-V0").glob("video*")):
        gt = {}
        for i, line in enumerate((root / "Anti-UAV-Tracking-V0GT" / f"{d.name}_gt.txt").read_text().split("\n")):
            v = [float(x) for x in line.replace(",", " ").split()]
            if len(v) == 4 and v[2] > 0:
                gt[i] = (v[0] + v[2] / 2, v[1] + v[3] / 2, v[2], v[3])
        yield {"set": "dut_tracking", "clip": d.name, "source": str(d / "%05d.jpg"), "fps": 30.0, "gt": gt}


def set_antiuav_test():
    """Anti-UAV300 TEST, visible videos (91; extracted from the RGBT zip). The detector trained on the train split
    of the same dataset: unseen videos, not an unseen dataset. exist = 0 is a labelled absence. The burned-in
    overlay is inpainted, as for the training images."""
    for c in _antiuav_clips(DS / "antiuav_test_video" / "test"):
        yield dict(c, set="antiuav_test")


def _antiuav_clips(root):
    mask = cv2.imread(r"C:\cuas_data\antiuav_overlay_mask.png", cv2.IMREAD_GRAYSCALE)

    def inpaint(frame):
        return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA) if mask is not None and mask.shape == frame.shape[:2] \
            else frame
    for d in sorted(root.iterdir()):
        j = json.loads((d / "visible.json").read_text())
        gt = {}
        for i, (e, r) in enumerate(zip(j["exist"], j["gt_rect"])):
            if not e:
                gt[i] = "absent"
            elif len(r) == 4 and r[2] > 0:
                gt[i] = (r[0] + r[2] / 2, r[1] + r[3] / 2, r[2], r[3])
        yield {"clip": d.name, "source": str(d / "visible.mp4"), "gt": gt, "preprocess": inpaint}


def set_multiview():
    """Multi-view drone tracking (Li et al. 2020): one hexacopter filmed by phones and cameras, centre points only.
    Rows of 0 0 mean "not distinguishable" (unknown), not "no drone". Dataset 5 has no 2D labels: its declarations
    can only be audited. The frame id in the label files is taken as the video's frame index."""
    for v in sorted((DS / "multiview_drone_tracking" / "videos").glob("dataset*/cam*.mp4")):
        lab = DS / "multiview_drone_tracking" / "repo" / v.parent.name / "detections" / f"{v.stem}.txt"
        gt = None
        if lab.exists():
            gt = {}
            for line in lab.read_text().split("\n"):
                f = line.split()
                if len(f) == 3 and float(f[1]) > 0 and float(f[2]) > 0:
                    gt[int(float(f[0]))] = (float(f[1]), float(f[2]), None, None)
        yield {"set": "multiview", "clip": f"{v.parent.name}_{v.stem}", "source": str(v), "gt": gt}


GAREUD_TAGS = {"tree": "trees", "woodland": "trees", "building": "buildings", "night": "night", "dusk": "dusk",
               "snow": "snow", "reed": "occlusion", "lamp": "lights", "traffic-light": "lights",
               "overexposure": "glare", "sky": "sky"}


def set_gareud():
    """GAREUD-Real (ground-to-air RGB part only; CC BY-NC 4.0): 15 recordings from a fixed tripod, 1024x576 at
    30 fps, frames named by timestamp. labels.txt: `timestamp class xc yc w h` (normalised); a timestamp alone is a
    frame with no drone (an explicit absence, as the dataset defines it). Conditions come from each recording's
    sequence_info.txt (night 10-100 lux with street lamps, dusk, complex trees, woodland, reeds, snow, glare)."""
    for d in sorted((DS / "gareud" / "unpacked").glob("GAREUD_R_*")):
        frames = sorted((d / "rgb_todo").glob("*.jpg"), key=lambda p: float(p.stem))
        if not frames:
            continue
        W, H = 1024, 576
        im = cv2.imread(str(frames[0]))
        if im is not None:
            H, W = im.shape[:2]
        lab = {}
        for line in (d / "labels.txt").read_text(encoding="utf-8-sig").split("\n"):
            f = line.split()
            if not f:
                continue
            boxes = lab.setdefault(f[0], [])
            for i in range(1, len(f) - 4, 5):
                xc, yc, w, h = (float(v) for v in f[i + 1:i + 5])
                boxes.append((xc * W, yc * H, w * W, h * H))
        gt = {n: (lab[p.stem] or "absent") for n, p in enumerate(frames) if p.stem in lab}
        info = (d / "sequence_info.txt").read_text(encoding="utf-8-sig").lower()
        cond = next((line for line in info.split("\n") if line.startswith("scene_conditions")), "")
        tags = sorted({v for k, v in GAREUD_TAGS.items() if k in cond})
        yield {"set": "gareud", "clip": d.name.replace("GAREUD_", ""), "source": frames, "fps": 30.0, "gt": gt,
               "tags": tags}


def set_fred():
    """FRED test split (Apache-2.0): 47 tripod sequences, 1280x720 at 30 fps, frames named by wall-clock time;
    coordinates_rgb.txt gives `seconds: x1, y1, x2, y2, id, model` per drone (several drones in some sequences).
    The seconds are taken from the first frame (checked on the contact sheet). A frame with no line is UNKNOWN (the
    labels do not say a missing line means no drone). Night comes from brightness (the paper: > 1 h low light)."""
    split_dirs = [(s, d) for s in ("test", "train") if (DS / "fred" / "unpacked" / s).exists()
                  for d in sorted((DS / "fred" / "unpacked" / s).iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0)]
    for split, d in split_dirs:       # train holds only the dusk sequences 30-33, 35, 36 (never trained on)
        frames = sorted((d / "RGB").glob("*.jpg"))
        if not frames:
            continue

        def secs(p):
            h, m, s = p.stem.split("_")[-3:]
            return 3600 * int(h) + 60 * int(m) + float(s)
        t = np.array([secs(p) for p in frames])
        t -= t[0]
        gt = {}
        for line in (d / "coordinates_rgb.txt").read_text().split("\n"):
            if ":" not in line:
                continue
            ts, rest = line.split(":", 1)
            v = rest.split(",")
            x1, y1, x2, y2 = (float(a) for a in v[:4])
            n = int(np.argmin(np.abs(t - float(ts))))
            if abs(t[n] - float(ts)) < 0.02:
                gt.setdefault(n, []).append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        yield {"set": "fred", "clip": d.name if split == "test" else f"train{d.name}", "source": frames, "fps": 30.0,
               "gt": gt}


def set_antiuav_trainval():
    """TRAINING SOURCE (not a test): Anti-UAV300 train + val visible videos (160 + 67). The detector trained on
    every 40th frame of the train split, so these give track features, not detector scores, worth learning from.
    Day and night; tracks off the one labelled drone are clutter (night lights, buildings)."""
    for split in ("train", "val"):
        for c in _antiuav_clips(DS / "antiuav_trainval_video" / split):
            yield dict(c, set="antiuav_trainval", clip=f"{split}_{c['clip']}", obj_class="drone")


def set_halmstad_train():
    """TRAINING SOURCE (not a test): Halmstad TRAIN-split videos (split_videos, seed 0, as every detector run).
    Each video holds one labelled object of one class; "drone" positions here mark THAT object, whatever its class
    (obj_class), so tracks on birds, airplanes and helicopters become not-drone examples. Frames past the end of
    the labels, or unlabelled, are unknown (14 section 5: "no label" is not "no drone")."""
    from mcos_decoder import load_groundtruth
    from cuas.datasets.halmstad import NAME_RE, split_videos
    src = Path(r"C:\cuas_data\halmstad\Data\Video_V")
    videos = sorted(src.glob("V_*.mp4"))
    for v, part in split_videos(videos).items():
        m = NAME_RE.search(v.name)
        if part != "train" or not m:
            continue
        labels = load_groundtruth(str(v.with_name(v.stem + "_LABELS.mat")))
        gt = {i: (b[0] + b[2] / 2, b[1] + b[3] / 2, b[2], b[3]) for i, b in enumerate(labels)
              if b is not None and len(b) == 4 and b[2] > 0}
        yield {"set": "halmstad_train", "clip": v.stem, "source": str(v), "gt": gt, "obj_class": m.group(1).lower()}


def set_folder(folder, tag):
    """Own footage or any folder of videos with no labels: only declarations, all to be audited by eye."""
    for v in sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in VIDEO_EXT):
        yield {"set": tag, "clip": v.stem, "source": str(v), "gt": None}


SETS = {"dut_tracking": set_dut_tracking, "antiuav_test": set_antiuav_test, "multiview": set_multiview,
        "gareud": set_gareud, "fred": set_fred,
        "antiuav_trainval": set_antiuav_trainval, "halmstad_train": set_halmstad_train}
TRAIN_SETS = {"antiuav_trainval", "halmstad_train"}     # sources for fitting the decision layer, never test sets:
                                                        # their results go to <model>_train, out of the report


# --- scoring

def match(a, b) -> bool:
    """a, b: (x, y, w, h) top-left boxes, or a gt (cx, cy, w, h) with None sizes, given as b."""
    ax, ay = a[0] + a[2] / 2, a[1] + a[3] / 2
    if b[2] is None:
        bx, by, side = b[0], b[1], 0.0
    else:
        bx, by, side = b[0], b[1], max(b[2], b[3])
    return np.hypot(ax - bx, ay - by) <= max(12.0, 0.5 * max(a[2], a[3], side))


class Recording:
    """The track classifier, passed through, keeping every feature row it is asked about. The pipeline asks once
    per fresh track per frame, in the order of the states it returns, so the Scorer pairs rows with tracks."""

    def __init__(self, clf):
        self.clf, self.rows = clf, []

    def predict_proba(self, X):
        self.rows.append(np.asarray(X, float)[0].tolist())
        return self.clf.predict_proba(X)

    def __getattr__(self, name):
        return getattr(self.clf, name)


class Scorer:
    def __init__(self, gt, conf, drone_cls, fps, crop_dir, clip, recording=None):
        self.gt, self.conf, self.drone_cls, self.fps = gt, conf, drone_cls, fps
        self.crop_dir, self.clip, self.recording = crop_dir, clip, recording
        self.n = self.pos = self.absent = self.det_drone = self.det_any = self.cover = 0
        self.first_pos_t = None
        self.bright = []
        self.tracks = {}

    def __call__(self, n, t, frame, dets, states):
        self.n += 1
        if n % 30 == 0:
            g = cv2.cvtColor(cv2.resize(frame, (96, 54)), cv2.COLOR_BGR2GRAY)
            self.bright.append(float(g.mean()))
        g = None if self.gt is None else self.gt.get(n)
        drone = g is not None and g != "absent"
        gs = (g if isinstance(g, list) else [g]) if drone else []      # several drones in a frame: a list
        for one in gs:                                                  # counted per drone, not per frame
            self.pos += 1
            self.first_pos_t = t if self.first_pos_t is None else self.first_pos_t
            hits = [d for d in dets if d[5] >= self.conf and match(d, one)]
            self.det_drone += any(d[4] == self.drone_cls for d in hits)
            self.det_any += bool(hits)
            self.cover += any(not s["coasting"] and match(s["box"], one) for s in states)
        self.absent += g == "absent"
        for s in states:
            k = self.tracks.setdefault(s["track_id"], {"first_t": t, "first_frame": n, "status": [], "declared_t": None,
                                                       "declared_frame": None, "final_label": None, "max_w": 0.0,
                                                       "windows": []})
            k["last_t"], k["final_label"] = t, s["label"]
            k["max_w"] = max(k["max_w"], float(s["box"][2]))
            # One letter a frame, kept in the results so the scoring can be redone without re-running:
            # o = on a labelled drone, x = off it (or on a labelled-absent frame), ? = no answer, - = coasting.
            k["status"].append("-" if s["coasting"] else "o" if drone and any(match(s["box"], one) for one in gs)
                               else "x" if drone or g == "absent" else "?")
            if s["label"] == "drone" and k["declared_t"] is None:
                k["declared_t"], k["declared_frame"] = t, n
                k["crop"] = self._crop(frame, s["box"], s["track_id"])
        if self.recording is not None:
            fresh = [s for s in states if s["p_fresh"]]
            if len(self.recording.rows) != len(fresh):
                raise RuntimeError(f"frame {n}: {len(self.recording.rows)} classifier calls, {len(fresh)} fresh tracks")
            for s, x in zip(fresh, self.recording.rows):
                # Every classifier window: t, track age, usable, p, then the feature row (FEATURE_NAMES order), so
                # any classifier or decision rule can be replayed offline on the same tracks (e40_decision.py).
                self.tracks[s["track_id"]]["windows"].append(
                    [round(t, 4), round(s["duration_s"], 4), bool(s["p_usable"]), s["drone_prob"], *x])
            self.recording.rows.clear()

    def _crop(self, frame, box, tid):
        x, y, w, h = box
        cx, cy, r = x + w / 2, y + h / 2, max(48, int(2 * max(w, h)))
        H, W = frame.shape[:2]
        x0, y0 = int(max(0, cx - r)), int(max(0, cy - r))
        img = frame[y0:int(min(H, cy + r)), x0:int(min(W, cx + r))].copy()
        cv2.rectangle(img, (int(x - x0), int(y - y0)), (int(x - x0 + w), int(y - y0 + h)), (0, 0, 255), 1)
        p = self.crop_dir / f"{self.clip}_track{tid}.jpg"
        cv2.imwrite(str(p), cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST) if r < 96 else img)
        return p.name

    def result(self):
        return rescore({
            "frames": self.n, "fps": self.fps, "minutes": self.n / self.fps / 60,
            "brightness": float(np.median(self.bright)) if self.bright else None,
            "labelled": self.gt is not None, "drone_frames": self.pos, "absent_frames": self.absent,
            "first_drone_t": self.first_pos_t,
            "det_recall_drone": self.det_drone / self.pos if self.pos else None,
            "det_recall_any": self.det_any / self.pos if self.pos else None,
            "track_cover": self.cover / self.pos if self.pos else None,
            "tracks": len(self.tracks),
            "track_status": {str(tid): {"first_frame": k["first_frame"], "declared_frame": k["declared_frame"],
                                        "final_label": k["final_label"], "status": "".join(k["status"]),
                                        "windows": k["windows"]}
                             for tid, k in self.tracks.items()},
            "_declared": {str(tid): {"t": round(k["declared_t"], 2), "crop": k.get("crop"), "max_w": round(k["max_w"], 1),
                                     "duration_s": round(k["last_t"] - k["first_t"], 2)}
                          for tid, k in self.tracks.items() if k["declared_t"] is not None},
        })


def verdict(status: str) -> str:
    on, off = status.count("o"), status.count("x")
    return "unverified" if on + off == 0 else ("on_target" if on >= 0.5 * (on + off) else "off_target")


def rescore(r: dict, window_s: float = 2.0) -> dict:
    """Declaration and track verdicts from a clip result's per-frame track status (so saved results can be rescored).
    A declaration is judged by what the track followed in the `window_s` seconds up to it: a track declared while on
    the drone that later drifts off is a correct declaration plus a drift, not a false one."""
    decl = {"on_target": [], "off_target": [], "unverified": []}
    drifted, drone_tracks = 0, 0
    called_not_drone = 0
    w = max(1, int(round(window_s * r["fps"])))
    for tid, k in r["track_status"].items():
        whole = verdict(k["status"])
        if whole == "on_target":
            drone_tracks += 1
            called_not_drone += k["final_label"] == "not_drone"
        if k["declared_frame"] is None:
            continue
        i = k["declared_frame"] - k["first_frame"]
        v = verdict(k["status"][max(0, i - w + 1):i + 1])
        d = dict(r["_declared"][tid], track_id=int(tid), whole_track=whole)
        decl[v].append(d)
        drifted += v == "on_target" and whole == "off_target"
    first = min((d["t"] for d in decl["on_target"]), default=None)
    r.update(declarations=decl, drifted_after_declaration=drifted,
             drone_declared=bool(decl["on_target"]) if r["drone_frames"] else None,
             delay_s=None if first is None or r["first_drone_t"] is None else round(first - r["first_drone_t"], 2),
             drone_tracks=drone_tracks, drone_tracks_called_not_drone=called_not_drone)
    return r


# --- running

def load_model(name):
    m = json.loads((MODELS / name / "model.json").read_text())
    m["dir"] = MODELS / name
    return m


def run_clip(c, model, out_dir, render, force):
    f = out_dir / c["set"] / f"{c['clip']}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())
    crops = out_dir / c["set"] / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    from cuas.classify import TrackClassifier
    from cuas.detector import load
    det = load(str(model["dir"] / model["weights"]), conf=0.1, imgsz=model["tile"])
    drone_cls = next(k for k, v in det.names.items() if v == "drone")
    fps = c.get("fps")
    if not fps:
        cap = cv2.VideoCapture(c["source"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()
    rec = Recording(TrackClassifier.load(str(model["dir"] / model["clf"])))
    sc = Scorer(c["gt"], model["conf"], drone_cls, fps, crops, c["clip"], recording=rec)
    r = run(c["source"], detector=det, classifier=rec,
            conf=model["conf"], evidence=model["evidence"], scan_every=model["scan_every"], tile=model["tile"],
            fps=fps, preprocess=c.get("preprocess"), on_frame=sc, quiet=True,
            out=str(out_dir / c["set"] / f"{c['clip']}.mp4") if render else None)
    src = c["source"] if isinstance(c["source"], str) else str(Path(c["source"][0]).parent)
    res = {"set": c["set"], "clip": c["clip"], "source": src, "model": model["name"], "tags": c.get("tags", []),
           "obj_class": c.get("obj_class", "drone"),
           **sc.result(),
           "timing_ms": {k: round(r["summary"][k], 1) for k in ("motion_ms", "detector_ms", "track_ms")}}
    f.write_text(json.dumps(res, indent=1))
    return res


def conditions_of(r, tags):
    cond = list(tags.get(f"{r['set']}/{r['clip']}", r.get("tags") or tags.get(r["set"], [])))
    if r["brightness"] is not None and r["brightness"] < NIGHT and "night" not in cond:
        cond.append("night")
    if not any(c in cond for c in ("night", "dusk")):
        cond.append("day")
    return cond


def report(out_dir):
    tags = json.loads(CONDITIONS.read_text()) if CONDITIONS.exists() else {}
    tags = {k: v for k, v in tags.items() if not k.startswith("_")}
    rs = [rescore(json.loads(p.read_text())) for p in sorted(out_dir.glob("*/*.json"))]
    lines = [f"Field test, model {out_dir.name}: {len(rs)} clips, {sum(r['minutes'] for r in rs):.0f} min", ""]
    hdr = (f"{'set/clip':38s} {'cond':22s} {'min':>5s} {'drone fr':>8s} {'det(dr)':>7s} {'det(any)':>8s} "
           f"{'track':>6s} {'decl':>5s} {'delay':>6s} {'false':>5s} {'unver':>5s} {'drift':>5s} {'notdr':>5s}")
    lines.append(hdr)

    def fmt(v, p=2):
        return "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
    groups = defaultdict(list)
    for r in rs:
        cond = conditions_of(r, tags)
        for c in cond:
            groups[(r["set"], c)].append(r)
            groups[("ALL", c)].append(r)
        d = r["declarations"]
        lines.append(f"{(r['set'] + '/' + r['clip'])[:38]:38s} {','.join(cond)[:22]:22s} {r['minutes']:5.1f} "
                     f"{r['drone_frames']:8d} {fmt(r['det_recall_drone']):>7s} {fmt(r['det_recall_any']):>8s} "
                     f"{fmt(r['track_cover']):>6s} {fmt({True: 'yes', False: 'NO', None: '-'}[r['drone_declared']]):>5s} "
                     f"{fmt(r['delay_s'], 1):>6s} {len(d['off_target']):5d} {len(d['unverified']):5d} "
                     f"{r['drifted_after_declaration']:5d} "
                     f"{r['drone_tracks_called_not_drone']:2d}/{r['drone_tracks']:<2d}")
    lines += ["(drift = declared on the drone, off target over the whole track; notdr = drone tracks whose last label",
              "was not_drone / all drone tracks)",
              "", "By condition (frames pooled; declared = share of labelled clips whose drone was declared;",
              "false/h = declared off-target tracks per hour; every false or unverified declaration is a crop to audit)",
              f"{'set':14s} {'condition':10s} {'clips':>5s} {'hours':>6s} {'det(dr)':>7s} {'det(any)':>8s} "
              f"{'track':>6s} {'declared':>9s} {'med delay':>9s} {'false':>5s} {'false/h':>7s} {'unver':>5s}"]
    for (s, c), g in sorted(groups.items(), key=lambda kv: (kv[0][0] != "ALL", kv[0])):
        pos = sum(r["drone_frames"] for r in g)
        hours = sum(r["minutes"] for r in g) / 60

        def pooled(key):
            return sum((r[key] or 0) * r["drone_frames"] for r in g) / pos if pos else None
        lab = [r for r in g if r["drone_declared"] is not None]
        delays = [r["delay_s"] for r in lab if r["delay_s"] is not None]
        false = sum(len(r["declarations"]["off_target"]) for r in g)
        unver = sum(len(r["declarations"]["unverified"]) for r in g)
        declared = f"{sum(r['drone_declared'] for r in lab)}/{len(lab)}" if lab else "-"
        lines.append(f"{s:14s} {c:10s} {len(g):5d} {hours:6.2f} {fmt(pooled('det_recall_drone')):>7s} "
                     f"{fmt(pooled('det_recall_any')):>8s} {fmt(pooled('track_cover')):>6s} "
                     f"{declared:>9s} "
                     f"{fmt(float(np.median(delays)) if delays else None, 1):>9s} {false:5d} "
                     f"{fmt(false / hours if hours else None, 0):>7s} {unver:5d}")
    text = "\n".join(lines)
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    print(text)


def sheet(clips, out):
    """One labelled frame per clip (the middle labelled one) with the label drawn, tiled 4 across, to tag
    backgrounds by eye into field_conditions.json."""
    thumbs = []
    for c in clips:
        cap, _ = open_source(c["source"])
        drone = [k for k, v in (c["gt"] or {}).items() if v != "absent"]
        n = sorted(drone)[len(drone) // 2] if drone else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, fr = cap.read()
        cap.release()
        if not ok:
            continue
        g = (c["gt"] or {}).get(n)
        for one in (g if isinstance(g, list) else [g]) if g and g != "absent" else []:
            r = 30 if one[2] is None else int(max(one[2], one[3]) / 2 + 30)
            cv2.circle(fr, (int(one[0]), int(one[1])), r, (0, 0, 255), 4)
        t = cv2.resize(fr, (480, int(480 * fr.shape[0] / fr.shape[1])))
        t = cv2.copyMakeBorder(t, 0, 270 - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) if t.shape[0] < 270 else t[:270]
        cv2.putText(t, c["clip"], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        thumbs.append(t)
    while len(thumbs) % 4:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [np.hstack(thumbs[i:i + 4]) for i in range(0, len(thumbs), 4)]
    out.parent.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(rows), 4):      # 16 clips a sheet
        cv2.imwrite(str(out.with_name(f"{out.stem}_{i // 4}.jpg")), np.vstack(rows[i:i + 4]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"{len(thumbs)} thumbnails -> {out.parent}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cuas_v1", help="frozen bundle in C:\\cuas_data\\models")
    ap.add_argument("--set", nargs="*", default=[], choices=sorted(SETS))
    ap.add_argument("--folder", help="a folder of unlabelled videos (own footage)")
    ap.add_argument("--tag", default="own", help="set name for --folder")
    ap.add_argument("--clips", nargs="*", help="only these clip names")
    ap.add_argument("--render", action="store_true", help="also write an annotated video per clip")
    ap.add_argument("--force", action="store_true", help="re-run clips that already have results")
    ap.add_argument("--sheet", nargs="*", help="contact sheets of these sets instead of running")
    ap.add_argument("--report", action="store_true", help="only rebuild the report from saved results")
    a = ap.parse_args()
    model = load_model(a.model)
    out_dir = OUT / model["name"]
    if a.sheet is not None:
        for s in a.sheet:
            sheet(list(SETS[s]()), OUT / "sheets" / f"{s}.jpg")
        return
    if not a.report:
        clips = [c for s in a.set for c in SETS[s]()] + (list(set_folder(a.folder, a.tag)) if a.folder else [])
        clips = [c for c in clips if not a.clips or c["clip"] in a.clips]
        for i, c in enumerate(clips):
            r = run_clip(c, model, OUT / f"{model['name']}_train" if c["set"] in TRAIN_SETS else out_dir,
                         a.render, a.force)
            d = r["declarations"]
            print(f"[{i + 1}/{len(clips)}] {c['set']}/{c['clip']}: {r['minutes']:.1f} min, det {r['det_recall_drone']}, "
                  f"declared {r['drone_declared']} (delay {r['delay_s']}), false {len(d['off_target'])}, "
                  f"unverified {len(d['unverified'])}", flush=True)
    if not all(s in TRAIN_SETS for s in a.set) or a.report or a.folder:
        report(out_dir)


if __name__ == "__main__":
    main()
