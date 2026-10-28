"""
E20: the whole system on whole videos. How often does it DECLARE a drone when there is none, and how long does it
take to declare a real one?

Everything before this scored a component: the detector per image, the track classifier per window. The project's
headline number, false drones per hour, belongs to the system: detections -> tracker -> track classifier -> label,
frame by frame, exactly as `python -m cuas` runs (cuas.pipeline.TrackPipeline).

Held-out Halmstad VALIDATION videos (the detector never trained on them; the track classifier is fitted here on the
TRAIN videos only). A video holds one labelled class, so:
  * bird / airplane / helicopter videos: every track declared "drone" is a false declaration -> per hour of video;
  * drone videos: is the labelled drone's track declared, and how many seconds after it first appears?
Deciders compared on the same tracks:
  frame alarm     no tracking: any "drone" box at or above the threshold in a frame (what the detector alone gives)
  detector votes  a track is a drone if most of its boxes were classed "drone" (in-domain only: E15)
  track only      motion + shape features, no detector class (the out-of-domain fallback, 14 section 5)
  detector+track  both
A track is declared the first time its label has been "drone" for HOLD seconds without a break. The label is
re-read several times a second on overlapping windows, so with no hold a track gets many chances to be called a
drone once: per-window accuracy overstates what the system does.
Detections come from the cached detector run (m2_detect_videos.py), so this needs no GPU.

    C:\\cuas_venv\\Scripts\\python experiments/e20_system_eval.py [--tag dets_E12]        (~3 min, CPU)

Caveats: in-domain footage; ~6 minutes of non-drone video, so the rate has a wide interval (printed); Halmstad's
"no label" is not "no drone" (14 section 5), so a false declaration here can be an unlabelled real drone.
"""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.classify import DETECTOR, TRACK, TrackClassifier  # noqa: E402
from cuas.datasets.halmstad import CLASSES, NAME_RE, split_videos  # noqa: E402
from cuas.eval import iou  # noqa: E402
from cuas.pipeline import TrackPipeline  # noqa: E402
from cuas.track_features import FEATURE_NAMES  # noqa: E402

SRC = Path(r"C:\cuas_data\halmstad\Data\Video_V")
OUT = Path(r"C:\cuas_data\m2")
CONF = 0.4          # the score that starts a track, and the frame-alarm threshold
HOLDS = (0.0, 0.5, 1.0)


class Votes:
    """Decider with the classifier's interface: p(drone) = share of the track's boxes the detector classed drone."""
    k = FEATURE_NAMES.index("cls_drone_frac")

    def predict_proba(self, X):
        return np.asarray(X, float)[:, self.k]

    def decide(self, p, X, durations):
        out = np.where(np.asarray(p) >= 0.7, "drone", np.where(np.asarray(p) <= 0.3, "not_drone", "unknown"))
        out[np.asarray(durations) < 1.0] = "unknown"
        return out


def fit(tag, features):
    """Track classifier on 2-s windows at 30 fps of the TRAIN videos only."""
    import pandas as pd
    d = pd.read_csv(OUT / f"features_{tag}.csv")
    d = d[(d.fps == 30) & (d.window_s.astype(str) == "2.0") & (d.split == "train")].reset_index(drop=True)
    X = np.column_stack([d[c].to_numpy(float) if c in features else np.full(len(d), np.nan) for c in FEATURE_NAMES])
    keep = [f for f in features if d[f].notna().any()]
    return TrackClassifier(keep).fit(X, (d.cls == "drone").to_numpy(int), d.video.to_numpy()), len(d), d.video.nunique()


def poisson_upper(n, hours):
    """Upper 95% bound on a rate from n events (exact Poisson)."""
    from scipy.stats import chi2
    return chi2.ppf(0.95, 2 * (n + 1)) / 2 / hours


def main(tag):
    import cv2
    from mcos_decoder import load_groundtruth
    with gzip.open(OUT / f"{tag}.json.gz", "rt") as f:
        dets_all = json.load(f)
    videos = sorted(SRC.glob("V_*.mp4"))
    val = [v for v, part in split_videos(videos).items() if part == "val"]
    deciders = {"detector votes": Votes()}
    for name, feats in (("track only", TRACK), ("detector+track", DETECTOR + TRACK)):
        deciders[name], n_win, n_vid = fit(tag, feats)
    print(f"track classifiers fitted on {n_win} windows of {n_vid} TRAIN videos; evaluating {len(val)} val videos", flush=True)

    rows = []           # video, class, seconds, frames with a drone box, then per decider: declarations...
    for vid in sorted(val):
        cls = NAME_RE.search(vid.name).group(1).lower()
        labels = load_groundtruth(str(vid.with_name(vid.stem + "_LABELS.mat")))
        by_frame = {}
        for fr, x, y, w, h, c, s in dets_all[vid.stem]["dets"]:
            by_frame.setdefault(int(fr), []).append((x, y, w, h, int(c), s))
        pipes = {name: TrackPipeline(classifier=d) for name, d in deciders.items()}
        declared = {(name, h): {} for name in deciders for h in HOLDS}     # track id -> (time, on the labelled target?)
        since = {name: {} for name in deciders}         # track id -> time its label last became "drone"
        alerts = {name: set() for name in deciders}
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n, alarm_frames, first_gt = 0, 0, None
        while n < len(labels):                          # past the last label nothing can be scored (14 section 5)
            ok, frame = cap.read()
            if not ok:
                break
            t, gt = n / fps, labels[n]
            if gt is not None and first_gt is None:
                first_gt = t
            dets = by_frame.get(n, [])
            alarm_frames += any(d[4] == 0 and d[5] >= CONF for d in dets)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for name, pipe in pipes.items():
                for s in pipe.update(t, dets, gray):
                    if s["response"] in ("alert_operator", "request_engagement_decision"):
                        alerts[name].add(s["track_id"])
                    if s["label"] != "drone":
                        since[name].pop(s["track_id"], None)
                        continue
                    t_on = since[name].setdefault(s["track_id"], t)
                    for h in HOLDS:
                        if t - t_on >= h and s["track_id"] not in declared[name, h]:
                            declared[name, h][s["track_id"]] = (t, gt is not None and iou(s["box"], gt) >= 0.1)
            n += 1
        cap.release()
        row = {"video": vid.stem, "cls": cls, "seconds": n / fps, "alarm_frames": alarm_frames, "frames": n}
        for name, h in declared:
            on = [t for t, hit in declared[name, h].values() if hit]
            row[f"{name}, hold {h}"] = {"declared": len(declared[name, h]), "on_target": len(on),
                                        "delay": (min(on) - first_gt) if on and first_gt is not None else None,
                                        "alerts": len(alerts[name])}
        rows.append(row)
        print(f"{vid.stem}: {n} frames", flush=True)

    non = [r for r in rows if r["cls"] != "drone"]
    drone = [r for r in rows if r["cls"] == "drone"]
    hours = sum(r["seconds"] for r in non) / 3600
    lines = [f"E20 system evaluation ({tag}): Halmstad VAL videos, {len(drone)} drone and {len(non)} non-drone "
             f"({hours * 60:.1f} min of non-drone video: " + ", ".join(
                 f"{c} {sum(r['cls'] == c for r in non)}" for c in CLASSES[1:]) + ").",
             f"Track start threshold and frame-alarm threshold: conf {CONF}.", "",
             f"{'decider':26s} {'false drone declarations on non-drone videos':>46s} {'drone videos declared':>23s} "
             f"{'delay s: median / p90':>23s} {'false on drone videos':>22s}"]
    af = sum(r["alarm_frames"] for r in non)
    lines.append(f"{'frame alarm':26s} {f'{af} frames = {af / hours:,.0f} /h (in {sum(r["alarm_frames"] > 0 for r in non)} videos)':>46s} "
                 f"{f'{sum(r["alarm_frames"] > 0 for r in drone)}/{len(drone)}':>23s} {'-':>23s} {'-':>22s}")
    summary = {"hours_non_drone": hours, "frame_alarm_per_hour": af / hours}
    keys = [f"{name}, hold {h}" for name in deciders for h in HOLDS]
    for name in keys:
        fd = sum(r[name]["declared"] for r in non)
        on_other = sum(r[name]["on_target"] for r in non)
        got = [r[name]["delay"] for r in drone if r[name]["delay"] is not None]
        extra = sum(r[name]["declared"] - r[name]["on_target"] for r in drone)
        delay = f"{np.median(got):.2f} / {np.percentile(got, 90):.2f}" if got else "-"
        lines.append(f"{name:26s} {f'{fd} = {fd / hours:.0f} /h (95% upper {poisson_upper(fd, hours):.0f}); {on_other} on the labelled object':>46s} "
                     f"{f'{len(got)}/{len(drone)}':>23s} {delay:>23s} {extra:>22d}")
        summary[name] = {"false_declarations": fd, "per_hour": fd / hours, "upper95_per_hour": poisson_upper(fd, hours),
                         "on_labelled_distractor": on_other, "drone_videos_declared": len(got),
                         "delay_median_s": float(np.median(got)) if got else None,
                         "delay_p90_s": float(np.percentile(got, 90)) if got else None, "false_on_drone_videos": extra,
                         "alerts_non_drone": sum(r[name]["alerts"] for r in non),
                         "alerts_drone": sum(r[name]["alerts"] for r in drone)}
    lines += ["", "Rules floor (alert_operator or higher; no protected zone set, so only tracks closing on the camera "
              "can raise it): " + "; ".join(f"{n}: {summary[n]['alerts_non_drone']} on non-drone videos, "
                                            f"{summary[n]['alerts_drone']} on drone videos" for n in keys[::len(HOLDS)])]
    text = "\n".join(lines)
    (OUT / f"e20_system_{tag}.txt").write_text(text, encoding="utf-8")
    (OUT / f"e20_system_{tag}.json").write_text(json.dumps({"summary": summary, "videos": rows}, indent=1))
    with open(OUT / f"e20_system_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "cls", "seconds", "alarm_frames"] + [f"{n}: {k}" for n in keys for k in ("declared", "on_target", "delay")])
        w.writerows([[r["video"], r["cls"], round(r["seconds"], 2), r["alarm_frames"]]
                     + [r[n][k] for n in keys for k in ("declared", "on_target", "delay")] for r in rows])
    print("\n" + text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dets_E12", help="detections file <tag>.json.gz and features_<tag>.csv in C:/cuas_data/m2")
    main(ap.parse_args().tag)
