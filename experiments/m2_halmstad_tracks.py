"""
M2 experiments on Halmstad (CPU): can track features tell drones from birds, airplanes and helicopters, how
many seconds of track does that take, and how well calibrated is the answer?

Halmstad videos each hold ONE class (V_DRONE_*, V_BIRD_*, ...), so every track gets a label for free.

Two sources of detections:
* **oracle** (default): ground-truth boxes with centre/size jitter and 10% dropped detections. Isolates
  tracker + features + classifier from detector errors. Class/score are NOT passed (they would leak the label).
* **detector** (`--dets dets_<run>.json.gz` from `m2_detect_videos.py`, run on Colab): real detections of all
  classes at conf >= 0.05. The target track is the one matching the ground truth (IoU >= 0.3) in the most
  frames; other confirmed tracks are counted as "other tracks" (clutter or unlabelled objects).

    C:\\cuas_venv\\Scripts\\python experiments/m2_halmstad_tracks.py extract                 (~3 min)
    C:\\cuas_venv\\Scripts\\python experiments/m2_halmstad_tracks.py extract --dets C:/cuas_data/m2/dets_E12.json.gz
    C:\\cuas_venv\\Scripts\\python experiments/m2_halmstad_tracks.py classify [--tag dets_E12]

Outputs in C:\\cuas_data\\m2\\: features_<tag>.csv, tracking_<tag>.csv, classify_<tag>.txt, track_clf_<tag>.pkl
(the calibrated classifier for `cuas.pipeline`, trained on 2-s windows at 30 fps).

Threat to validity: each class was filmed in its own sessions, so recording conditions (box size, position)
can separate classes by themselves. `classify` reports a size/position-only probe to expose that.
"""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.classify import DETECTOR, KINEMATIC, SHAPE, TRACK, TrackClassifier, ece  # noqa: E402
from cuas.datasets.halmstad import CLASSES, NAME_RE, split_videos  # noqa: E402
from cuas.detector import nms  # noqa: E402
from cuas.eval import iou  # noqa: E402
from cuas.track import Tracker  # noqa: E402
from cuas.track_features import FEATURE_NAMES, silhouette_at, track_features  # noqa: E402

SRC = Path(r"C:\cuas_data\halmstad\Data\Video_V")
OUT = Path(r"C:\cuas_data\m2")
WINDOWS = (1.0, 2.0, 4.0, None)          # seconds of track per sample; None = whole track
RATES = {30: 1, 10: 3}                   # frames per second -> keep every Nth frame
PROBES = ["w_med_px", "y_med_px", "x_med_px"]   # not features: used only for the confound probe


def damaged(box, rng, jitter=0.05, size_jit=0.05, p_drop=0.1):
    """Ground-truth box -> oracle detection with centre/size noise (fractions of box width), or None."""
    if box is None or rng.random() < p_drop:
        return None
    x, y, w, h = box
    cx, cy = x + w / 2 + rng.normal(0, jitter * w), y + h / 2 + rng.normal(0, jitter * w)
    w2, h2 = w * (1 + rng.normal(0, size_jit)), h * (1 + rng.normal(0, size_jit))
    return (cx - w2 / 2, cy - h2 / 2, w2, h2)


def _windows(rows, area, bright):
    for win in WINDOWS:
        spans = [(rows[0, 0], rows[-1, 0] + 1e-9)] if win is None else \
            [(s, s + win) for s in np.arange(rows[0, 0], rows[-1, 0] - 0.8 * win + 1e-9, win)]
        for s0, s1 in spans:
            k = (rows[:, 0] >= s0) & (rows[:, 0] < s1)
            if k.sum() >= 5:
                f = track_features(rows[k], area[k], bright[k])
                probe = [float(np.median(rows[k, 3])), float(np.median(rows[k, 2])), float(np.median(rows[k, 1]))]
                yield ("full" if win is None else win), round(float(s0), 3), f, probe


def extract(dets_path=None, seed=0):
    import cv2
    from mcos_decoder import load_groundtruth

    tag = "oracle" if dets_path is None else Path(dets_path).name.split(".")[0]
    dets_all = None
    if dets_path:
        with gzip.open(dets_path, "rt") as f:
            dets_all = json.load(f)
    videos = sorted(SRC.glob("V_*.mp4"))
    split = split_videos(videos)
    OUT.mkdir(parents=True, exist_ok=True)
    rows_out, summary = [], []
    for vid in videos:
        m = NAME_RE.search(vid.name)
        cls = CLASSES.index(m.group(1).lower())
        labels = load_groundtruth(str(vid.with_name(vid.stem + "_LABELS.mat")))
        rng = np.random.default_rng([seed, int(m.group(2)), cls])
        by_frame = {}
        if dets_all is not None:
            for fr, x, y, w, h, c, s in dets_all[vid.stem]["dets"]:
                by_frame.setdefault(int(fr), []).append((x, y, w, h, int(c), s))
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []                        # (t, gt box or None, dets, {det index: (area, contrast)})
        for i in range(len(labels)):
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if dets_all is None:
                d = damaged(labels[i], rng)
                dets = [] if d is None else [(*d, 0, 0.9)]        # class/score carry no label information
            else:
                dets = nms(by_frame.get(i, []), class_aware=False)     # as cuas.pipeline does
            frames.append((i / fps, labels[i], dets, [silhouette_at(gray, d) for d in dets]))
        cap.release()
        for rate, step in RATES.items():
            tr = Tracker()
            sil = {}                       # time -> [(centre x, centre y, (area, contrast)), ...]
            for t, _, dets, s in frames[::step]:
                sil[round(t, 4)] = [(d[0] + d[2] / 2, d[1] + d[3] / 2, st) for d, st in zip(dets, s)]
                tr.update(t, dets)
            gt = {round(t, 4): b for t, b, _, _ in frames[::step]}
            tracks = tr.all_tracks()

            def matched(track):
                r = track.rows()
                hit = [gt.get(round(t, 4)) is not None and
                       iou((cx - w / 2, cy - h / 2, w, h), gt[round(t, 4)]) >= 0.3 for t, cx, cy, w, h in r[:, :5]]
                return int(np.sum(hit))
            if dets_all is None:
                main = max(tracks, key=lambda k: len(k.history)) if tracks else None
                others = 0
            else:
                scored = sorted(((matched(k), k) for k in tracks), key=lambda x: -x[0])
                main = scored[0][1] if scored and scored[0][0] >= max(5, 0.5 * len(scored[0][1].history)) else None
                others = len(tracks) - (main is not None)
            n_gt = sum(b is not None for _, b, _, _ in frames[::step])
            summary.append((vid.stem, split[vid], rate, len(tracks), others,
                            0 if main is None else len(main.history), n_gt, len(frames[::step]) / rate))
            if main is None:
                continue
            rows = main.rows()

            def stats(r):                  # silhouette of the detection this row came from (same frame, same centre)
                cands = sil.get(round(r[0], 4), [])
                if not cands:
                    return (np.nan, np.nan)
                return min(cands, key=lambda c: (c[0] - r[1]) ** 2 + (c[1] - r[2]) ** 2)[2]
            st = [stats(r) for r in rows]
            area, bright = np.array([a for a, _ in st]), np.array([b for _, b in st])
            for win, start, f, probe in _windows(rows, area, bright):
                rows_out.append([vid.stem, split[vid], CLASSES[cls], rate, win, start]
                                + [f[n] for n in FEATURE_NAMES] + probe)
        print(f"{vid.stem}: {len(frames)} frames", flush=True)
    with open(OUT / f"features_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "split", "cls", "fps", "window_s", "start_s"] + FEATURE_NAMES + PROBES)
        w.writerows(rows_out)
    with open(OUT / f"tracking_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "split", "fps", "n_tracks", "other_tracks", "main_track_obs", "gt_frames", "seconds"])
        w.writerows(summary)
    s30 = [s for s in summary if s[2] == 30]
    found = [s for s in s30 if s[5] > 0]
    cov = np.array([s[5] / s[6] for s in found if s[6]])
    print(f"wrote {len(rows_out)} samples ({tag})")
    print(f"30 fps: target track found in {len(found)}/{len(s30)} videos; it holds median {np.median(cov):.0%} of "
          f"labelled frames (10th pct {np.percentile(cov, 10):.0%}); other tracks "
          f"{sum(s[4] for s in s30) / (sum(s[7] for s in s30) / 60):.1f} per minute of video")


def _auc_ci(y, p, groups, rng, n_boot=1000):
    from sklearn.metrics import roc_auc_score
    vids = np.unique(groups)
    where = {v: np.flatnonzero(groups == v) for v in vids}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([where[v] for v in rng.choice(vids, len(vids))])
        if 0 < y[idx].mean() < 1:
            boots.append(roc_auc_score(y[idx], p[idx]))
    return roc_auc_score(y, p), *np.quantile(boots, [0.025, 0.975])


def classify(tag="oracle", seed=0):
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1, user_api="openmp")      # small data: 1 thread is ~50x faster here (16 #46)

    df = pd.read_csv(OUT / f"features_{tag}.csv")
    detector = tag != "oracle"
    sets = {"track (motion + shape)": TRACK, "motion only": KINEMATIC, "shape / flap only": SHAPE}
    if detector:
        sets.update({"detector votes only": DETECTOR, "detector + track": DETECTOR + TRACK})
    sets["CONFOUND probe: box size + position"] = PROBES
    main_set = "detector + track" if detector else "track (motion + shape)"
    rng = np.random.default_rng(seed)
    lines = [f"M2 classification on features_{tag}.csv: drone vs not-drone. AUC with 95% interval from "
             "resampling videos; balanced accuracy at p = 0.5; ECE before -> after isotonic calibration."]

    def fit_eval(d_train, d_test, cols):
        """Out-of-fold (d_test is None) or train/test. Returns calibrated p on the evaluated rows, raw p, y."""
        Xtr = d_train[cols].to_numpy(float)
        full = lambda d: np.column_stack([d[c].to_numpy(float) if c in cols else np.full(len(d), np.nan)
                                          for c in FEATURE_NAMES])
        feats = [c for c in cols if c in FEATURE_NAMES]
        if feats:
            clf = TrackClassifier(feats, seed=seed).fit(full(d_train), (d_train.cls == "drone").to_numpy(int),
                                                        d_train.video.to_numpy())
            if d_test is None:
                return clf.oof_prob, clf.oof_raw, clf
            return clf.predict_proba(full(d_test)), clf.predict_raw(full(d_test)), clf
        # probe columns are not in FEATURE_NAMES: plain out-of-fold model
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import StratifiedGroupKFold
        y = (d_train.cls == "drone").to_numpy(int)
        p = np.zeros(len(d_train))
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xtr, y, d_train.video):
            m = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed).fit(Xtr[a], y[a])
            p[b] = m.predict_proba(Xtr[b])[:, 1]
        if d_test is not None:
            m = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed).fit(Xtr, y)
            return m.predict_proba(d_test[cols].to_numpy(float))[:, 1], None, None
        return p, p, None

    protocols = [("5-fold grouped CV, all videos", None)]
    if detector:
        protocols.append(("train on train videos, TEST on the detector's held-out val videos", "val"))
    for fps in sorted(df.fps.unique(), reverse=True):
        for win in ["1.0", "2.0", "4.0", "full"]:
            d = df[(df.fps == fps) & (df.window_s.astype(str) == win)].reset_index(drop=True)
            if d.empty:
                continue
            for pname, test_split in protocols:
                dtr = d if test_split is None else d[d.split == "train"].reset_index(drop=True)
                dte = None if test_split is None else d[d.split == test_split].reset_index(drop=True)
                ev = dtr if dte is None else dte
                y = (ev.cls == "drone").to_numpy(int)
                lines.append(f"\n{fps} fps, window {win} s, {pname}: {len(ev)} samples from "
                             f"{ev.video.nunique()} videos ({y.mean():.0%} drone)")
                for name, cols in sets.items():
                    cols = [c for c in cols if dtr[c].notna().any()]
                    p, raw, _ = fit_eval(dtr, dte, cols)
                    # AUC ranks the raw scores (per-fold calibrators would add ties); ECE and accuracy use calibrated p
                    auc, lo, hi = _auc_ci(y, p if raw is None else raw, ev.video.to_numpy(), rng)
                    cal = f"ECE {ece(raw, y):.3f} -> {ece(p, y):.3f}" if raw is not None and raw is not p else \
                        f"ECE {ece(p, y):.3f}"
                    lines.append(f"  {name:38s} AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]  "
                                 f"bal.acc {balanced_accuracy_score(y, p > 0.5):.3f}  {cal}")
    # The classifiers the pipeline can use: 2-s windows at 30 fps, all videos, calibrated. With real detections,
    # also a TRACK-ONLY one: detector class votes are perfect on Halmstad but fail on a new dataset (E15: 38% of
    # in-scope DUT drones called helicopter/airplane), so the track-only model is the out-of-domain fallback.
    d = df[(df.fps == 30) & (df.window_s.astype(str) == "2.0")].reset_index(drop=True)
    y = (d.cls == "drone").to_numpy(int)
    track_set = "track (motion + shape)"
    for set_name, suffix in [(main_set, "")] + ([(track_set, "_trackonly")] if main_set != track_set else []):
        cols = [c for c in sets[set_name] if d[c].notna().any()]
        p, raw, clf = fit_eval(d, None, cols)
        X = np.column_stack([d[c].to_numpy(float) if c in cols else np.full(len(d), np.nan) for c in FEATURE_NAMES])
        lab = clf.decide(p, X, d.duration_s.to_numpy(float))
        dec = lab != "unknown"
        acc = np.mean((lab[dec] == "drone") == (y[dec] == 1)) if dec.any() else float("nan")
        lines.append(f"\nPipeline classifier ({set_name}, 2-s windows, 30 fps; out-of-fold): ECE {ece(raw, y):.3f} -> "
                     f"{ece(p, y):.3f} after calibration; 'unknown' for {1 - dec.mean():.0%} of windows; "
                     f"accuracy on the decided ones {acc:.3f}")
        clf.save(OUT / f"track_clf_{tag}{suffix}.pkl")
        lines.append(f"saved {OUT / f'track_clf_{tag}{suffix}.pkl'}")
    text = "\n".join(lines)
    (OUT / f"classify_{tag}.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "classify"])
    ap.add_argument("--dets", help="detections json.gz from m2_detect_videos.py (default: oracle boxes)")
    ap.add_argument("--tag", default="oracle", help="features_<tag>.csv to classify")
    a = ap.parse_args()
    if a.stage == "extract":
        extract(a.dets)
    else:
        classify(a.tag)
