"""
E26: the sequential track decision (cuas/decide.py) against the hold rule, on the same tracks.

Question: at the same rate of false drone declarations per hour, does accumulating the classifier's evidence declare
more drone videos, sooner, than "drone for h seconds without a break" (E20)? E20 showed where it matters: with the
track-only classifier (the fallback outside Halmstad-like footage) the hold rule gives 71 false declarations per hour
at h = 0 and 10 per hour at h = 1 s, where it misses 5 of 23 drone videos. With detector features the in-domain
rate is already 0, so that classifier is reported but cannot discriminate.

Protocol (no tuning on the reported videos):
  1. Cross-fitted TRAIN videos: 5 folds by video, stratified by class. Each fold's tracks are classified by a track
     classifier fitted on the OTHER folds' train windows. Both rules are replayed on those tracks over a grid of
     settings; per rule family the operating point is chosen there: the most drone videos declared with at most
     BUDGET false declarations per hour (ties: the shorter median delay). BUDGET is provisional: doc 13 sets no
     per-hour requirement yet.
  2. VAL videos (held out, as in E20; classifier fitted on all train windows): the chosen settings are scored once.
     The whole val grid is printed too, marked as not used for any choice.
Detections: the cached E12 detector run on all 285 Halmstad videos. The detector trained on the TRAIN videos, so its
boxes there are cleaner than on new footage; the track classifier's probabilities there are honest (cross-fitted).

Each video is decoded once; the pipeline runs with its default per-window labels and records every state, and both
rules are replayed offline on the recorded states (the SPRT through cuas.decide.SequentialDecision, the same code
`python -m cuas` would use). The hold-rule replay on val must reproduce E20's numbers; this is checked.

    C:\\cuas_venv\\Scripts\\python experiments/e26_sequential.py [--tag dets_E12]        (~15-20 min, CPU)
Output: C:/cuas_data/m2/e26_sequential_<tag>.txt / .json
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuas.classify import DETECTOR, TRACK, TrackClassifier  # noqa: E402
from cuas.datasets.halmstad import NAME_RE, split_videos  # noqa: E402
from cuas.decide import SequentialDecision  # noqa: E402
from cuas.eval import iou  # noqa: E402
from cuas.pipeline import TrackPipeline  # noqa: E402
from cuas.track_features import FEATURE_NAMES  # noqa: E402
from e20_system_eval import OUT, SRC, poisson_upper  # noqa: E402

HOLDS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
ACCEPTS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
BUDGET = 10.0           # false declarations per hour of non-drone video (provisional, see the docstring)
FOLDS = 5
CLASSIFIERS = {"track only": TRACK, "detector+track": DETECTOR + TRACK}


def fit(tag, features, videos):
    """Track classifier on 2-s windows at 30 fps of the given TRAIN videos (as e20_system_eval.fit)."""
    import pandas as pd
    d = pd.read_csv(OUT / f"features_{tag}.csv")
    d = d[(d.fps == 30) & (d.window_s.astype(str) == "2.0") & (d.split == "train") & d.video.isin(videos)]
    d = d.reset_index(drop=True)
    X = np.column_stack([d[c].to_numpy(float) if c in features else np.full(len(d), np.nan) for c in FEATURE_NAMES])
    keep = [f for f in features if d[f].notna().any()]
    return TrackClassifier(keep).fit(X, (d.cls == "drone").to_numpy(int), d.video.to_numpy())


def record(vid, clfs, dets_all):
    """One video through the pipeline, once per classifier. Returns its class, seconds, first labelled time, and per
    classifier {track id: [(t, window_label, p, fresh, usable, age, on the labelled object), ...]}."""
    import cv2
    from mcos_decoder import load_groundtruth
    labels = load_groundtruth(str(vid.with_name(vid.stem + "_LABELS.mat")))
    by_frame = {}
    for fr, x, y, w, h, c, s in dets_all[vid.stem]["dets"]:
        by_frame.setdefault(int(fr), []).append((x, y, w, h, int(c), s))
    pipes = {name: TrackPipeline(classifier=clf) for name, clf in clfs.items()}
    tracks = {name: {} for name in clfs}
    cap = cv2.VideoCapture(str(vid))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n, first_gt = 0, None
    while n < len(labels):                              # past the last label nothing can be scored (14 section 5)
        ok, frame = cap.read()
        if not ok:
            break
        t, gt = n / fps, labels[n]
        if gt is not None and first_gt is None:
            first_gt = t
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for name, pipe in pipes.items():
            for s in pipe.update(t, by_frame.get(n, []), gray):
                on = gt is not None and iou(s["box"], gt) >= 0.1
                tracks[name].setdefault(s["track_id"], []).append(
                    (t, s["window_label"], s["drone_prob"], s["p_fresh"], s["p_usable"], s["duration_s"], on))
        n += 1
    cap.release()
    return {"video": vid.stem, "cls": NAME_RE.search(vid.name).group(1).lower(), "seconds": n / fps,
            "first_gt": first_gt, "tracks": tracks}


def declare_hold(states, h, prior):
    """(time, on target) of the first declaration under 'drone for h s without a break', or None (E20's rule)."""
    since = None
    for t, wl, p, fresh, usable, age, on in states:
        if wl != "drone":
            since = None
            continue
        since = t if since is None else since
        if t - since >= h:
            return t, on
    return None


def declare_sprt(states, accept, prior):
    """(time, on target) of the first declaration by accumulated evidence, or None."""
    dec = SequentialDecision(accept=accept, prior=prior)
    for t, wl, p, fresh, usable, age, on in states:
        label = dec.update(0, t, p, usable, age) if fresh else dec.label(0)
        if label == "drone":
            return t, on
    return None


def score(recs, name, rule, setting, priors):
    """E20's numbers for one rule and setting over a set of recorded videos."""
    non = [r for r in recs if r["cls"] != "drone"]
    drone = [r for r in recs if r["cls"] == "drone"]
    hours = sum(r["seconds"] for r in non) / 3600
    false_non, delays, false_drone = 0, [], 0
    for r in recs:
        decl = [rule(st, setting, priors[r["video"]]) for st in r["tracks"][name].values()]
        decl = [d for d in decl if d is not None]
        if r["cls"] != "drone":
            false_non += len(decl)
            continue
        on = [t for t, hit in decl if hit]
        false_drone += sum(not hit for _, hit in decl)
        if on and r["first_gt"] is not None:
            delays.append(min(on) - r["first_gt"])
    return {"false": false_non, "per_hour": false_non / hours, "upper95": poisson_upper(false_non, hours),
            "declared": len(delays), "drone_videos": len(drone),
            "delay_med": float(np.median(delays)) if delays else None,
            "delay_p90": float(np.percentile(delays, 90)) if delays else None, "false_on_drone_videos": false_drone,
            "hours": hours}


def grid(recs, name, priors):
    return {("hold", h): score(recs, name, declare_hold, h, priors) for h in HOLDS} | \
           {("sprt", a): score(recs, name, declare_sprt, a, priors) for a in ACCEPTS}


def choose(results, family):
    """The setting with the most drone videos declared within BUDGET; ties to the shorter median delay."""
    ok = [(k, v) for k, v in results.items() if k[0] == family and v["per_hour"] <= BUDGET]
    if not ok:
        return None
    return max(ok, key=lambda kv: (kv[1]["declared"], -(kv[1]["delay_med"] or 99)))[0]


def fmt(k, v):
    rule = f"hold {k[1]:.2f} s" if k[0] == "hold" else f"evidence {k[1]:.1f}"
    d = f"{v['delay_med']:.2f} / {v['delay_p90']:.2f}" if v["delay_med"] is not None else "-"
    return (f"  {rule:16s} {v['false']:4d} = {v['per_hour']:5.1f} /h (95% up {v['upper95']:5.1f})   "
            f"{v['declared']:3d}/{v['drone_videos']:<3d} {d:>13s}   {v['false_on_drone_videos']:4d}")


def main(tag):
    with gzip.open(OUT / f"{tag}.json.gz", "rt") as f:
        dets_all = json.load(f)
    split = split_videos(sorted(SRC.glob("V_*.mp4")))   # the full list, exactly as E20 and the feature table split it
    videos = sorted(v for v in split if v.stem in dets_all)
    train = [v for v in videos if split[v] == "train"]
    val = [v for v in videos if split[v] == "val"]
    # Folds of the TRAIN videos, stratified by class (deterministic)
    rng, fold_of = np.random.default_rng(26), {}
    for cls in sorted({NAME_RE.search(v.name).group(1) for v in train}):
        vs = [v for v in train if NAME_RE.search(v.name).group(1) == cls]
        rng.shuffle(vs)
        for i, v in enumerate(vs):
            fold_of[v] = i % FOLDS

    recs = {"train (cross-fitted)": [], "val": []}
    priors = {name: {} for name in CLASSIFIERS}
    for k in range(FOLDS):
        fit_on = [v.stem for v in train if fold_of[v] != k]
        clfs = {name: fit(tag, feats, fit_on) for name, feats in CLASSIFIERS.items()}
        for v in [v for v in train if fold_of[v] == k]:
            recs["train (cross-fitted)"].append(record(v, clfs, dets_all))
            for name, clf in clfs.items():
                priors[name][v.stem] = clf.prior
        print(f"fold {k + 1}/{FOLDS}: {sum(fold_of[v] == k for v in train)} videos", flush=True)
    clfs = {name: fit(tag, feats, [v.stem for v in train]) for name, feats in CLASSIFIERS.items()}
    for v in val:
        recs["val"].append(record(v, clfs, dets_all))
        for name, clf in clfs.items():
            priors[name][v.stem] = clf.prior
    print(f"val: {len(val)} videos", flush=True)

    head = (f"  {'rule':16s} {'false declarations, non-drone videos':>38s}   {'drone videos':>7s} "
            f"{'delay med/p90 s':>13s}   {'false on drone videos'}")
    lines, out = [f"E26 sequential decision against the hold rule ({tag}). Operating points chosen on cross-fitted "
                  f"TRAIN videos: most drone videos declared with at most {BUDGET:g} false declarations per hour."], {}
    for name in CLASSIFIERS:
        res = {split_name: grid(r, name, priors[name]) for split_name, r in recs.items()}
        tr, va = res["train (cross-fitted)"], res["val"]
        hours = {s: next(iter(v.values()))["hours"] * 60 for s, v in res.items()}
        picks = {fam: choose(tr, fam) for fam in ("hold", "sprt")}
        lines += ["", f"=== {name} classifier ===",
                  f"Chosen on train ({hours['train (cross-fitted)']:.1f} min of non-drone video), scored once on val "
                  f"({hours['val']:.1f} min):", head]
        for fam, k in picks.items():
            if k is None:
                lines.append(f"  {fam}: no setting within the budget on train")
                continue
            lines += [fmt(k, tr[k]) + "   <- train", fmt(k, va[k]) + "   <- VAL"]
        for split_name, r in res.items():
            lines += ["", f"Full grid, {split_name}" + (" (not used for any choice)" if split_name == "val" else "")
                      + ":", head] + [fmt(k, v) for k, v in r.items()]
        out[name] = {"picks": {fam: list(k) if k else None for fam, k in picks.items()},
                     "grid": {s: {f"{k[0]} {k[1]}": v for k, v in r.items()} for s, r in res.items()}}
    # The replay must reproduce E20 (same classifier, same tracks, same rule)
    e20 = OUT / f"e20_system_{tag}.json"
    if e20.exists():
        ref = json.loads(e20.read_text())["summary"]
        for name in CLASSIFIERS:
            for h in (0.0, 0.5, 1.0):
                got, want = out[name]["grid"]["val"][f"hold {h}"], ref[f"{name}, hold {h}"]
                same = (got["false"], got["declared"]) == (want["false_declarations"], want["drone_videos_declared"])
                lines.append(f"check against E20, {name}, hold {h}: " + ("reproduced" if same else
                             f"DIFFERS (E26 {got['false']} false / {got['declared']} declared, E20 "
                             f"{want['false_declarations']} / {want['drone_videos_declared']})"))
    text = "\n".join(lines)
    (OUT / f"e26_sequential_{tag}.txt").write_text(text, encoding="utf-8")
    (OUT / f"e26_sequential_{tag}.json").write_text(json.dumps(out, indent=1))
    print("\n" + text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dets_E12", help="detections file <tag>.json.gz and features_<tag>.csv in C:/cuas_data/m2")
    main(ap.parse_args().tag)
