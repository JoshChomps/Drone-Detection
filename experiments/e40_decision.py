"""
E40: the decision layer on unseen footage. The field test (16 #93) found the system tracks the drone most of the
time on new datasets but often calls it "not drone": the track classifier was fitted on Halmstad tracks only.

The field test saves every classifier window of every track (time, track age, feature row) and every track's
per-frame status against the labels. Tracking never depends on the classifier, so any classifier and decision rule
can be REPLAYED on the same tracks offline, in seconds, with the same scoring (field_test.rescore). This script:

  1. builds labelled windows from the TRAINING SOURCES only (field_test sets halmstad_train, antiuav_trainval, run
     with the same frozen detector, in D:\\cuas_data\\field_test\\<model>_train):
       drone   = a window whose 2 s are mostly ON the labelled object, in a video whose object is a drone;
       not     = mostly on a labelled bird / airplane / helicopter, or mostly OFF the labelled drone (clutter, night
                 lights) in an Anti-UAV video (one drone a video; in Halmstad "no label" is not "no drone", so
                 off-target Halmstad tracks are left out);
  2. fits candidate classifiers (cuas.classify.TrackClassifier; folds and calibration grouped by video);
  3. chooses the evidence threshold on the training sources by cross-fitted replay (out-of-fold p), so no test set
     touches a choice;
  4. replays each candidate on the test sets and prints the field-test table per condition beside v1's.

    C:\\cuas_venv\\Scripts\\python experiments/e40_decision.py [--model cuas_v1] [--save cuas_v2]

The replay of v1's own classifier must reproduce v1's declarations exactly; the script checks that first.
"""
import argparse
import json
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import field_test as ft  # noqa: E402
from cuas.classify import DETECTOR, TRACK, TrackClassifier  # noqa: E402
from cuas.decide import SequentialDecision  # noqa: E402
from cuas.track_features import FEATURE_NAMES  # noqa: E402

WIN_S = 2.0
NF = len(FEATURE_NAMES)


def load(folder):
    return [json.loads(p.read_text()) for p in sorted(Path(folder).glob("*/*.json"))]


def windows(r):
    """Per track: list of (t, age, x) in time order."""
    out = {}
    for tid, k in r["track_status"].items():
        out[tid] = [(w[0], w[1], np.array([np.nan if v is None else v for v in w[4:4 + NF]], float))
                    for w in k.get("windows", [])]
    return out


def window_label(r, k, t):
    """1 drone, 0 not drone, None unclear, for a window ending at time t on track k."""
    f = int(round(t * r["fps"])) - k["first_frame"]
    s = k["status"][max(0, f - int(WIN_S * r["fps"]) + 1):f + 1]
    on, off = s.count("o"), s.count("x")
    if on + off < 0.5 * len(s) or not s:
        return None
    if on >= 0.8 * (on + off):
        return int(r.get("obj_class", "drone") == "drone")
    if off >= 0.8 * (on + off) and r["set"] != "halmstad_train":
        return 0
    return None


def training_windows(rs):
    X, y, g = [], [], []
    for r in rs:
        ws = windows(r)
        for tid, k in r["track_status"].items():
            for t, age, x in ws[tid]:
                lab = window_label(r, k, t)
                if lab is not None and age >= 1.0:
                    X.append(x), y.append(lab), g.append(f"{r['set']}/{r['clip']}")
    return np.array(X), np.array(y), np.array(g)


def replay(r, clf, accept, p_of=None):
    """Declarations of every track of clip result r under classifier clf and evidence threshold `accept`.
    p_of(clip_key, tid, i) -> p overrides the classifier (cross-fitted replay). Returns a rescored copy of r."""
    dec = SequentialDecision(accept=accept, prior=clf.drone_share())
    r = dict(r)
    declared = {}
    ws = windows(r)
    for tid, k in r["track_status"].items():
        if not ws[tid]:
            continue
        X = np.array([x for _, _, x in ws[tid]])
        p = clf.predict_proba(X) if p_of is None else np.array([p_of(tid, i) for i in range(len(X))])
        ood = clf.out_of_range(X) > clf.max_out_of_range
        for i, (t, age, _) in enumerate(ws[tid]):
            usable = age >= clf.min_duration and not ood[i]
            if dec.update(tid, t, p[i], usable, age) == "drone":
                declared[tid] = t
                break
    ts = {}
    for tid, k in r["track_status"].items():
        k = dict(k)
        k["declared_frame"] = int(round(declared[tid] * r["fps"])) if tid in declared else None
        ts[tid] = k
    old = r.get("_declared", {})
    r["track_status"] = ts
    r["_declared"] = {tid: {"t": round(t, 2), "crop": old.get(tid, {}).get("crop"),
                            "max_w": old.get(tid, {}).get("max_w"), "duration_s": None}
                      for tid, t in declared.items()}
    return ft.rescore(r)


def table(rs, title):
    tags = json.loads(ft.CONDITIONS.read_text())
    tags = {k: v for k, v in tags.items() if not k.startswith("_")}
    groups = defaultdict(list)
    for r in rs:
        for c in ft.conditions_of(r, tags):
            groups[(r["set"], c)].append(r)
            groups[("ALL", c)].append(r)
        groups[("ALL", "all")].append(r)
    lines = [title, f"{'set':14s} {'condition':10s} {'clips':>5s} {'hours':>6s} {'declared':>9s} {'med delay':>9s} "
                    f"{'false':>5s} {'false/h':>7s} {'unver':>5s}"]
    for (s, c), g in sorted(groups.items(), key=lambda kv: (kv[0][0] != "ALL", kv[0])):
        hours = sum(r["minutes"] for r in g) / 60
        lab = [r for r in g if r["drone_declared"] is not None]
        delays = [r["delay_s"] for r in lab if r["delay_s"] is not None]
        false = sum(len(r["declarations"]["off_target"]) for r in g)
        unver = sum(len(r["declarations"]["unverified"]) for r in g)
        decl = f"{sum(r['drone_declared'] for r in lab)}/{len(lab)}" if lab else "-"
        lines.append(f"{s:14s} {c:10s} {len(g):5d} {hours:6.2f} {decl:>9s} "
                     f"{(f'{np.median(delays):.1f}' if delays else '-'):>9s} {false:5d} "
                     f"{(f'{false / hours:.0f}' if hours else '-'):>7s} {unver:5d}")
    return "\n".join(lines)


def summary(rs):
    lab = [r for r in rs if r["drone_declared"] is not None]
    hours = sum(r["minutes"] for r in rs) / 60
    false = sum(len(r["declarations"]["off_target"]) for r in rs)
    return sum(r["drone_declared"] for r in lab) / max(len(lab), 1), false / hours if hours else 0.0, false


def cross_fitted_p(clf_factory, X, y, g, rs_train):
    """Out-of-fold calibrated p for every training-source window (folds by video), keyed (clip, tid, i)."""
    from sklearn.model_selection import StratifiedGroupKFold
    clip_folds = {}
    keys = sorted(set(g))
    labelled_clips = {f"{r['set']}/{r['clip']}" for r in rs_train}
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g))
    models = []
    for k, (tr, te) in enumerate(folds):
        models.append(clf_factory().fit(X[tr], y[tr], g[tr]))
        for c in set(g[te]):
            clip_folds[c] = k
    rng = np.random.default_rng(0)
    for c in labelled_clips - set(keys):             # clips with no labelled window: any fold model is out-of-fold
        clip_folds[c] = int(rng.integers(len(folds)))
    return models, clip_folds


def choose_accept(clf_factory, X, y, g, rs_train, grid=(2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0), max_false_h=None):
    """Evidence threshold chosen on the training sources by cross-fitted replay: the lowest threshold whose false
    declaration rate is at most `max_false_h` (default: v1's rate on the same sources)."""
    models, clip_folds = cross_fitted_p(clf_factory, X, y, g, rs_train)
    rows = []
    for a in grid:
        rep = []
        for r in rs_train:
            m = models[clip_folds[f"{r['set']}/{r['clip']}"]]
            rep.append(replay(r, m, a))
        rows.append((a, *summary(rep)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cuas_v1")
    ap.add_argument("--save", help="write a new bundle (copy of --model with the chosen classifier and threshold)")
    a = ap.parse_args()
    base = ft.load_model(a.model)
    v1clf = TrackClassifier.load(str(base["dir"] / base["clf"]))
    test = [r for r in load(ft.OUT / a.model) if "windows" in next(iter(r["track_status"].values()), {"windows": []})]
    train = load(ft.OUT / f"{a.model}_train")
    out = []

    class Out(list):
        def append(self, line):
            print(line, flush=True)
            super().append(line)

        def __iadd__(self, lines):
            for line in lines:
                self.append(line)
            return self
    out = Out()
    out.append(f"E40, model {a.model}: {len(test)} test clips, {len(train)} training-source clips")

    # 1. replaying v1 must reproduce v1
    mism = 0
    for r in test:
        rr = replay(r, v1clf, base["evidence"])
        orig = {tid for tid, k in r["track_status"].items() if k["declared_frame"] is not None}
        new = {tid for tid, k in rr["track_status"].items() if k["declared_frame"] is not None}
        mism += orig != new
    out.append(f"replay check (v1 classifier, evidence {base['evidence']}): {mism} of {len(test)} clips differ")

    # 2. training windows
    X, y, g = training_windows(train)
    by_set = defaultdict(lambda: [0, 0])
    for yi, gi in zip(y, g):
        by_set[gi.split("/")[0]][yi] += 1
    out.append(f"training windows: {len(y)} ({int(y.sum())} drone) from {len(set(g))} videos; "
               + ", ".join(f"{s}: {d} drone / {n} not" for s, (n, d) in sorted(by_set.items())))

    cands = {"track": lambda: TrackClassifier(TRACK), "track+det": lambda: TrackClassifier(TRACK + DETECTOR)}
    v1_train = [replay(r, v1clf, base["evidence"]) for r in train]
    rate_v1, fh_v1, _ = summary(v1_train)
    out.append(f"v1 on the training sources: declared {rate_v1:.2f} of drone clips, {fh_v1:.0f} false/h")
    out.append("\n" + table([replay(r, v1clf, base["evidence"]) for r in test], "=== v1 (replayed) on the test sets ==="))
    chosen = {}
    for name, make in cands.items():
        rows = choose_accept(make, X, y, g, train)
        out.append(f"\n{name}: cross-fitted on the training sources (accept, declared share, false/h, false):")
        out += [f"  {acc:5.1f}  {d:.2f}  {fh:6.1f}  {n}" for acc, d, fh, n in rows]
        ok = [row for row in rows if row[2] <= max(fh_v1, 1e-9)]
        acc = ok[0][0] if ok else rows[-1][0]
        clf = make().fit(X, y, g)
        chosen[name] = (clf, acc)
        out.append("\n" + table([replay(r, clf, acc) for r in test],
                                f"=== {name}, accept {acc} (chosen on training sources) on the test sets ==="))
    text = "\n".join(out)
    (ft.OUT / a.model / "e40_decision.txt").write_text(text, encoding="utf-8")
    if a.save:
        name = a.save
        clf, acc = chosen[next(iter(chosen))]
        d = ft.MODELS / name
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(base["dir"] / base["weights"], d / base["weights"])
        with open(d / "track_clf.pkl", "wb") as f:
            pickle.dump(clf, f)
        m = {k: v for k, v in base.items() if k != "dir"}
        m.update(name=name, clf="track_clf.pkl", evidence=acc,
                 clf_from=f"e40_decision.py on {a.model}_train (halmstad_train + antiuav_trainval windows)")
        (d / "model.json").write_text(json.dumps(m, indent=2))
        print(f"saved {d}")


if __name__ == "__main__":
    main()
