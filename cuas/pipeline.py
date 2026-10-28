"""
Per-frame M2 pipeline: detections -> tracker -> silhouettes -> track features -> calibrated classifier ->
time to collision -> rules floor. One object used by replay, the experiments and the live system.

    pipe = TrackPipeline(classifier=TrackClassifier.load("track_clf.pkl"))
    for t, frame in video:                                   # t = capture time in seconds
        dets = detector(frame)                               # [(x, y, w, h, cls, score), ...]
        for s in pipe.update(t, dets, gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)):
            print(s["track_id"], s["label"], s["drone_prob"], s["ttc_s"], s["response"])

Features are recomputed every `feature_every` updates of a track (they change slowly; Lomb-Scargle is the
costly part), over its last `window_s` seconds: train the classifier on windows of the same length.

Memory and per-frame cost are bounded: the default tracker keeps only the last few seconds of each track and
no finished tracks, so a track that sits on a lamp for hours costs the same as a new one.

Labels: by default a track's label is the classifier's label for its latest window. With `decision=` (a
cuas.decide.SequentialDecision) the label is instead the accumulated evidence over all its windows (E26). Each
state also carries the window's own label and whether the classifier was read this frame and could be trusted
(`window_label`, `p_fresh`, `p_usable`), so experiments can replay other decision rules on the same tracks.
"""
import numpy as np

from cuas.detector import nms
from cuas.rules import rules_floor
from cuas.track import Tracker
from cuas.track_features import FEATURE_NAMES, silhouette_at, track_features, ttc_expansion


class TrackPipeline:
    def __init__(self, tracker=None, classifier=None, window_s: float = 2.0, feature_every: int = 5,
                 in_zone=None, ttc_window_s: float = 2.0, decision=None):
        self.keep_s = max(window_s, ttc_window_s) + 0.5
        self.tracker = tracker or Tracker(history_s=self.keep_s, keep_finished=False)
        self.classifier = classifier
        self.window_s, self.feature_every, self.ttc_window_s = window_s, feature_every, ttc_window_s
        self.in_zone = in_zone              # optional callable(track_state) -> bool
        self.decision = decision            # optional cuas.decide.SequentialDecision
        self._sil = {}                      # track id -> {t: (area, contrast)}
        self._cache = {}                    # track id -> (updates seen, drone_prob, label)

    def update(self, t: float, dets, gray=None) -> list:
        # One object boxed under two classes (a drone also called "helicopter") must be one detection here.
        active = self.tracker.update(t, nms([tuple(d) for d in dets], class_aware=False))
        live = {tr.id for tr in self.tracker.tracks}
        for store in (self._sil, self._cache):
            for tid in [k for k in store if k not in live]:
                del store[tid]
                if self.decision is not None:
                    self.decision.drop(tid)
        for tr in self.tracker.tracks:
            if gray is not None and tr.t_last == t:
                r = tr.history[-1]
                sil = self._sil.setdefault(tr.id, {})
                sil[t] = silhouette_at(gray, (r[1] - r[3] / 2, r[2] - r[4] / 2, r[3], r[4]))
                for old in [k for k in sil if k < t - self.keep_s]:
                    del sil[old]
        return [self._state(tr, t) for tr in active]

    def _state(self, tr, t):
        rows = tr.rows()
        s = {"track_id": tr.id, "t": t, "box": tr.box, "n_obs": tr.hits,
             "duration_s": float(tr.t_last - tr.t0), "coasting": tr.t_last < t,
             "velocity_px_s": (float(tr.kf.x[2]), float(tr.kf.x[3])),
             "ttc_s": ttc_expansion(rows[:, 0], rows[:, 3], self.ttc_window_s)}
        seen, p, label = self._cache.get(tr.id, (0, None, "unknown"))
        fresh = usable = False
        if self.classifier is not None and (seen % self.feature_every == 0):
            recent = rows[rows[:, 0] >= t - self.window_s]
            if len(recent):
                sil = self._sil.get(tr.id, {})
                area = np.array([sil.get(r, (np.nan, np.nan))[0] for r in recent[:, 0]])
                bright = np.array([sil.get(r, (np.nan, np.nan))[1] for r in recent[:, 0]])
                f = track_features(recent, area, bright)
                X = np.array([[f[n] for n in FEATURE_NAMES]])
                p = float(self.classifier.predict_proba(X)[0])
                # The whole track's age counts for "too short to judge", not the (trimmed) window's.
                label = str(self.classifier.decide([p], X, [s["duration_s"]])[0])
                # Can this reading be trusted? Not on a track shorter than the classifier's minimum, nor with
                # features outside its training range (the same tests that make decide() say "unknown").
                ood = hasattr(self.classifier, "out_of_range") and                     self.classifier.out_of_range(X)[0] > self.classifier.max_out_of_range
                fresh, usable = True, s["duration_s"] >= getattr(self.classifier, "min_duration", 0.0) and not ood
        self._cache[tr.id] = (seen + 1, p, label)
        s["drone_prob"] = p                 # None until classified: the rules floor treats that as unknown
        s["window_label"], s["p_fresh"], s["p_usable"] = label, fresh, usable
        if self.decision is not None:
            label = self.decision.update(tr.id, t, p, usable, s["duration_s"]) if fresh else self.decision.label(tr.id)
        s["label"] = label
        s["in_protected_zone"] = bool(self.in_zone(s)) if self.in_zone else False
        s["response"] = rules_floor(s)
        return s
