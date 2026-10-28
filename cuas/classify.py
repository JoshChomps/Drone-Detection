"""
Calibrated drone-vs-not-drone classifier over track features, with an "unknown" output (milestone M2).

* Model: gradient-boosted trees (handles missing features, e.g. no flap estimate on a 1-s track).
* Calibration: the raw scores are mapped to probabilities by isotonic regression fitted on OUT-OF-FOLD
  predictions (cross-fitting, folds grouped by video), so the calibrator never sees scores the model was
  trained on. Rules and operators then get a probability that means what it says (12 §5, 14 §5: raw ECE ~0.1).
* "unknown" (open-set): a track is not labelled when it is too short, when p(drone) sits in the undecided
  band, or when its features lie outside what the model saw in training (the model's answer would be a guess).

    clf = TrackClassifier(features).fit(X, y, groups)   # X: rows of ALL FEATURE_NAMES; `features` picks the
                                                        # ones used (default all). y: 1 = drone; groups: video ids
    p = clf.predict_proba(X_new)
    labels = clf.decide(p, X_new, durations)       # "drone" / "not_drone" / "unknown"
    clf.save(path); TrackClassifier.load(path)
"""
import pickle

import numpy as np

from cuas.track_features import FEATURE_NAMES

# Feature groups. DETECTOR features only mean something with real detector output: on ground-truth boxes the
# "class" is the true label, so using them there would leak the answer.
KINEMATIC = ["speed_bl_med", "speed_bl_p90", "hover_frac", "straightness", "turn_rate_med", "accel_bl_p90", "vert_frac"]
SHAPE = ["size_cv", "aspect_med", "aspect_cv", "aspect_peak_hz", "aspect_peak_pow", "area_peak_hz", "area_peak_pow",
         "bright_peak_hz", "bright_peak_pow"]
TRACK = KINEMATIC + SHAPE
DETECTOR = ["score_mean", "cls_drone_frac"]


class TrackClassifier:
    def __init__(self, features=None, n_splits: int = 5, seed: int = 0, band=(0.3, 0.7), min_duration: float = 1.0,
                 max_out_of_range: float = 0.3):
        self.features = list(features or FEATURE_NAMES)
        self.n_splits, self.seed = n_splits, seed
        self.band, self.min_duration, self.max_out_of_range = band, min_duration, max_out_of_range

    def _model(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=self.seed)

    def _X(self, X):
        return np.asarray(X, float)[:, self._keep]

    def fit(self, X, y, groups):
        # Small data: sklearn's default of one OpenMP thread per core made each fit ~50x slower on this laptop
        # (3.1 s vs 0.06 s, 16 threads vs 1; 16 #46). One thread is fastest here.
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api="openmp"):
            return self._fit(X, y, groups)

    def _fit(self, X, y, groups):
        from sklearn.isotonic import IsotonicRegression
        from sklearn.model_selection import StratifiedGroupKFold
        X, y = np.asarray(X, float), np.asarray(y, int)
        self.prior = float(y.mean())                     # drone share the calibrated p carries (cuas.decide)
        cols = np.array([FEATURE_NAMES.index(f) for f in self.features])
        self._keep = cols[np.isfinite(X[:, cols]).any(axis=0)]      # drop features never measured (all NaN)
        Xk = X[:, self._keep]
        oof = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.seed).split(Xk, y, groups):
            oof[te] = self._model().fit(Xk[tr], y[tr]).predict_proba(Xk[te])[:, 1]
        self.oof_raw = oof
        iso = lambda: IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")   # noqa: E731
        # Honest calibrated out-of-fold probabilities: the calibrator for each fold is fitted on the OTHER folds'
        # scores (scoring an isotonic fit on its own training scores gives ECE ~0 by construction; 16 #46).
        self.oof_prob = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.seed + 1).split(Xk, y, groups):
            self.oof_prob[te] = iso().fit(oof[tr], y[tr]).predict(oof[te])
        self.calibrator = iso().fit(oof, y)              # deployed calibrator: fitted on all out-of-fold scores
        self.model = self._model().fit(Xk, y)
        self.lo, self.hi = np.nanpercentile(Xk, 1, axis=0), np.nanpercentile(Xk, 99, axis=0)
        return self

    def predict_raw(self, X):
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api="openmp"):
            return self.model.predict_proba(self._X(X))[:, 1]

    def predict_proba(self, X):
        return self.calibrator.predict(self.predict_raw(X))

    def drone_share(self) -> float:
        """Drone share of the training windows, which the calibrated p carries (cuas.decide takes it out again).
        Classifiers saved before 2026-10-05 lack `prior`; it is recovered exactly from the calibrator, because an
        isotonic fit preserves the mean of its targets."""
        prior = getattr(self, "prior", None)
        return float(prior) if prior is not None else float(np.mean(self.calibrator.predict(self.oof_raw)))

    def out_of_range(self, X):
        """Share of each row's measured features that fall outside the training 1st-99th percentile."""
        Xk = self._X(X)
        measured = np.isfinite(Xk)
        outside = measured & ((Xk < self.lo) | (Xk > self.hi))
        return outside.sum(axis=1) / np.maximum(measured.sum(axis=1), 1)

    def decide(self, p, X, durations):
        p, durations = np.asarray(p, float), np.asarray(durations, float)
        ood = self.out_of_range(X) > self.max_out_of_range
        out = np.where(p >= self.band[1], "drone", np.where(p <= self.band[0], "not_drone", "unknown"))
        out[(durations < self.min_duration) | ood] = "unknown"
        return out

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error: bin-weighted gap between predicted p(drone) and the observed drone share."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))
