"""
Multi-target tracker for a fixed camera (milestone M2, step 1): constant-velocity Kalman filter in image
coordinates + ByteTrack-style two-stage association + a time-based track life cycle.

Design choices (doc 18 §3, doc 09 §2-3):
* Association gates on the Kalman-PREDICTED position (Mahalanobis distance), not on box overlap. At 10 fps a
  crossing FPV at 100 m jumps ~400 px between frames, so IoU matching would drop it.
* Everything runs on timestamps (seconds at exposure), not frame counts: dropped frames and variable rates
  are handled, and ages/timeouts are in seconds.
* Two stages (ByteTrack): high-score detections first, then low-score ones for tracks still unmatched, so a
  distant drone whose score flickers keeps its track. Only high-score detections start new tracks.
* For a fixed camera, pixel position is bearing. A PTZ or a moving camera needs a world-frame state (doc 09).

Boxes are (x, y, w, h[, cls[, score]]) in pixels, top-left origin, as everywhere in `cuas`.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

CHI2_2DOF_99 = 9.21     # gate: 99% of true matches have Mahalanobis d^2 below this (2 measured coordinates)


class KalmanCV:
    """Nearly-constant-velocity filter on (x, y, vx, vy) with continuous white-noise acceleration.

    q is the acceleration power spectral density in px^2/s^3; Q = q [[dt^3/3, dt^2/2], [dt^2/2, dt]] per axis
    (the exact discretisation, doc 09 §3.2). R is the measurement variance of the box centre in px^2.
    """

    def __init__(self, x, y, pos_var: float, vel_var: float):
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([pos_var, pos_var, vel_var, vel_var])

    @staticmethod
    def _F_Q(dt: float, q: float):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        a, b, c = dt ** 3 / 3, dt ** 2 / 2, dt
        Q = q * np.array([[a, 0, b, 0], [0, a, 0, b], [b, 0, c, 0], [0, b, 0, c]])
        return F, Q

    def predicted(self, dt: float, q: float):
        """(state, covariance) predicted dt seconds ahead, without changing the filter."""
        F, Q = self._F_Q(dt, q)
        return F @ self.x, F @ self.P @ F.T + Q

    def predict(self, dt: float, q: float):
        self.x, self.P = self.predicted(dt, q)

    def gate(self, z, r_var: float, x=None, P=None) -> float:
        """Squared Mahalanobis distance of measurement z = (cx, cy) from the (given or current) state."""
        x = self.x if x is None else x
        P = self.P if P is None else P
        y = np.asarray(z, float) - x[:2]
        S = P[:2, :2] + np.eye(2) * r_var
        return float(y @ np.linalg.solve(S, y))

    def update(self, z, r_var: float):
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        S = H @ self.P @ H.T + np.eye(2) * r_var
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z, float) - H @ self.x)
        # Joseph form keeps P symmetric and positive definite (doc 05 lesson).
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ (np.eye(2) * r_var) @ K.T


class IMMFilter:
    """Interacting Multiple Model filter (doc 09 §3.3): a nearly-constant-velocity model and a
    nearly-constant-acceleration model on one shared state (x, y, vx, vy, ax, ay), mixed every step.

    Same interface as KalmanCV, so the tracker can use either. `q` (passed to predict) is the CV model's
    acceleration PSD (px^2/s^3); the CA model uses its own jerk PSD `q_jerk` (px^2/s^5). `mu` holds the model
    probabilities [CV, CA]; mu[1] rising means the filter thinks the target is manoeuvring.
    """

    def __init__(self, x, y, pos_var: float, vel_var: float, acc_var: float = 1e6, q_jerk: float = 1e7,
                 p_stay: float = 0.95):
        x0 = np.array([x, y, 0.0, 0.0, 0.0, 0.0])
        P0 = np.diag([pos_var, pos_var, vel_var, vel_var, acc_var, acc_var])
        self.xs, self.Ps = [x0.copy(), x0.copy()], [P0.copy(), P0.copy()]
        self.mu = np.array([0.9, 0.1])
        self.Pi = np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]])
        self.q_jerk = q_jerk
        self._pred = None
        self._combine(self.xs, self.Ps, self.mu)

    def _models(self, dt: float, q: float):
        I2, Z2 = np.eye(2), np.zeros((2, 2))
        # CV: acceleration is forced to zero and kept near zero (small Q), so both models share one state.
        F_cv = np.block([[I2, dt * I2, Z2], [Z2, I2, Z2], [Z2, Z2, Z2]])
        a, b, c = dt ** 3 / 3, dt ** 2 / 2, dt
        Q_cv = np.zeros((6, 6))
        Q_cv[:4, :4] = q * np.block([[a * I2, b * I2], [b * I2, c * I2]])
        Q_cv[4:, 4:] = 1e-6 * I2
        F_ca = np.block([[I2, dt * I2, dt ** 2 / 2 * I2], [Z2, I2, dt * I2], [Z2, Z2, I2]])
        Q_ca = self.q_jerk * np.block([
            [dt ** 5 / 20 * I2, dt ** 4 / 8 * I2, dt ** 3 / 6 * I2],
            [dt ** 4 / 8 * I2, dt ** 3 / 3 * I2, dt ** 2 / 2 * I2],
            [dt ** 3 / 6 * I2, dt ** 2 / 2 * I2, dt * I2]])
        return [(F_cv, Q_cv), (F_ca, Q_ca)]

    def _combine(self, xs, Ps, mu):
        self.x = sum(m * x for m, x in zip(mu, xs))
        self.P = sum(m * (P + np.outer(x - self.x, x - self.x)) for m, x, P in zip(mu, xs, Ps))

    def _predict_all(self, dt: float, q: float):
        c = self.Pi.T @ self.mu                               # predicted model probabilities
        w = (self.Pi * self.mu[:, None]) / c[None, :]         # mixing weights mu_{i|j}
        xs, Ps = [], []
        for j, (F, Q) in enumerate(self._models(dt, q)):
            x0 = sum(w[i, j] * self.xs[i] for i in range(2))
            P0 = sum(w[i, j] * (self.Ps[i] + np.outer(self.xs[i] - x0, self.xs[i] - x0)) for i in range(2))
            xs.append(F @ x0)
            Ps.append(F @ P0 @ F.T + Q)
        return xs, Ps, c

    def predicted(self, dt: float, q: float):
        xs, Ps, c = self._predict_all(dt, q)
        x = sum(m * v for m, v in zip(c, xs))
        P = sum(m * (Pj + np.outer(v - x, v - x)) for m, v, Pj in zip(c, xs, Ps))
        return x, P

    def predict(self, dt: float, q: float):
        self._pred = self._predict_all(dt, q)
        self._combine(self._pred[0], self._pred[1], self._pred[2])

    def gate(self, z, r_var: float, x=None, P=None) -> float:
        x = self.x if x is None else x
        P = self.P if P is None else P
        y = np.asarray(z, float) - x[:2]
        return float(y @ np.linalg.solve(P[:2, :2] + np.eye(2) * r_var, y))

    def update(self, z, r_var: float):
        xs, Ps, c = self._pred if self._pred is not None else (self.xs, self.Ps, self.mu)
        H = np.zeros((2, 6))
        H[0, 0] = H[1, 1] = 1.0
        R = np.eye(2) * r_var
        new_x, new_P, lik = [], [], []
        for x, P in zip(xs, Ps):
            S = H @ P @ H.T + R
            y = np.asarray(z, float) - H @ x
            K = P @ H.T @ np.linalg.inv(S)
            I_KH = np.eye(6) - K @ H
            new_x.append(x + K @ y)
            new_P.append(I_KH @ P @ I_KH.T + K @ R @ K.T)
            lik.append(np.exp(-0.5 * y @ np.linalg.solve(S, y)) / (2 * np.pi * np.sqrt(np.linalg.det(S))))
        mu = c * np.array(lik)
        self.mu = mu / mu.sum() if mu.sum() > 0 else c
        self.xs, self.Ps, self._pred = new_x, new_P, None
        self._combine(self.xs, self.Ps, self.mu)


@dataclass
class Track:
    id: int
    kf: KalmanCV
    t_last: float
    w: float
    h: float
    cls: int = 0
    hits: int = 1
    confirmed: bool = False
    t0: float = 0.0     # time of the first detection (history may be trimmed, so it can't be read from there)
    # One row per matched detection: t, cx, cy, w, h, score, cls, filtered x, y, vx, vy.
    history: list = field(default_factory=list)

    @property
    def box(self):
        cx, cy = self.kf.x[:2]
        return (cx - self.w / 2, cy - self.h / 2, self.w, self.h)

    def rows(self) -> np.ndarray:
        return np.array(self.history, float).reshape(-1, 11)


class Tracker:
    """
    tracker = Tracker()
    for t, dets in stream:                # t in seconds; dets = [(x, y, w, h, cls, score), ...]
        active = tracker.update(t, dets)  # confirmed tracks updated or coasting at time t

    Parameters (pixels and seconds):
      high / low     score thresholds for stage 1 and stage 2 (low-score detections never start tracks)
      q              acceleration PSD, px^2/s^3. Default 2e5: a velocity random walk of ~450 px/s over 1 s
      r_frac, r_min  centre noise std = max(r_min, r_frac * box width): loose boxes on bigger targets
      v0             prior velocity std of a new track (px/s), wide enough for a fast crosser's 2nd frame
      n_confirm      matched detections needed to confirm a track
      max_coast      seconds a confirmed track survives without a detection; tentative ones get tentative_coast
      motion         "cv" (KalmanCV) or "imm" (IMMFilter: CV + constant-acceleration, for manoeuvring targets)
      history_s      keep only this many seconds of rows per track (None = all). A live system sets it: a track on
                     a lamp lasts for hours, and everything downstream reads only the last few seconds
      keep_finished  keep timed-out tracks in `finished` for feature extraction after a clip (off for live use)
    """

    def __init__(self, high: float = 0.4, low: float = 0.1, q: float = 2e5, r_frac: float = 0.1,
                 r_min: float = 1.0, v0: float = 1500.0, n_confirm: int = 3, max_coast: float = 1.0,
                 tentative_coast: float = 0.25, gate: float = CHI2_2DOF_99, size_alpha: float = 0.3,
                 motion: str = "cv", history_s: float = None, keep_finished: bool = True):
        self.high, self.low, self.q, self.r_frac, self.r_min, self.v0 = high, low, q, r_frac, r_min, v0
        self.n_confirm, self.max_coast, self.tentative_coast = n_confirm, max_coast, tentative_coast
        self.gate_d2, self.size_alpha = gate, size_alpha
        self.history_s, self.keep_finished = history_s, keep_finished
        if motion not in ("cv", "imm"):
            raise ValueError(f"motion must be 'cv' or 'imm', not {motion!r}")
        self.motion = motion
        self.tracks: list = []
        self.finished: list = []   # tracks removed after timing out, kept for feature extraction
        self._next_id = 1
        self._t = None

    def _r_var(self, w: float) -> float:
        return max(self.r_min, self.r_frac * w) ** 2

    def _match(self, tracks, dets, t):
        """Hungarian matching on gated Mahalanobis d^2. Returns (pairs, unmatched track idx, unmatched det idx)."""
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        big = 1e9
        cost = np.full((len(tracks), len(dets)), big)
        for i, tr in enumerate(tracks):
            x, P = tr.kf.predicted(t - tr.t_last, self.q)
            for j, d in enumerate(dets):
                z = (d[0] + d[2] / 2, d[1] + d[3] / 2)
                d2 = tr.kf.gate(z, self._r_var(d[2]), x, P)
                if d2 <= self.gate_d2:
                    cost[i, j] = d2
        rows, cols = linear_sum_assignment(cost)
        pairs = [(r, c) for r, c in zip(rows, cols) if cost[r, c] < big]
        mt, md = {r for r, _ in pairs}, {c for _, c in pairs}
        return pairs, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(dets)) if j not in md]

    def _apply(self, tr: Track, d, t):
        tr.kf.predict(t - tr.t_last, self.q)
        tr.kf.update((d[0] + d[2] / 2, d[1] + d[3] / 2), self._r_var(d[2]))
        a = self.size_alpha
        tr.w, tr.h = (1 - a) * tr.w + a * d[2], (1 - a) * tr.h + a * d[3]
        tr.t_last, tr.hits, tr.cls = t, tr.hits + 1, int(d[4]) if len(d) > 4 else tr.cls
        tr.confirmed = tr.confirmed or tr.hits >= self.n_confirm
        tr.history.append(self._row(t, d, tr))
        if self.history_s is not None:
            while tr.history[0][0] < t - self.history_s:
                tr.history.pop(0)

    @staticmethod
    def _row(t, d, tr):
        cx, cy = d[0] + d[2] / 2, d[1] + d[3] / 2
        score = d[5] if len(d) > 5 else 1.0
        cls = d[4] if len(d) > 4 else 0
        return [t, cx, cy, d[2], d[3], score, cls, *tr.kf.x[:4]]

    def update(self, t: float, dets) -> list:
        if self._t is not None and t < self._t:
            raise ValueError(f"timestamps must not go backwards ({t} < {self._t})")
        self._t = t
        dets = [tuple(d) for d in dets]
        score = lambda d: d[5] if len(d) > 5 else 1.0
        high = [d for d in dets if score(d) >= self.high]
        low = [d for d in dets if self.low <= score(d) < self.high]

        # Stage 1: high-score detections, confirmed tracks FIRST and tentative tracks on what is left. One joint
        # assignment would not do this whatever the order: a new track has a huge covariance (v0), so its
        # Mahalanobis distance to any nearby detection is smaller than an established track's, and it wins
        # (16 #54: reproduced, and 14 times in 25,000 updates on Halmstad detections).
        confirmed = [tr for tr in self.tracks if tr.confirmed]
        tentative = [tr for tr in self.tracks if not tr.confirmed]
        pairs, un_c, un_d = self._match(confirmed, high, t)
        for i, j in pairs:
            self._apply(confirmed[i], high[j], t)
        left = [high[j] for j in un_d]
        pairs_t, un_t, un_d = self._match(tentative, left, t)
        for i, j in pairs_t:
            self._apply(tentative[i], left[j], t)
        # Stage 2: tracks still unmatched against low-score detections.
        rest = [confirmed[i] for i in un_c] + [tentative[i] for i in un_t]
        pairs2, _, _ = self._match(rest, low, t)
        for i, j in pairs2:
            self._apply(rest[i], low[j], t)
        # New tentative tracks from unmatched high-score detections.
        for j in un_d:
            d = left[j]
            new = IMMFilter if self.motion == "imm" else KalmanCV
            kf = new(d[0] + d[2] / 2, d[1] + d[3] / 2, self._r_var(d[2]), self.v0 ** 2)
            tr = Track(self._next_id, kf, t, d[2], d[3], int(d[4]) if len(d) > 4 else 0, t0=t)
            tr.history.append(self._row(t, d, tr))
            tr.confirmed = tr.hits >= self.n_confirm
            self._next_id += 1
            self.tracks.append(tr)
        # Life cycle: drop tracks that have coasted too long.
        live = []
        for tr in self.tracks:
            limit = self.max_coast if tr.confirmed else self.tentative_coast
            if t - tr.t_last <= limit:
                live.append(tr)
            elif tr.confirmed and self.keep_finished:
                self.finished.append(tr)
        self.tracks = live
        return [tr for tr in self.tracks if tr.confirmed]

    def all_tracks(self) -> list:
        """Every confirmed track, finished or still live (for feature extraction after a clip)."""
        return self.finished + [tr for tr in self.tracks if tr.confirmed]
