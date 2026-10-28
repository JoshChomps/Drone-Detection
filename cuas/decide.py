"""
Sequential track decision (E26): declare a track a drone once enough evidence has ACCUMULATED, instead of on the
first window the classifier calls "drone".

Why. The track classifier is re-read several times a second on overlapping 2-s windows. Declaring on the first
"drone" window gives every track many chances to be called a drone once, which is where E20's 71 false declarations
per hour (track-only classifier) come from. A fixed hold time ("drone for 1 s without a break") helps but throws
away how sure each window was, and one doubtful window restarts the clock.

How. Wald's sequential probability ratio test on the calibrated probability. Each window adds its log-likelihood
ratio for drone against not-drone,

    llr = logit(p) - logit(prior)

where p is the classifier's calibrated p(drone) and `prior` the drone share it was trained on (the calibration
bakes that prior into p, so it is taken out again). The running sum is declared "drone" at `accept` and
"not_drone" at `reject`.

Overlapping windows are not independent: at 30 fps and one reading every 5 frames, twelve readings share most of
one 2-s window. Counting each in full would count the same seconds of flight twelve times. So each reading is
weighted by the share of its window that is new since the last reading used (`dt / window_s`, at most 1): a track
then gains about one window's evidence per window length, however often it is read.

The sum is clamped to [reject, accept + margin]. The floor lets a track that looked like a bird become a drone again
after `accept - reject` of contrary evidence; the ceiling keeps a long-declared track from becoming unshakeable.
Readings the classifier cannot vouch for (track shorter than the classifier's minimum, features outside its
training range) add nothing.

Thresholds are log-odds of evidence. A deployment where drones are rarer than in training needs a higher `accept`
for the same posterior; choose it on footage from the site, by false declarations per hour (E20, E26).

    dec = SequentialDecision(accept=3.0, prior=clf.prior)
    label = dec.update(track_id, t, p, usable=True, age=track_age_s)    # "drone" / "not_drone" / "unknown"
    dec.drop(track_id)                                                  # when the track ends
"""
import math


class SequentialDecision:
    def __init__(self, accept: float = 3.0, reject: float = None, prior: float = 0.5, window_s: float = 2.0,
                 p_clip: float = 0.02, margin: float = 2.0):
        self.accept = float(accept)
        self.reject = -self.accept if reject is None else float(reject)
        self.prior_logit = math.log(prior / (1 - prior))
        self.window_s, self.p_clip, self.margin = window_s, p_clip, margin
        self._llr, self._t = {}, {}

    def update(self, key, t: float, p, usable: bool = True, age: float = None) -> str:
        """Add one reading of the classifier for track `key` at time t; returns the track's label."""
        if p is None or not usable:
            return self.label(key)
        last = self._t.get(key)
        new_s = (t - last) if last is not None else (self.window_s if age is None else age)
        w = min(1.0, max(new_s, 0.0) / self.window_s)
        q = min(max(float(p), self.p_clip), 1 - self.p_clip)   # an isotonic calibrator outputs exact 0s and 1s
        v = self._llr.get(key, 0.0) + w * (math.log(q / (1 - q)) - self.prior_logit)
        self._llr[key] = min(max(v, self.reject), self.accept + self.margin)
        self._t[key] = t
        return self.label(key)

    def evidence(self, key) -> float:
        return self._llr.get(key, 0.0)

    def label(self, key) -> str:
        v = self.evidence(key)
        return "drone" if v >= self.accept else "not_drone" if v <= self.reject else "unknown"

    def drop(self, key):
        self._llr.pop(key, None)
        self._t.pop(key, None)
