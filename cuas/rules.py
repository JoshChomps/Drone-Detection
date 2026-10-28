"""
Deterministic rules floor (doc 12 §3-4, without the Jev advisor): the minimum response for a track, from
hard, auditable conditions. Synchronous, no network, never triggers an effector: the top level asks a human.

Track fields used:
  ttc_s                time to reach the CAMERA from optical expansion (s), None if not closing
  drone_prob           calibrated p(drone) from cuas.classify (doc 12 called it detector_drone_prob); None or
                       label == "unknown" means NOT YET CLASSIFIED, which is not the same as "not a drone"
  in_protected_zone    bool (the caller decides, e.g. from a zone mask or range gate)
  remote_id_authorised bool; Remote ID is unauthenticated, so it may only relax NON-imminent tracks
Thresholds are site policy; the camera sits inside the zone, so IMMINENT_TTC_S includes crossing the zone.
"""
RESPONSES = ["log_only", "slew_ptz_for_identification", "alert_operator", "request_engagement_decision"]

IMMINENT_TTC_S = 10.0     # a closing track due at the camera within this time is imminent
DECIDE_NOW_TTC_S = 3.0    # this close, a human decision barely fits (13 §3), so ask for one now
DRONE_PROB = 0.5


def rules_floor(track: dict) -> str:
    ttc = track.get("ttc_s")
    # 0 counts: ttc_expansion clamps at 0 when the target is due now (inside OR outside the zone).
    imminent = ttc is not None and 0 <= ttc < IMMINENT_TTC_S
    p, label = track.get("drone_prob"), track.get("label")
    # The classifier's label wins when there is one: "unknown" (track too short, p in the undecided band, features
    # outside the training range) means p is not to be trusted, however high it is.
    drone = label == "drone" if label is not None else (p is not None and p >= DRONE_PROB)
    unknown = label == "unknown" if label is not None else p is None
    in_zone = bool(track.get("in_protected_zone", False))
    # Imminence is checked BEFORE Remote ID on purpose: a spoofed "friendly" ID must never silence it.
    if imminent and drone and ttc < DECIDE_NOW_TTC_S:
        return "request_engagement_decision"
    if imminent and (drone or in_zone):
        return "alert_operator"
    # Not yet classified is not "not a drone": the fastest threats are the ones with too little track to classify
    # (an FPV gives under a second), so a closing unknown must not fall through to log_only. It gets what it lacks,
    # identification. It does NOT alert an operator: on real Halmstad tracks of distant aircraft and birds, 1-2
    # tracks a minute read "TTC under 3 s" (box-size error is correlated in time, so the fit's noise test
    # under-counts it), and alerting on those gave 4 false alerts in 6 minutes of non-drone video (16 #56).
    if imminent and unknown:
        return "slew_ptz_for_identification"
    if track.get("remote_id_authorised", False):
        return "log_only"
    if drone and in_zone:
        return "alert_operator"
    if drone or in_zone:
        return "slew_ptz_for_identification"
    return "log_only"


def combine(floor: str, advice, min_confidence: float = 0.7) -> str:
    """An advisor (e.g. Jev, deferred) may RAISE the rules floor when confident, never lower it."""
    if advice and advice.get("suggested") in RESPONSES and advice.get("confidence", 0) >= min_confidence:
        return max(floor, advice["suggested"], key=RESPONSES.index)
    return floor
