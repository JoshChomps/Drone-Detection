"""
The program: a video file, camera or stream in; tracks, labels, time to collision and responses out.

    frame -> motion cueing -> crops around movers and current tracks (+ a full tiled scan every `scan_every`
    frames, for hoverers) -> detector -> tracker -> track classifier -> time to collision -> rules floor

    python -m cuas clip.mp4 --weights best.pt --clf track_clf.pkl --out out.mp4 --events events.jsonl
    python -m cuas 0 --weights best.pt --clf track_clf.pkl --show        (camera 0, live; q quits)
    python -m cuas rtsp://host/stream --weights best.pt --events events.jsonl
    python -m cuas clip.mp4                                               (motion cueing only: timing, proposals)
    python -m cuas sim.avi --weights blob                                 (no weights: the cuas.sim stand-in detector)

Without --clf every track stays "unknown" (the rules floor still alerts on unknown tracks closing on the camera).
Recorded video is timed by frame number; a live source by the clock at capture, so late frames are handled.
Prints per-stage timing, the numbers that decide what fits on the PC or the Raspberry Pi.
"""
import argparse
import csv
import json
import time

import cv2

from cuas.detector import detect_in_crops
from cuas.motion import MotionCue
from cuas.pipeline import TrackPipeline
from cuas.rules import RESPONSES
from cuas.tiling import crops_around, grid_tiles
from cuas.track import Tracker

COLOURS = {"drone": (0, 0, 255), "not_drone": (0, 170, 0), "unknown": (0, 200, 255)}      # BGR


class ImageList:
    """A list of image paths read like a video (datasets that store frames under timestamp names)."""

    def __init__(self, paths):
        self.paths, self.i = list(paths), 0

    def isOpened(self):
        return bool(self.paths)

    def read(self):
        if self.i >= len(self.paths):
            return False, None
        self.i += 1
        img = cv2.imread(str(self.paths[self.i - 1]))
        return img is not None, img

    def get(self, prop):
        return len(self.paths) if prop == cv2.CAP_PROP_FRAME_COUNT else 0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.i = int(value)

    def release(self):
        pass


def open_source(source):
    """(capture, live). A number is a camera index; a URL is a stream; a list is image paths; anything else is a
    file (or an image-sequence pattern such as dir/%05d.jpg)."""
    if isinstance(source, (list, tuple)):
        return ImageList(source), False
    s = str(source)
    live = s.isdigit() or "://" in s
    cap = cv2.VideoCapture(int(s) if s.isdigit() else s)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {source}")
    return cap, live


def draw(frame, props, states, banner):
    for x, y, bw, bh in props:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + bw), int(y + bh)), (200, 200, 200), 1)
    for s in states:
        x, y, bw, bh = s["box"]
        c = COLOURS[s["label"]]
        cv2.rectangle(frame, (int(x) - 3, int(y) - 3), (int(x + bw) + 3, int(y + bh) + 3), c, 1 if s["coasting"] else 2)
        p = "" if s["drone_prob"] is None else f" p={s['drone_prob']:.2f}"
        ttc = "" if s["ttc_s"] is None else f" ttc {s['ttc_s']:.1f}s"
        cv2.putText(frame, f"#{s['track_id']} {s['label']}{p}{ttc}", (int(x) - 3, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    if banner:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 24), (0, 0, 160), -1)
        cv2.putText(frame, banner, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def run(source, weights=None, clf=None, mask=None, out=None, csv_path=None, events_path=None, scan_every=15,
        tile=640, conf=0.4, max_frames=None, scale=0.5, var_threshold=16, show=False, detector=None,
        classifier=None, quiet=False, evidence=5.0, fps=None, preprocess=None, on_frame=None) -> dict:
    """detector / classifier: ready objects (tests, experiments) instead of `weights` / `clf` paths.
    fps: overrides the source's frame rate (an image sequence reports none). preprocess(frame) -> frame runs before
    everything else (e.g. inpainting a burned-in overlay). on_frame(n, t, frame, dets, states) is called after every
    frame with the detections (all scores) and the track states, for scoring (experiments/field_test.py).
    conf: the score that may START a track; detections down to 0.1 can continue one (cuas.track).
    evidence: a track is declared a drone once its accumulated evidence reaches this (cuas.decide; 5.0 chosen on
    cross-fitted Halmstad train videos in E26). 0 = the old rule, the label of the latest window.
    Returns {"rows": per-frame proposals and detections, "tracks": {id: summary}, "events": [...], "summary": {...}}."""
    cap, live = open_source(source)
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30
    mask_img = cv2.imread(mask, cv2.IMREAD_GRAYSCALE) if mask else None
    cue = MotionCue(scale=scale, var_threshold=var_threshold, mask=mask_img)
    det = detector
    if det is None and weights == "blob":
        from cuas.sim import BlobDetector
        det = BlobDetector()
    elif det is None and weights:
        from cuas.detector import UltralyticsDetector
        det = UltralyticsDetector(weights, conf=min(conf, 0.1), imgsz=tile)
    if classifier is None and clf:
        from cuas.classify import TrackClassifier
        classifier = TrackClassifier.load(clf)
    decision = None
    if classifier is not None and evidence > 0:
        from cuas.decide import SequentialDecision
        share = classifier.drone_share() if hasattr(classifier, "drone_share") else 0.5
        decision = SequentialDecision(accept=evidence, prior=share)
    pipe = TrackPipeline(Tracker(high=conf, history_s=2.5, keep_finished=False), classifier, decision=decision)
    names = getattr(det, "names", {})
    writer, rows, events, tracks = None, [], [], {}
    t_motion = t_det = t_track = 0.0
    n_crops = n = 0
    states, t_start = [], time.monotonic()
    ef = open(events_path, "w") if events_path else None
    try:
        while max_frames is None or n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if preprocess is not None:
                frame = preprocess(frame)
            t =time.monotonic() - t_start if live else n / fps
            h, w = frame.shape[:2]
            tiled = w > tile or h > tile
            t0 = time.perf_counter()
            props = cue(frame) if tiled or not det else []     # a frame that fits one tile needs no cueing
            t1 = time.perf_counter()
            dets = []
            if det:
                if tiled:
                    # Crops follow EVERY live track, tentative ones too: a hoverer is only found by the periodic
                    # scan, and a tentative track that is not looked at again expires before the next scan.
                    origins = crops_around(props + [tr.box for tr in pipe.tracker.tracks], w, h, tile)
                    if n % scan_every == 0:
                        origins = list(dict.fromkeys(origins + grid_tiles(w, h, tile)))
                else:
                    origins = [(0, 0)]
                n_crops += len(origins)
                dets = detect_in_crops(frame, origins, det, tile)
            t2 = time.perf_counter()
            states = pipe.update(t, dets, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) if det else []
            t3 = time.perf_counter()
            t_motion, t_det, t_track = t_motion + t1 - t0, t_det + t2 - t1, t_track + t3 - t2
            if on_frame is not None:
                on_frame(n, t, frame, dets, states)
            rows += [(n, "motion", x, y, bw, bh, "", "") for x, y, bw, bh in props]
            rows += [(n, "det", x, y, bw, bh, c, round(s, 3)) for x, y, bw, bh, c, s in dets if s >= conf]
            for s in states:
                k = tracks.setdefault(s["track_id"], {"first_t": t, "label": "unknown", "response": "log_only",
                                                      "declared_drone_t": None, "min_ttc_s": None})
                k.update(last_t=t, n_obs=s["n_obs"], label=s["label"])
                if s["ttc_s"] is not None:
                    k["min_ttc_s"] = s["ttc_s"] if k["min_ttc_s"] is None else min(k["min_ttc_s"], s["ttc_s"])
                new = []
                if s["label"] == "drone" and k["declared_drone_t"] is None:
                    k["declared_drone_t"] = t
                    new.append("declared_drone")
                if RESPONSES.index(s["response"]) > RESPONSES.index(k["response"]):
                    k["response"] = s["response"]
                    new.append(s["response"])
                for kind in new:
                    e = {"t": round(t, 3), "frame": n, "track_id": s["track_id"], "event": kind, "label": s["label"],
                         "drone_prob": s["drone_prob"], "ttc_s": s["ttc_s"], "box": [round(float(v), 1) for v in s["box"]]}
                    events.append(e)
                    if ef:
                        ef.write(json.dumps(e) + "\n")
                        ef.flush()
                    if not quiet:
                        print(f"  t={t:7.2f}s track {s['track_id']}: {kind}"
                              + ("" if s["ttc_s"] is None else f" (TTC {s['ttc_s']:.1f} s)"), flush=True)
            if out or show:
                worst = max((s["response"] for s in states), key=RESPONSES.index, default="log_only")
                vis = draw(frame.copy(), props, states, worst.upper().replace("_", " ") if RESPONSES.index(worst) >= 2 else "")
                if out:
                    if writer is None:
                        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    writer.write(vis)
                if show:
                    cv2.imshow("cuas (q quits)", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            n += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        if ef:
            ef.close()
        if show:
            cv2.destroyAllWindows()
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerows([("frame", "kind", "x", "y", "w", "h", "cls", "score")] + rows)
    m = max(n, 1)
    summary = {"frames": n, "fps": fps, "live": live, "motion_ms": 1000 * t_motion / m, "detector_ms": 1000 * t_det / m,
               "track_ms": 1000 * t_track / m, "crops_per_frame": n_crops / m,
               "proposals_per_frame": sum(r[1] == "motion" for r in rows) / m, "tracks": len(tracks),
               "declared_drone": sum(k["declared_drone_t"] is not None for k in tracks.values()),
               "alerts": sum(RESPONSES.index(k["response"]) >= 2 for k in tracks.values()), "class_names": names}
    if not quiet:
        print(f"{n} frames at {fps:.0f} fps: motion {summary['motion_ms']:.1f} ms/frame, "
              f"{summary['proposals_per_frame']:.1f} proposals/frame"
              + (f"; detector {summary['detector_ms']:.1f} ms/frame on {summary['crops_per_frame']:.1f} crops/frame; "
                 f"tracking + classification {summary['track_ms']:.1f} ms/frame\n"
                 f"{summary['tracks']} confirmed tracks, {summary['declared_drone']} declared drone, "
                 f"{summary['alerts']} raised an alert or higher" if det else ""))
    return {"rows": rows, "tracks": tracks, "events": events, "summary": summary}


def main():
    ap = argparse.ArgumentParser(prog="python -m cuas", description="Counter-drone detection over a video, camera or stream.")
    ap.add_argument("source", help="video file, camera index (0, 1, ...) or stream URL")
    ap.add_argument("--weights", help="detector weights (.pt); 'blob' = the stand-in detector for simulated clips")
    ap.add_argument("--clf", help="track classifier (.pkl from experiments/m2_halmstad_tracks.py classify)")
    ap.add_argument("--mask", help="uint8 image, 255 = sky to watch, 0 = ignore")
    ap.add_argument("--out", help="annotated output video (.mp4)")
    ap.add_argument("--events", help="JSON lines: one per track declared a drone or raising its response")
    ap.add_argument("--csv", help="proposals and detections per frame")
    ap.add_argument("--show", action="store_true", help="live window (q quits)")
    ap.add_argument("--scan_every", type=int, default=15, help="full tiled scan period in frames")
    ap.add_argument("--conf", type=float, default=0.4, help="detector score that may start a track")
    ap.add_argument("--scale", type=float, default=0.5, help="motion-cueing downscale")
    ap.add_argument("--var_threshold", type=float, default=16, help="MOG2 sensitivity (lower = fainter)")
    ap.add_argument("--max_frames", type=int)
    ap.add_argument("--evidence", type=float, default=5.0,
                    help="evidence needed to declare a drone (E26; higher = fewer false declarations, later); 0 = "
                         "declare on the first window labelled drone")
    a = ap.parse_args()
    run(a.source, a.weights, a.clf, a.mask, a.out, a.csv, a.events, a.scan_every, conf=a.conf,
        max_frames=a.max_frames, scale=a.scale, var_threshold=a.var_threshold, show=a.show, evidence=a.evidence)


if __name__ == "__main__":
    main()
