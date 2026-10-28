"""
Synthetic scenes with ground truth: targets fly 3-D paths, a pinhole camera projects them, each is drawn at its
TRUE pixel size on a sky background, and the frame is then degraded like a camera's (defocus, mount shake, sensor
noise). For testing the tracker, time to collision, the rules floor and the whole program end to end, with a known
answer and controllable noise. NOT for training or scoring the detector: a drawn blob says nothing about what a
real drone looks like (11 row 9).

    python -m cuas.sim --out C:/cuas_data/sim/approach.avi          (also writes approach.json: the ground truth)
    python -m cuas C:/cuas_data/sim/approach.avi --weights blob --out C:/cuas_data/sim/approach_out.mp4

Coordinates: metres, camera at the origin, e = east (image right), n = north (the way the camera looks),
u = up. The optical axis is raised `tilt_deg` above the horizon (13 section 2.4: lower edge just above the skyline).

Paths from PX4: fly the mission in PX4 SITL (Gazebo or jMAVSim; Linux/WSL2), convert the log with pyulog's
`ulog2csv`, and pass the `vehicle_local_position` CSV to `from_px4_csv`. That gives real flight dynamics (hover
jitter, turns, wind response) without using the simulator's rendered pixels.
"""
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Camera:
    w: int = 1280
    h: int = 720
    fx: float = 3000.0          # focal length in px. Pi HQ + 16 mm at full resolution is ~10,300 (13 section 2)
    tilt_deg: float = 6.0

    def project(self, p, size_m: float):
        """(cx, cy, width_px, range_m) of a point (e, n, u), or None behind the camera."""
        e, n, u = p
        c, s = np.cos(np.radians(self.tilt_deg)), np.sin(np.radians(self.tilt_deg))
        depth, up = n * c + u * s, -n * s + u * c
        if depth <= 1.0:
            return None
        rng = float(np.linalg.norm(p))
        return self.w / 2 + self.fx * e / depth, self.h / 2 - self.fx * up / depth, self.fx * size_m / rng, rng


@dataclass
class Target:
    name: str
    path: object                # callable t -> (e, n, u) in metres
    size_m: float = 0.35        # widest dimension: rotor span of a Mavic-class quad, wingspan of a bird
    cls: int = 0                # 0 drone, 1 bird (project class ids)
    flap_hz: float = 0.0        # birds: the visible wingspan oscillates at this rate
    shade: float = 60.0         # grey level of the body (sky is ~170-230)


def line(p0, v):
    p0, v = np.asarray(p0, float), np.asarray(v, float)
    return lambda t: p0 + v * t


def approach(p0, speed):
    """Straight at the camera from p0 at `speed` m/s (stops 5 m short)."""
    p0 = np.asarray(p0, float)
    d = np.linalg.norm(p0)
    return lambda t: p0 * max(d - speed * t, 5.0) / d


def hover(p, wobble_m: float = 0.3, hz: float = 0.4):
    p = np.asarray(p, float)
    return lambda t: p + wobble_m * np.array([np.sin(2 * np.pi * hz * t), 0.0, 0.5 * np.sin(2 * np.pi * 1.7 * hz * t)])


def waypoints(times, points):
    """Linear interpolation through (time, (e, n, u)) samples; held at the ends."""
    times, points = np.asarray(times, float), np.asarray(points, float)
    return lambda t: np.array([np.interp(t, times, points[:, k]) for k in range(3)])


def from_px4_csv(path, camera_ned=(0.0, 0.0, 0.0)):
    """Path from a PX4 `vehicle_local_position` CSV (ulog2csv): timestamp in microseconds, x north, y east, z DOWN,
    relative to the vehicle's local origin. `camera_ned` is the camera's position in that same frame."""
    with open(path, newline="") as f:
        rows = [(float(r["timestamp"]), float(r["x"]), float(r["y"]), float(r["z"])) for r in csv.DictReader(f)]
    a = np.array(rows)
    north, east, down = (a[:, k + 1] - camera_ned[k] for k in range(3))
    return waypoints((a[:, 0] - a[0, 0]) / 1e6, np.column_stack([east, north, -down]))


def sky(w: int, h: int, rng, margin: int = 40):
    """A sky brighter towards the horizon with soft cloud at two scales, `margin` px larger than the frame on every
    side (room for mount shake). Deliberately free of small hard-edged structure: clutter goes in as targets."""
    H, W = h + 2 * margin, w + 2 * margin
    big = np.repeat(np.linspace(185, 215, H, dtype=np.float32)[:, None], W, axis=1)
    for sigma, amp in ((60, 16), (15, 5)):
        n = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigma)
        big += n / n.std() * amp
    return big


def render(targets, cam: Camera = None, seconds: float = 6.0, fps: float = 30.0, background=None,
           jitter_px: float = 1.0, blur: float = 0.7, noise: float = 2.0, seed: int = 0):
    """Yield (t, frame BGR uint8, truth): truth = [(x, y, w, h, cls, name, range_m), ...] of the targets in view,
    in frame pixels. background: a grey float image at least 40 px larger than the frame on every side, or None
    for a procedural sky. jitter_px: mount shake (uniform, whole image). blur: lens/defocus Gaussian sigma in px.
    noise: sensor noise sigma in grey levels."""
    cam = cam or Camera()
    rng = np.random.default_rng(seed)
    m = 40
    bg = sky(cam.w, cam.h, rng, m) if background is None else np.asarray(background, np.float32)
    for k in range(int(round(seconds * fps))):
        t = k / fps
        f = bg[m:m + cam.h, m:m + cam.w].copy()
        truth = []
        for tg in targets:
            pr = cam.project(np.asarray(tg.path(t), float), tg.size_m)
            if pr is None:
                continue
            cx, cy, wpx, r = pr
            span = wpx * (0.35 + 0.65 * abs(np.sin(np.pi * tg.flap_hz * t))) if tg.flap_hz else wpx
            hpx = max(0.45 * wpx, 1.0) if not tg.flap_hz else max(0.3 * wpx, 1.0)
            if not (-span < cx < cam.w + span and -hpx < cy < cam.h + hpx):
                continue
            # Sub-pixel anti-aliased ellipse (1/16 px), so a 3-px target fades instead of snapping to the grid.
            a = np.zeros(f.shape, np.float32)
            cv2.ellipse(a, (int(round(cx * 16)), int(round(cy * 16))), (max(int(span * 8), 8), max(int(hpx * 8), 8)),
                        0, 0, 360, 1.0, -1, cv2.LINE_AA, 4)
            f = f * (1 - a) + tg.shade * a
            truth.append([cx - span / 2, cy - hpx / 2, span, hpx, tg.cls, tg.name, r])
        if blur:
            f = cv2.GaussianBlur(f, (0, 0), blur)
        dx, dy = rng.uniform(-jitter_px, jitter_px, 2) if jitter_px else (0.0, 0.0)
        if jitter_px:       # the scene shifts as a whole; the background beyond the frame fills the border
            full = bg.copy()
            full[m:m + cam.h, m:m + cam.w] = f
            f = cv2.warpAffine(full, np.float32([[1, 0, dx], [0, 1, dy]]), full.shape[::-1])[m:m + cam.h, m:m + cam.w]
        if noise:
            f = f + rng.normal(0, noise, f.shape)
        frame = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        yield t, frame, [(x + dx, y + dy, w, h, c, n, r) for x, y, w, h, c, n, r in truth]


class BlobDetector:
    """Stand-in for a trained detector, with the same call signature (cuas.detector): small dark blobs against sky,
    by a black top-hat (what a closing over `size` px fills in), so clouds and other structure larger than `size`
    do not respond. It lets the tracker, classifier, rules and program run end to end without weights or a GPU.
    It is NOT a drone detector: every small dark blob is reported as class 0."""

    names = {0: "blob"}

    def __init__(self, size: int = 41, k: float = 8.0, min_contrast: float = 12.0, min_area: int = 2,
                 max_area: int = 4000, score: float = 0.9):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        self.k, self.min_contrast, self.min_area, self.max_area, self.score = k, min_contrast, min_area, max_area, score

    def __call__(self, images):
        out = []
        for img in images:
            g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img, (0, 0), 1)
            d = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, self.kernel).astype(np.float32)
            noise = 1.4826 * float(np.median(np.abs(d - np.median(d)))) + 1e-6
            mask = (d > max(self.k * noise + float(np.median(d)), self.min_contrast)).astype(np.uint8)
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            dets = []
            for x, y, w, h, area in stats[1:n]:
                # Blobs on the crop's edge are cut targets or filter edge effects; an overlapping crop sees them whole.
                if not self.min_area <= area <= self.max_area or x < 2 or y < 2 or x + w > g.shape[1] - 2                         or y + h > g.shape[0] - 2:
                    continue
                # Box at half the blob's peak contrast: the threshold box grows with the blur, and a constant few
                # px added to a small width would flatten its growth rate (and so stretch the time to collision).
                p = d[y:y + h, x:x + w]
                ys, xs = np.nonzero(p >= 0.5 * p.max())
                dets.append((float(x + xs.min()), float(y + ys.min()), float(xs.max() - xs.min() + 1),
                             float(ys.max() - ys.min() + 1), 0, self.score))
            out.append(dets)
        return out


def demo_scene():
    """A drone closing on the camera, a drone hovering far away, and a flapping bird crossing."""
    return [
        Target("drone_approach", approach((-25.0, 300.0, 60.0), 30.0)),
        Target("drone_hover", hover((30.0, 220.0, 30.0))),
        Target("bird", line((-35.0, 150.0, 12.0), (9.0, 0.0, 0.5)), size_m=0.9, cls=1, flap_hz=5.0, shade=75.0),
    ]


def write_video(path, frames, fps: float = 30.0):
    """Write (t, frame, truth) items as MJPG video; returns the ground truth per frame."""
    writer, truth = None, []
    for t, frame, tr in frames:
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, frame.shape[1::-1])
        writer.write(frame)
        truth.append({"t": t, "targets": [{"box": [round(float(v), 2) for v in b[:4]], "cls": b[4], "name": b[5],
                                           "range_m": round(b[6], 1)} for b in tr]})
    if writer:
        writer.release()
    return truth


def main():
    ap = argparse.ArgumentParser(description="Render the demo scene (ground truth alongside as .json).")
    ap.add_argument("--out", required=True, help="output video (.avi, MJPG)")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--noise", type=float, default=2.0, help="sensor noise sigma, grey levels")
    ap.add_argument("--blur", type=float, default=0.7, help="lens blur sigma, px")
    ap.add_argument("--jitter", type=float, default=1.0, help="mount shake, px")
    ap.add_argument("--px4_csv", help="vehicle_local_position CSV from ulog2csv: replaces the approaching drone's path")
    ap.add_argument("--camera_ned", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="camera position in the PX4 frame")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    targets = demo_scene()
    if a.px4_csv:
        targets[0] = Target("px4", from_px4_csv(a.px4_csv, a.camera_ned))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    truth = write_video(a.out, render(targets, seconds=a.seconds, fps=a.fps, jitter_px=a.jitter, blur=a.blur,
                                      noise=a.noise, seed=a.seed), a.fps)
    Path(a.out).with_suffix(".json").write_text(json.dumps({"fps": a.fps, "frames": truth}))
    print(f"{len(truth)} frames -> {a.out} (+ {Path(a.out).with_suffix('.json').name})")


if __name__ == "__main__":
    main()
