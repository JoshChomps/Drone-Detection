"""
A short video that shows where the project stands, made only from footage the detector never trained on.

Part 1: the whole program (motion cueing -> detector -> tracker -> track decision) on three held-out Halmstad
        videos: a drone, a bird, an airplane. The overlay is the program's own (cuas/replay.py).
Part 2: the same drone video put through blur + sensor noise + compression (the E21 "field" condition), with the
        plain model on the left and the model trained on degraded images (E22a) on the right, both at the same
        threshold. A counter shows in how many frames each has drawn a "drone" box.

    C:\\cuas_venv\\Scripts\\python experiments/make_demo.py            (CPU, a few minutes)
    -> C:/cuas_data/demo/progress_demo.mp4

It is a picture of progress, not a measurement: three videos, one dataset, in-domain. The numbers are in 15 and 19.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuas.detector import UltralyticsDetector  # noqa: E402
from cuas.replay import run  # noqa: E402
from e21_degradation import field  # noqa: E402

D = Path(os.environ.get("CUAS_DATA", "C:/cuas_data"))
VIDEOS = D / "halmstad" / "Data" / "Video_V"
OUT = D / "demo"
W, H, BAND, FPS = 1280, 512, 44, 30
PLAIN, DEGRADED = "E18_base_s1", "E22a_s1"       # same seed, final weights
CLF = D / "m2" / "track_clf_dets_E12.pkl"
CLIPS = [("V_DRONE_037", "A drone"), ("V_BIRD_034", "A bird"), ("V_AIRPLANE_017", "An airplane")]
CONF, SECONDS = 0.40, 9


def weights(run_name):
    return str(D / "runs" / run_name / "weights" / "last.pt")


def canvas(caption, sub=""):
    img = np.zeros((H + BAND, W, 3), np.uint8)
    cv2.putText(img, caption, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, sub, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
    return img


def card(writer, lines, seconds=2.5):
    img = np.zeros((H + BAND, W, 3), np.uint8)
    for i, (text, size) in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 1)
        cv2.putText(img, text, ((W - tw) // 2, 200 + 46 * i), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1, cv2.LINE_AA)
    for _ in range(int(seconds * FPS)):
        writer.write(img)


def part1(writer, det):
    for name, what in CLIPS:
        tmp = OUT / f"_{name}.mp4"
        r = run(str(VIDEOS / f"{name}.mp4"), clf=str(CLF), out=str(tmp), conf=CONF, max_frames=SECONDS * FPS,
                detector=det, quiet=True)
        s = r["summary"]
        declared = [e for e in r["events"] if e.get("label") == "drone"]
        verdict = (f"declared a drone {declared[0]['t']:.1f} s after it appears" if declared else "no drone declared")
        print(f"{name}: {s}; {verdict}", flush=True)
        cap = cv2.VideoCapture(str(tmp))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            img = canvas(f"{what} (held-out video {name}): {verdict}",
                         "grey = moving region, box = track, colour = the track's label")
            img[BAND:, (W - frame.shape[1]) // 2:(W + frame.shape[1]) // 2] = frame[:H]
            writer.write(img)
        cap.release()
        tmp.unlink()


def part2(writer, plain, degraded):
    name = CLIPS[0][0]
    cap, rng, found, n = cv2.VideoCapture(str(VIDEOS / f"{name}.mp4")), np.random.default_rng(0), [0, 0], 0
    while n < SECONDS * FPS:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        noisy = field(frame, rng)
        img = canvas("The same video through blur + sensor noise + compression, both models at threshold 0.40",
                     "left: trained on clean images            right: trained on degraded images (E22a)")
        for k, det in enumerate((plain, degraded)):
            view = noisy.copy()
            boxes = [d for d in det([noisy])[0] if d[4] == 0 and d[5] >= CONF]
            found[k] += bool(boxes)
            for x, y, bw, bh, _, score in boxes:
                cv2.rectangle(view, (int(x) - 3, int(y) - 3), (int(x + bw) + 3, int(y + bh) + 3), (0, 220, 0), 2)
                cv2.putText(view, f"drone {score:.2f}", (int(x) - 3, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 220, 0), 1, cv2.LINE_AA)
            cv2.putText(view, f"drone boxed in {found[k]} of {n} frames", (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            img[BAND:, k * 640:k * 640 + view.shape[1]] = view[:H]
        writer.write(img)
    cap.release()
    print(f"degraded {name}: plain boxed the drone in {found[0]} of {n} frames, E22a in {found[1]}", flush=True)
    return found, n


def main():
    OUT.mkdir(exist_ok=True)
    plain = UltralyticsDetector(weights(PLAIN), conf=0.05, device="cpu")
    degraded = UltralyticsDetector(weights(DEGRADED), conf=0.05, device="cpu")
    raw = OUT / "_progress_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H + BAND))
    card(writer, [("Vision-based drone detection: where it stands", 1.0), ("Footage the detector never trained on", 0.7)])
    card(writer, [("Part 1: the whole program on held-out video", 0.9),
                  ("motion cueing  ->  detector  ->  tracker  ->  track decision", 0.6)], 2.0)
    part1(writer, degraded)
    card(writer, [("Part 2: a noisy, blurred, compressed camera", 0.9),
                  ("before and after training on degraded images", 0.6)], 2.0)
    part2(writer, plain, degraded)                        # ends on the degraded video (user, 2026-10-06)
    writer.release()
    final = OUT / "progress_demo.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:                                                  # pip install imageio-ffmpeg bundles a binary
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg = "ffmpeg"
    try:
        code = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                               "-pix_fmt", "yuv420p", "-crf", "23", str(final)]).returncode
    except FileNotFoundError:
        code = 1
    if code == 0:
        raw.unlink()                                          # H.264 plays in a browser; mp4v often does not
    else:
        raw.replace(final)
    print(f"wrote {final}")


if __name__ == "__main__":
    main()
