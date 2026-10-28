"""
M2 on REAL detector output (GPU; run on Colab): run a trained detector over every frame of the Halmstad
visible videos and save all detections at conf >= 0.05, so tracking and track classification can then be
redone locally on the CPU (`m2_halmstad_tracks.py extract --dets <file>`).

Halmstad frames are 640x512, i.e. one detector tile each, never rescaled.

    python experiments/m2_detect_videos.py --weights <run>/weights/best.pt --src <clone>/Data/Video_V \
        --out <data>/m2/dets_<run>.json.gz

Output: {video_stem: {"fps": 30.0, "n_frames": N, "dets": [[frame, x, y, w, h, cls, score], ...]}}.
Caveat: E12/E14 were trained on frames of the Halmstad TRAIN videos, so detections there are optimistic.
`m2_halmstad_tracks.py classify` therefore tests on the validation videos (the detector's own split).
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.detector import UltralyticsDetector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--src", required=True, help="Halmstad Data/Video_V")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    import cv2
    det = UltralyticsDetector(a.weights, conf=0.05, imgsz=640, device=0)
    videos = sorted(Path(a.src).glob("V_*.mp4"))
    if not videos:
        raise SystemExit(f"no V_*.mp4 in {a.src}")
    out, t0, n_total = {}, time.time(), 0
    for vid in videos:
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        rows, frames, idx = [], [], 0
        while True:
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
            if frames and (not ok or len(frames) == a.batch):
                for k, dets in enumerate(det(frames)):
                    rows += [[idx + k, round(float(x), 1), round(float(y), 1), round(float(w), 1),
                              round(float(h), 1), int(c), round(float(s), 3)] for x, y, w, h, c, s in dets]
                idx += len(frames)
                frames = []
            if not ok:
                break
        cap.release()
        out[vid.stem] = {"fps": float(fps), "n_frames": idx, "dets": rows}
        if len(out) == 1:
            json.dumps(out)                   # fail on the first video, not after all 285 (16 #50)
        n_total += idx
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, "wt") as f:
        json.dump(out, f)
    print(f"{len(videos)} videos, {n_total} frames in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
