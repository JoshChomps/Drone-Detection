"""
Tests for the cuas package: numpy + OpenCV (+ scipy, scikit-learn for M2), no GPU, no network, no datasets.

    python -m unittest discover -s tests
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cuas import tiling, zoomout  # noqa: E402
from cuas.datasets.halmstad import split_videos  # noqa: E402
from cuas.detector import detect_in_crops  # noqa: E402
from cuas.eval import (bootstrap_ci, evaluate, format_report, image_counts, iou, recall_precision,  # noqa: E402
                       run_files, run_split, video_of, w_min)
from cuas.motion import MotionCue  # noqa: E402
from cuas.replay import run as replay  # noqa: E402
from cuas.classify import TrackClassifier, ece  # noqa: E402
from cuas.datasets.dvb import label_stream, parse_annotations  # noqa: E402
from cuas.decide import SequentialDecision  # noqa: E402
from cuas.pipeline import TrackPipeline  # noqa: E402
from cuas.rules import combine, rules_floor  # noqa: E402
from cuas.sim import BlobDetector, Camera, Target, approach, from_px4_csv, hover, line, render, write_video  # noqa: E402
from cuas.track import IMMFilter, KalmanCV, Tracker  # noqa: E402
from cuas.track_features import FEATURE_NAMES, silhouette, track_features, ttc_expansion  # noqa: E402


class Tiling(unittest.TestCase):
    def test_origins_cover_and_end_flush(self):
        self.assertEqual(len(tiling.grid_tiles(3840, 2160)), 32)          # doc 13 section 3.1
        o = tiling.tile_origins(3840, 640)
        self.assertEqual((o[0], o[-1]), (0, 3200))
        self.assertEqual(tiling.tile_origins(512, 640), [0])

    def test_boxes_in_tile_clips(self):
        boxes = [(630, 10, 20, 10, 0), (100, 100, 5, 5, 1)]
        self.assertEqual(tiling.boxes_in_tile(boxes, 0, 0, 640, 640), [(630, 10, 10, 10, 0), (100, 100, 5, 5, 1)])
        self.assertEqual(tiling.boxes_in_tile(boxes, 0, 0, 640, 640, min_visible=0.6), [(100, 100, 5, 5, 1)])

    def test_training_tiles(self):
        rng = np.random.default_rng(1)
        box = (2000.5, 1000.2, 9.0, 7.0, 0)
        for _ in range(50):
            tiles = tiling.training_tiles(3840, 2160, [box], 640, n_negative=2, rng=rng)
            pos = [t for t in tiles if t[2]]
            self.assertEqual(len(pos), 1)
            (ox, oy, tb), = pos
            self.assertEqual(len(tb), 1)
            self.assertAlmostEqual(tb[0][2], 9.0)                         # not shrunk or clipped
            self.assertTrue(0 <= ox <= 3200 and 0 <= oy <= 1520)
            self.assertEqual(len(tiles) - 1, 2)                            # negatives are empty tiles

    def test_box_starting_off_frame_does_not_crash(self):
        # A target leaving the view is labelled partly outside the frame; this raised ValueError (16 #54).
        for box in [(-5.0, 300.0, 50.0, 20.0, 0), (1900.0, -8.0, 15.0, 30.0, 0)]:
            (ox, oy, tb), = tiling.training_tiles(1920, 1080, [box], 640, n_negative=0, rng=np.random.default_rng(0))
            self.assertTrue(0 <= ox <= 1280 and 0 <= oy <= 440)
            self.assertEqual(len(tb), 1)                                   # the visible part is still labelled

    def test_small_frame_single_tile(self):
        (ox, oy, tb), = tiling.training_tiles(640, 512, [(300, 200, 8, 6, 2)], 640, n_negative=0)
        self.assertEqual((ox, oy, tb), (0, 0, [(300, 200, 8, 6, 2)]))

    def test_yolo_roundtrip(self):
        boxes = [(10, 20, 6, 4, 0), (300, 100, 50, 30, 1)]
        back = tiling.read_yolo_labels(tiling.yolo_lines(boxes, 640, 512), 640, 512)
        for a, b in zip(boxes, back):
            np.testing.assert_allclose(a, b, atol=1e-3)

    def test_tile_split_positive_tiles_only(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            (src / "images").mkdir(parents=True)
            (src / "labels").mkdir()
            cv2.imwrite(str(src / "images" / "a.jpg"), np.full((1080, 1920, 3), 100, np.uint8))
            (src / "labels" / "a.txt").write_text(tiling.yolo_lines([(1500, 800, 110, 70, 0)], 1920, 1080))
            cv2.imwrite(str(src / "images" / "empty.jpg"), np.full((1080, 1920, 3), 100, np.uint8))
            (src / "labels" / "empty.txt").write_text("")
            self.assertEqual(tiling.tile_split(src, Path(d) / "dst"), 1)       # no tile from the empty frame
            (img,) = list((Path(d) / "dst" / "images").iterdir())
            self.assertEqual(cv2.imread(str(img)).shape[:2], (640, 640))
            lbl = (Path(d) / "dst" / "labels" / (img.stem + ".txt")).read_text()
            (x, y, w, h, c), = tiling.read_yolo_labels(lbl, 640, 640)
            self.assertAlmostEqual(w, 110, delta=0.01)                          # true size, not rescaled

    def test_crops_around_merges_and_clips(self):
        o = tiling.crops_around([(100, 100, 5, 5), (150, 120, 4, 4), (3830, 2150, 6, 6)], 3840, 2160)
        self.assertEqual(o, [(0, 0), (3200, 1520)])
        # two boxes in the bottom-right corner, the second within the first tile's 32-px margin strip: both clip
        # to the same origin, which is returned once (it was returned twice before 16 #40)
        self.assertEqual(tiling.crops_around([(1900, 1070, 10, 8), (1880, 1060, 10, 8)], 1920, 1080), [(1280, 440)])


class Eval(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 10, 10)), 50 / 150)
        self.assertEqual(iou((0, 0, 1, 1), (5, 5, 1, 1)), 0.0)

    def test_recall_bins_and_false_alarms(self):
        imgs = [
            ([(10, 10, 6, 6, 0)], [(10, 10, 6, 6, 0, 0.9)]),               # 6-px drone found
            ([(10, 10, 6, 6, 0)], []),                                       # 6-px drone missed
            ([(10, 10, 20, 20, 0)], [(11, 10, 20, 20, 0, 0.8), (11, 10, 20, 20, 0, 0.7)]),  # duplicate on a drone
            ([(50, 50, 10, 10, 1)], [(50, 50, 10, 10, 0, 0.6)]),             # bird called drone
            ([], [(0, 0, 5, 5, 0, 0.5), (0, 0, 5, 5, 1, 0.5)]),              # background FA; bird pred ignored
        ]
        r = evaluate(imgs)
        self.assertEqual(r["n_gt"][1], 2)                 # 4-8 px bin
        self.assertEqual(r["recall"][1], 0.5)
        self.assertEqual(r["recall"][4], 1.0)             # 16-24 px bin
        self.assertEqual(r["false_drone_on_distractor"], {1: 1})
        self.assertEqual(r["false_drone_on_drone"], 1)    # the duplicate on the 20-px drone
        self.assertEqual(r["false_drone_on_background"], 1)
        self.assertEqual(r["false_drone_background_by_width"][1], 1)   # the 5-px background FA
        self.assertEqual(sum(r["false_drone_background_by_width"]), 1)
        self.assertIn("bg false drones", format_report(r))
        r2 = evaluate([([(10, 10, 20, 20, 0)], [(10, 10, 20, 20, 2, 0.8)])])   # drone boxed as "airplane"
        self.assertEqual((r2["recall"][4], r2["missed_drone_boxed_as_other"]), (0.0, {2: 1}))

    def test_w_min(self):
        r = {"bins": [[0, 4], [4, 8], [8, 12], [12, 16]], "n_gt": [30, 30, 30, 5], "recall": [0.2, 0.7, 0.9, 0.1]}
        self.assertEqual(w_min(r, 0.8), 8)                # the 5-sample bin is too small to count
        self.assertIsNone(w_min(r, 0.95))

    def test_run_split_reads_labels(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "images").mkdir()
            (Path(d) / "labels").mkdir()
            cv2.imwrite(str(Path(d) / "images" / "a.jpg"), np.zeros((64, 80, 3), np.uint8))
            (Path(d) / "labels" / "a.txt").write_text(tiling.yolo_lines([(10, 10, 8, 8, 0)], 80, 64))
            fake = lambda imgs: [[(10, 10, 8, 8, 0, 0.9), (0, 0, 2, 2, 0, 0.1)] for _ in imgs]  # noqa: E731
            (gts, preds), = run_split(fake, d, conf=0.25)
            self.assertEqual(len(preds), 1)
            self.assertEqual(evaluate([(gts, preds)])["recall"][2], 1.0)

    def test_run_spec(self):
        self.assertEqual(run_files("E14_hardneg"), ("E14_hardneg", "best.pt", ""))
        self.assertEqual(run_files("E14_hardneg@last"), ("E14_hardneg", "last.pt", "_last"))
        with self.assertRaises(ValueError):
            run_files("E14_hardneg@epoch20")

    def test_video_of(self):
        self.assertEqual(video_of("x/V_DRONE_001_00040_0_0.jpg"), "V_DRONE_001")
        self.assertEqual(video_of("20190925_111757_1_10_00040.jpg"), "20190925_111757_1_10")

    def test_image_counts(self):
        gts = [(10, 10, 6, 6, 0), (50, 50, 10, 10, 1)]
        preds = [(10, 10, 6, 6, 0, 0.9), (50, 50, 10, 10, 0, 0.6), (200, 200, 5, 5, 0, 0.5)]
        self.assertEqual(image_counts(gts, preds), [1, 1, 2, 1])           # bird FA + background FA

    def test_bootstrap_resamples_videos_not_frames(self):
        # 10 videos x 100 frames; recall is 1 in half the videos and 0 in the other half. A frame-level
        # bootstrap would give a very narrow interval; resampling videos must give a wide one.
        keys = [f"v{v}" for v in range(10) for _ in range(100)]
        rows = [[1, v % 2, 0, 0] for v in range(10) for _ in range(100)]
        rec, lo, hi = bootstrap_ci(keys, rows, lambda s: recall_precision(s)[0], n_boot=500)
        self.assertAlmostEqual(rec, 0.5)
        self.assertLess(lo, 0.3)
        self.assertGreater(hi, 0.7)
        # paired difference of identical models is exactly zero in every resample
        paired = [r + r for r in rows]
        self.assertEqual(bootstrap_ci(keys, paired, lambda s: s[5] / s[4] - s[1] / s[0], n_boot=50), (0.0, 0.0, 0.0))


class TiledEval(unittest.TestCase):
    def test_full_hd_is_tiled_not_downscaled(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "images").mkdir()
            (Path(d) / "labels").mkdir()
            img = np.zeros((1080, 1920, 3), np.uint8)
            img[900:908, 1700:1710] = 255                                 # a 10x8 target near a corner
            cv2.imwrite(str(Path(d) / "images" / "a.png"), img)
            (Path(d) / "labels" / "a.txt").write_text(tiling.yolo_lines([(1700, 900, 10, 8, 0)], 1920, 1080))
            seen = []

            def bright_spot(crops):                                       # "detects" the bright pixels it is given
                seen.extend(c.shape[:2] for c in crops)
                out = []
                for c in crops:
                    ys, xs = np.nonzero(c[..., 0] > 128)
                    out.append([(xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, 0, 0.9)]
                               if len(xs) else [])
                return out

            (gts, preds), = run_split(bright_spot, d, tile=640)
            self.assertTrue(all(s == (640, 640) for s in seen))           # only true-size tiles reach the model
            self.assertEqual([tuple(int(v) for v in p[:4]) for p in preds], [(1700, 900, 10, 8)])
            self.assertEqual(evaluate([(gts, preds)])["recall"][2], 1.0)


class Detection(unittest.TestCase):
    def test_crop_offsets_and_dedup(self):
        frame = np.zeros((1000, 1500, 3), np.uint8)
        calls = []

        def fake(crops):                                  # sees a 6-px object at tile coords depending on origin
            calls.append([c.shape for c in crops])
            return [[(700 - ox, 300 - oy, 6, 6, 0, 0.9)] for ox, oy in origins]

        origins = [(400, 0), (600, 100)]                  # both tiles contain the object at frame (700, 300)
        dets = detect_in_crops(frame, origins, fake)
        self.assertEqual(dets, [(700, 300, 6, 6, 0, 0.9)])
        self.assertEqual(calls[0], [(640, 640, 3)] * 2)


def synthetic_sky(n=60, h=360, w=480, target=True, jitter=1.5, seed=0):
    """Textured sky (soft cloud blobs) with whole-image jitter and an optional 3-px dark mover."""
    rng = np.random.default_rng(seed)
    big = np.full((h + 40, w + 40), 170, np.float32)
    for _ in range(25):
        cv2.circle(big, (int(rng.integers(0, w + 40)), int(rng.integers(0, h + 40))), int(rng.integers(10, 50)),
                   float(rng.integers(185, 230)), -1)
    big = cv2.GaussianBlur(big, (0, 0), 2)                   # cloud edges sharp enough for jitter to matter
    frames, truth = [], []
    for i in range(n):
        dx, dy = rng.uniform(-jitter, jitter, 2)
        m = np.float32([[1, 0, dx - 20], [0, 1, dy - 20]])     # shift left/up so the crop has no border
        f = cv2.warpAffine(big, m, (w + 40, h + 40))[:h, :w].copy()
        pos = None
        if target:
            pos = (60 + 4 * i, 100 + i)
            f[pos[1]:pos[1] + 3, pos[0]:pos[0] + 3] = 60
        f += rng.normal(0, 1.5, f.shape).astype(np.float32)
        frames.append(cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        truth.append(pos)
    return frames, truth


class Motion(unittest.TestCase):
    def test_finds_small_mover_despite_jitter(self):
        # (2.5, 3) is the case where an unclipped gradient let the dark target hijack registration.
        for jitter, seed, kw in [(1.5, 0, {}), (2.5, 3, {}), (2.5, 3, {"scale": 0.5}), (2.5, 3, {"refine": True})]:
            with self.subTest(jitter=jitter, seed=seed, **kw):
                frames, truth = synthetic_sky(jitter=jitter, seed=seed)
                cue, hits, spurious, scored = MotionCue(**{"scale": 1.0, **kw}), 0, 0, 0
                for i, (f, t) in enumerate(zip(frames, truth)):
                    props = cue(f)
                    if i < 15:
                        continue
                    scored += 1
                    near = [p for p in props if abs(p[0] - t[0]) < 8 and abs(p[1] - t[1]) < 8]
                    hits += bool(near)
                    spurious += len(props) - len(near)
                self.assertGreaterEqual(hits / scored, 0.9)
                self.assertLessEqual(spurious / scored, 0.5)

    def test_stabilisation_suppresses_jitter(self):
        frames, _ = synthetic_sky(target=False, jitter=2.5, seed=3)
        on, off = MotionCue(scale=1.0), MotionCue(scale=1.0, stabilise=False)
        n_on = sum(len(on(f)) for f in frames)
        n_off = sum(len(off(f)) for f in frames)
        self.assertLess(n_on, n_off / 5 + 1)


def changing_sky(n=380, h=360, w=480, jitter=2.0, seed=0, fade=(40, 300)):
    """Mount jitter while one cloud field cross-fades into an unrelated one (the weather changing), with a
    3-px dark mover crossing every 100 frames."""
    rng = np.random.default_rng(seed)

    def clouds():
        big = np.full((h + 40, w + 40), 170, np.float32)
        for _ in range(25):
            cv2.circle(big, (int(rng.integers(0, w + 40)), int(rng.integers(0, h + 40))), int(rng.integers(10, 50)),
                       float(rng.integers(185, 230)), -1)
        return cv2.GaussianBlur(big, (0, 0), 2)
    a, b = clouds(), clouds()
    frames, truth = [], []
    for i in range(n):
        mix = float(np.clip((i - fade[0]) / (fade[1] - fade[0]), 0, 1))
        dx, dy = rng.uniform(-jitter, jitter, 2)
        f = cv2.warpAffine((1 - mix) * a + mix * b, np.float32([[1, 0, dx - 20], [0, 1, dy - 20]]),
                           (w + 40, h + 40))[:h, :w].copy()
        pos = (40 + (i % 100) * 4, 100 + (i % 100))
        f[pos[1]:pos[1] + 3, pos[0]:pos[0] + 3] = 60
        f += rng.normal(0, 1.5, f.shape).astype(np.float32)
        frames.append(cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        truth.append(pos)
    return frames, truth


class MotionLongRun(unittest.TestCase):
    def test_reference_follows_a_changing_sky(self):
        # With the first frame kept as the reference for ever (ref_alpha=0) registration fails once the clouds
        # have changed: hit rate 0.44 and 56 false blobs per frame here (16 #54).
        frames, truth = changing_sky()
        cue, hits, spurious, scored = MotionCue(scale=1.0), 0, 0, 0
        for i, (f, t) in enumerate(zip(frames, truth)):
            props = cue(f)
            if i < 320:                                     # score only after the sky has fully changed
                continue
            scored += 1
            near = [p for p in props if abs(p[0] - t[0]) < 8 and abs(p[1] - t[1]) < 8]
            hits += bool(near)
            spurious += len(props) - len(near)
        self.assertGreaterEqual(hits / scored, 0.9)
        self.assertLessEqual(spurious / scored, 2.0)


class Replay(unittest.TestCase):
    def test_motion_only_replay_on_a_file(self):
        frames, _ = synthetic_sky(n=30)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "clip.avi")
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (480, 360))
            for f in frames:
                vw.write(f)
            vw.release()
            rows = replay(path, scale=1.0, quiet=True)["rows"]
        self.assertTrue(any(r[1] == "motion" for r in rows))


class Program(unittest.TestCase):
    def test_camera_and_px4_path(self):
        cam = Camera(w=1280, h=720, fx=3000.0, tilt_deg=0.0)
        cx, cy, wpx, rng = cam.project((10.0, 100.0, 5.0), 0.35)
        self.assertAlmostEqual(cx, 640 + 3000 * 10 / 100)                  # east is image right
        self.assertAlmostEqual(cy, 360 - 3000 * 5 / 100)                   # up is image up
        self.assertAlmostEqual(wpx, 3000 * 0.35 / rng)                     # w = f S / R (doc 13)
        self.assertIsNone(cam.project((0.0, -50.0, 0.0), 0.35))            # behind the camera
        with tempfile.TemporaryDirectory() as d:                           # PX4 logs are north, east, DOWN, microseconds
            p = Path(d) / "vehicle_local_position_0.csv"
            p.write_text("timestamp,x,y,z" + chr(10) + "1000000,100,10,-20" + chr(10) + "3000000,60,10,-20" + chr(10))
            path = from_px4_csv(p, camera_ned=(0.0, 0.0, -5.0))
            np.testing.assert_allclose(path(0.0), (10, 100, 15))          # east, north, up relative to the camera
            np.testing.assert_allclose(path(1.0), (10, 80, 15))

    def test_whole_program_on_a_simulated_scene(self):
        # A drone closing at 30 m/s from 153 m, a far hovering drone and a flapping bird crossing; mount shake,
        # blur and sensor noise; the stand-in detector on motion-cued crops; no classifier, so all "unknown".
        # Expected: three tracks; ONLY the closing one escalates (unknown and closing: look at it); TTC about right.
        targets = [Target("closing", approach((-25.0, 150.0, 30.0), 30.0)),
                   Target("hover", hover((30.0, 220.0, 30.0))),
                   Target("bird", line((-35.0, 150.0, 12.0), (9.0, 0.0, 0.5)), size_m=0.9, cls=1, flap_hz=5.0, shade=75.0)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "scene.avi")
            truth = write_video(path, render(targets, seconds=3.5))
            res = replay(path, weights="blob", quiet=True, events_path=os.path.join(d, "events.jsonl"))
            n_lines = len(Path(d, "events.jsonl").read_text().splitlines())
        self.assertEqual(res["summary"]["tracks"], 3)
        self.assertEqual(n_lines, len(res["events"]))
        ids = {e["track_id"] for e in res["events"]}
        self.assertEqual(len(ids), 1)                                      # the hoverer and the bird stay log_only
        self.assertEqual([e["event"] for e in res["events"]], ["slew_ptz_for_identification"])
        event = res["events"][0]
        gt = next(g for g in truth[event["frame"]]["targets"] if g["name"] == "closing")
        self.assertLess(abs(event["box"][0] - gt["box"][0]) + abs(event["box"][1] - gt["box"][1]), 10)
        closing = next(k for i, k in res["tracks"].items() if i == event["track_id"])
        self.assertAlmostEqual(closing["min_ttc_s"], truth[-1]["targets"][0]["range_m"] / 30.0, delta=0.6)   # at the end
        self.assertLess(res["summary"]["crops_per_frame"], 4)              # cued crops, not a full scan every frame

    def test_stand_in_detector_ignores_cloud_and_crop_edges(self):
        (_, frame, truth), = list(render([Target("d", hover((0.0, 150.0, 20.0)))], seconds=1 / 30))
        (det,), = BlobDetector()([frame])
        self.assertLess(abs(det[0] - truth[0][0]) + abs(det[1] - truth[0][1]), 3)
        x = int(truth[0][0]) + 2                                           # cut the target with the crop's edge
        self.assertEqual(BlobDetector()([np.ascontiguousarray(frame[:, x:])]), [[]])


class ZoomOut(unittest.TestCase):
    def test_mosaic_geometry(self):
        imgs = [np.full((512, 640, 3), 50 * n, np.uint8) for n in range(4)]
        boxes = [[(320, 256, 24, 20, n)] for n in range(4)]
        canvas, bl = zoomout.mosaic(imgs, boxes, 2, 640, 512)
        self.assertEqual(canvas.shape, (512, 640, 3))
        self.assertEqual(int(canvas[400, 500, 0]), 150)                    # 4th image lands bottom-right
        self.assertEqual(bl[3], (160 + 320, 128 + 256, 12, 10, 3))         # halved and offset

    def test_make_split_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "val"
            (src / "images").mkdir(parents=True)
            (src / "labels").mkdir()
            for n in range(10):
                cv2.imwrite(str(src / "images" / f"{n}.jpg"), np.full((512, 640, 3), 128, np.uint8))
                (src / "labels" / f"{n}.txt").write_text(tiling.yolo_lines([(300, 200, 30, 24, 0)], 640, 512))
            self.assertEqual(zoomout.make_split(src, Path(d) / "k3", 3), 1)   # 10 images -> one 3x3 mosaic
            (gts, _), = run_split(lambda imgs: [[] for _ in imgs], Path(d) / "k3")
            self.assertEqual(len(gts), 9)
            self.assertTrue(all(abs(g[2] - 10) < 0.5 for g in gts))           # 30 px -> 10 px


class Datasets(unittest.TestCase):
    def test_converter_skips_unlabelled_tail_and_bird_background(self):
        import types
        fake = types.ModuleType("mcos_decoder")
        # 40-frame videos: target absent for frames 0-4, labelled 5-24, and labels END at frame 25.
        fake.load_groundtruth = lambda p: [None] * 5 + [(100.0, 200.0, 9.0, 7.0)] * 20 + [None]
        sys.modules["mcos_decoder"] = fake
        try:
            from cuas.datasets.halmstad import convert
            with tempfile.TemporaryDirectory() as d:
                src = Path(d) / "Video_V"
                src.mkdir()
                for name in ["V_DRONE_001", "V_DRONE_002", "V_BIRD_001", "V_BIRD_002"]:
                    vw = cv2.VideoWriter(str(src / f"{name}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 25, (640, 512))
                    for _ in range(40):
                        vw.write(np.full((512, 640, 3), 180, np.uint8))
                    vw.release()
                    (src / f"{name}.avi").rename(src / f"{name}.mp4")
                convert(src, Path(d) / "out", every=5)
                names = [p.stem for p in (Path(d) / "out").rglob("*.jpg")]
                convert(src, Path(d) / "out_neg", every=5, neg_every=20)
                names_neg = [p.stem for p in (Path(d) / "out_neg").rglob("*.jpg")]
        finally:
            del sys.modules["mcos_decoder"]
        frames = {int(n.split("_")[3]) for n in names}
        self.assertTrue(frames <= set(range(26)))                         # nothing past the labels' end
        self.assertNotIn(0, frames)                                       # default: no unlabelled "background"
        neg = {n for n in names_neg if n.split("_")[3] == "00000"}
        self.assertTrue(neg and all(n.startswith("V_DRONE") for n in neg))  # opt-in, never from bird videos

    def test_antiuav_converter(self):
        import json as _json
        from cuas.datasets.antiuav import convert as convert_antiuav
        with tempfile.TemporaryDirectory() as d:
            seq = Path(d) / "val" / "20190925_000000_1_1"
            seq.mkdir(parents=True)
            vw = cv2.VideoWriter(str(seq / "visible.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 20, (320, 180))
            for _ in range(12):
                vw.write(np.full((180, 320, 3), 120, np.uint8))
            vw.release()
            (seq / "visible.avi").rename(seq / "visible.mp4")
            exist = [1] * 8 + [0] * 4
            rects = [[100, 50, 20, 12]] * 8 + [[]] * 4
            (seq / "visible.json").write_text(_json.dumps({"exist": exist, "gt_rect": rects}))
            counts = convert_antiuav(d, "val", Path(d) / "out", every=4, absent_every=10)
            self.assertEqual((counts["with_drone"], counts["absent"]), (2, 1))   # frames 0, 4 | frame 10
            lbl = (Path(d) / "out" / "labels" / "20190925_000000_1_1_00004.txt").read_text()
            np.testing.assert_allclose(tiling.read_yolo_labels(lbl, 320, 180)[0], (100, 50, 20, 12, 0), atol=1e-3)
            self.assertEqual((Path(d) / "out" / "labels" / "20190925_000000_1_1_00010.txt").read_text(), "")

    def test_antiuav_overlay_removed_moving_target_kept(self):
        from cuas.datasets.antiuav import inpaint_split, overlay_mask
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "images").mkdir()
            (Path(d) / "labels").mkdir()
            for n in range(30):
                f = np.full((180, 320, 3), 128, np.uint8) + rng.integers(0, 3, (180, 320, 3), dtype=np.uint8)
                cv2.putText(f, "AZ 286.9", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)  # fixed OSD
                x, y = 60 + 7 * n, 90 + (n % 5) * 6
                f[y:y + 6, x:x + 10] = 30                                                          # moving drone
                cv2.imwrite(str(Path(d) / "images" / f"{n:02d}.png"), f)
                (Path(d) / "labels" / f"{n:02d}.txt").write_text(tiling.yolo_lines([(x, y, 10, 6, 0)], 320, 180))
            mask = overlay_mask(sorted((Path(d) / "images").iterdir()), dilate=3)
            self.assertTrue(mask[5:25, 5:90].any())                       # the text is in the mask
            stats = inpaint_split(d, mask)
            self.assertEqual(stats["drones_touched_by_mask"], 0)           # moving targets are not overlay
            out = cv2.imread(str(Path(d) / "images" / "10.png"))
            self.assertLess(int(out[5:25, 5:90].max()), 200)               # white text gone
            self.assertLess(int(out[90 + 0 * 6:96, 130:140].min()), 60)    # the drone at frame 10 is still dark

    def test_dut_voc_converter(self):
        from cuas.datasets.dut import convert as convert_dut
        xml = ("<annotation><size><width>320</width><height>180</height><depth>3</depth></size>"
               "<object><name>UAV</name><bndbox><xmin>100</xmin><ymin>50</ymin><xmax>120</xmax><ymax>62</ymax></bndbox></object>"
               "</annotation>")
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "test"
            (src / "img").mkdir(parents=True)
            (src / "xml").mkdir()
            cv2.imwrite(str(src / "img" / "00001.jpg"), np.zeros((180, 320, 3), np.uint8))
            cv2.imwrite(str(src / "img" / "00002.jpg"), np.zeros((180, 320, 3), np.uint8))   # no annotation
            (src / "xml" / "00001.xml").write_text(xml)
            stats = convert_dut(src, Path(d) / "out")
            self.assertEqual((stats["images"], stats["images_without_xml"], stats["drones"]), (1, 1, 1))
            lbl = (Path(d) / "out" / "labels" / "00001.txt").read_text()
            np.testing.assert_allclose(tiling.read_yolo_labels(lbl, 320, 180)[0], (100, 50, 20, 12, 0), atol=1e-3)
            with self.assertRaises(SystemExit):
                convert_dut(src / "img", Path(d) / "out2")                  # no XML anywhere: fail loudly

    def test_split_by_video_is_stratified_and_deterministic(self):
        paths = [f"V_DRONE_{i:03d}.mp4" for i in range(10)] + [f"V_BIRD_{i:03d}.mp4" for i in range(5)] + ["x.mp4"]
        s = split_videos(paths, 0.2, seed=0)
        self.assertEqual(s, split_videos(paths, 0.2, seed=0))
        self.assertNotIn("x.mp4", s)
        self.assertEqual(sum(v == "val" for k, v in s.items() if "DRONE" in k), 2)
        self.assertEqual(sum(v == "val" for k, v in s.items() if "BIRD" in k), 1)


def _stream(tracks_fn, fps, seconds, score=0.9, w=10):
    """[(t, dets)] from a function t -> list of (cx, cy) centres."""
    out = []
    for k in range(int(fps * seconds)):
        t = k / fps
        out.append((t, [(cx - w / 2, cy - w / 2, w, w, 0, score) for cx, cy in tracks_fn(t)]))
    return out


class Tracking(unittest.TestCase):
    def test_kalman_learns_velocity(self):
        kf, rng = KalmanCV(0, 0, 4.0, 1e6), np.random.default_rng(0)
        for k in range(1, 31):
            kf.predict(0.1, 100.0)
            kf.update((100.0 * k * 0.1 + rng.normal(0, 1), 50.0 + rng.normal(0, 1)), 1.0)
        self.assertAlmostEqual(kf.x[2], 100.0, delta=5.0)
        self.assertAlmostEqual(kf.x[3], 0.0, delta=5.0)

    def test_fast_crossers_keep_identity_at_10fps(self):
        # Two targets 300 px/frame apart in motion (doc 18 section 3.1): boxes never overlap between
        # frames, so IoU matching would fail. Their paths cross at t = 0.5 s.
        fn = lambda t: [(100 + 3000 * t, 500 + 200 * t), (3100 - 3000 * t, 700 - 200 * t)]
        tr = Tracker()
        for t, dets in _stream(fn, 10, 2.0):
            tr.update(t, dets)
        tracks = tr.all_tracks()
        self.assertEqual(len(tracks), 2)
        for track in tracks:
            vx = track.rows()[:, 9]
            self.assertEqual(len(set(np.sign(vx[3:]))), 1)     # never jumped onto the other target
            self.assertGreaterEqual(len(track.history), 19)

    def test_low_score_detections_continue_but_never_start_tracks(self):
        tr = Tracker(high=0.5, low=0.1)
        for k in range(10):
            t = k / 30
            s = 0.9 if k < 4 else 0.2                             # score drops after the track is confirmed
            active = tr.update(t, [(100 + 30 * k, 100, 10, 10, 0, s), (900, 900, 10, 10, 0, 0.2)])
        self.assertEqual(len(active), 1)
        self.assertEqual(len(active[0].history), 10)
        self.assertEqual(len(tr.tracks), 1)                       # the low-score blob at (900, 900) never became one

    def test_tentative_track_cannot_take_a_confirmed_tracks_detection(self):
        # A new track has a huge covariance, so one joint assignment gave IT the detection (16 #54).
        tr = Tracker()
        for k in range(5):
            tr.update(k / 30, [(100 + 10 * k, 100, 10, 10, 0, 0.9)])
        tr.update(5 / 30, [(150, 100, 10, 10, 0, 0.9), (160, 108, 10, 10, 0, 0.9)])    # clutter next to the track
        self.assertEqual([(k.confirmed, k.hits) for k in tr.tracks], [(True, 6), (False, 1)])
        tr.update(6 / 30, [(160, 108, 10, 10, 0, 0.9)])       # one detection, inside both gates
        self.assertEqual([(k.confirmed, k.hits) for k in tr.tracks], [(True, 7), (False, 1)])

    def test_history_can_be_bounded_for_live_use(self):
        tr = Tracker(history_s=1.0, keep_finished=False)
        for k in range(300):                                   # 10 s on one target
            tr.update(k / 30, [(100 + k, 100, 10, 10, 0, 0.9)])
        track, = tr.tracks
        self.assertLessEqual(len(track.history), 32)
        self.assertEqual((track.hits, track.t0), (300, 0.0))   # age and count survive the trimming
        tr.update(20.0, [])
        self.assertEqual((tr.tracks, tr.finished), ([], []))

    def test_coasting_and_timeout(self):
        tr = Tracker(max_coast=0.5)
        for k in range(5):
            tr.update(k / 10, [(100 + 20 * k, 100, 10, 10)])
        tr.update(0.8, [(100 + 20 * 8, 100, 10, 10)])             # 0.4 s gap: re-associated after coasting
        self.assertEqual(len(tr.tracks), 1)
        self.assertEqual(tr.tracks[0].hits, 6)
        tr.update(1.5, [])                                        # 0.7 s without a detection: finished
        self.assertEqual((len(tr.tracks), len(tr.finished)), (0, 1))
        with self.assertRaises(ValueError):
            tr.update(1.0, [])

    def test_track_features_motion_and_flap(self):
        tr = Tracker()
        for t, dets in _stream(lambda t: [(100 + 200 * t, 300)], 30, 3.0):
            tr.update(t, dets)
        f = track_features(tr.all_tracks()[0].rows())
        self.assertGreater(f["straightness"], 0.98)
        self.assertEqual(f["hover_frac"], 0.0)
        self.assertAlmostEqual(f["speed_bl_med"], 20.0, delta=2.0)     # 200 px/s over a 10 px target
        # Hovering target with an 8 Hz shape oscillation (a flapping bird, seen at 30 fps).
        rows = []
        for k in range(90):
            t = k / 30
            h = 10 * (1 + 0.3 * np.sin(2 * np.pi * 8 * t))
            rows.append([t, 300, 300, 10, h, 0.9, 1, 300, 300, 0, 0])
        f = track_features(np.array(rows))
        self.assertEqual(f["hover_frac"], 1.0)
        self.assertAlmostEqual(f["aspect_peak_hz"], 8.0, delta=0.3)
        self.assertEqual(f["cls_drone_frac"], 0.0)

    def test_silhouette_both_polarities(self):
        img = np.full((21, 21), 200, np.uint8)
        img[8:13, 7:14] = 40                                      # dark 5 x 7 target on bright sky
        area, contrast = silhouette(img)
        self.assertEqual(area, 35)
        self.assertGreater(contrast, 3)
        area, _ = silhouette(255 - img)                           # bright target on dark sky
        self.assertEqual(area, 35)


def _manoeuvre(seed=0):
    """Cruise at 200 px/s, a hard turn (2,000 px/s^2 sideways for 0.5 s), then cruise again; 30 fps, 1 px noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 3, 1 / 30)
    pos, vel, out = np.array([100.0, 300.0]), np.array([200.0, 0.0]), []
    for k, tk in enumerate(t):
        if k:
            acc = np.array([0.0, 2000.0]) if 1.0 <= tk < 1.5 else np.zeros(2)
            vel = vel + acc / 30
            pos = pos + vel / 30
        out.append((tk, pos.copy(), pos + rng.normal(0, 1, 2)))
    return out


class M2(unittest.TestCase):
    def test_imm_follows_a_manoeuvre_better_than_cv(self):
        err = {}
        for name, make in (("cv", lambda z: KalmanCV(*z, 1.0, 1e6)), ("imm", lambda z: IMMFilter(*z, 1.0, 1e6))):
            data, f, e, mu_ca = _manoeuvre(), None, [], []
            for k, (tk, truth, z) in enumerate(data):
                if f is None:
                    f = make(z)
                    continue
                f.predict(1 / 30, 1e3)
                f.update(z, 1.0)
                if 1.0 <= tk < 2.0:
                    e.append(np.hypot(*(f.x[:2] - truth)))
                if name == "imm":
                    mu_ca.append((tk, f.mu[1]))
            err[name] = float(np.sqrt(np.mean(np.square(e))))
        self.assertLess(err["imm"], err["cv"])
        cruise = np.mean([m for tk, m in mu_ca if 0.5 <= tk < 1.0])
        turn = max(m for tk, m in mu_ca if 1.0 <= tk < 1.6)
        self.assertGreater(turn, cruise + 0.2)          # the manoeuvre model takes over during the turn
        tr = Tracker(motion="imm")                       # and the tracker accepts it
        for tk, _, z in _manoeuvre():
            tr.update(tk, [(z[0] - 5, z[1] - 5, 10, 10)])
        self.assertEqual(len(tr.all_tracks()), 1)
        with self.assertRaises(ValueError):
            Tracker(motion="spline")

    def test_ttc_from_expansion(self):
        # 0.35 m drone closing from 200 m at 20 m/s, f = 10,000 px: w = f S / R.
        t = np.arange(0, 2.0001, 1 / 30)
        w = 10000 * 0.35 / (200 - 20 * t)
        self.assertAlmostEqual(ttc_expansion(t, w), (200 - 20 * 2.0) / 20, delta=0.1)   # 8 s at t = 2 s
        self.assertIsNone(ttc_expansion(t, 10000 * 0.35 / (200 + 20 * t)))          # receding
        self.assertIsNone(ttc_expansion(t[:3], w[:3]))                               # too short

    def test_ttc_ignores_box_jitter_on_a_new_track(self):
        # Five boxes of a target that is NOT closing, widths jittering 5%: the fitted slope is noise. Before the
        # significance test, 75% of such tracks read "closing, TTC under 10 s" within their first 3 s (16 #54).
        rng, false_ttc = np.random.default_rng(0), 0
        for _ in range(500):
            ttc = ttc_expansion(np.arange(5) / 30, 12 * (1 + rng.normal(0, 0.05, 5)))
            false_ttc += ttc is not None and ttc < 10
        self.assertLess(false_ttc / 500, 0.05)
        false_ttc = 0                                        # and a 2-s track, read at every frame as the pipeline does
        for _ in range(200):
            t, w = np.arange(60) / 30, 12 * (1 + rng.normal(0, 0.05, 60))
            false_ttc += any((v := ttc_expansion(t[:k], w[:k])) is not None and v < 10 for k in range(5, 61))
        self.assertLess(false_ttc / 200, 0.03)
        t = np.arange(0, 1.0001, 1 / 30)                                             # a real approach still reads
        w = 10000 * 0.35 / (200 - 20 * t) * (1 + rng.normal(0, 0.02, len(t)))
        self.assertAlmostEqual(ttc_expansion(t, w), 9.0, delta=2.5)

    def test_classifier_calibration_unknown_and_save(self):
        rng = np.random.default_rng(0)
        n = 400
        y = rng.integers(0, 2, n)
        X = rng.normal(0, 1, (n, len(FEATURE_NAMES)))
        X[:, 0] = y * 1.5 + rng.normal(0, 1, n)           # one informative feature, overlapping classes
        X[:, 5] = np.nan                                  # a feature never measured is dropped, not fatal
        groups = np.arange(n) // 4                        # 4 samples per "video"
        clf = TrackClassifier(n_splits=4).fit(X, y, groups)
        self.assertLess(ece(clf.oof_prob, y), ece(clf.oof_raw, y) + 0.02)   # cross-fitted: must not make it worse
        self.assertAlmostEqual(clf.drone_share(), y.mean())
        del clf.prior                                     # an older pickle: recovered from the calibrator
        self.assertAlmostEqual(clf.drone_share(), y.mean())
        p = clf.predict_proba(X)
        self.assertTrue(np.all((p >= 0) & (p <= 1)))
        labels = clf.decide(p, X, np.full(n, 5.0))
        self.assertTrue(set(labels) <= {"drone", "not_drone", "unknown"})
        self.assertTrue(np.all(clf.decide(p, X, np.full(n, 0.5)) == "unknown"))   # too short to judge
        far = X[:1].copy()
        far[0, :] = 50.0                                  # far outside anything seen in training
        self.assertEqual(clf.decide([0.99], far, [5.0])[0], "unknown")
        with tempfile.TemporaryDirectory() as d:
            clf.save(Path(d) / "c.pkl")
            np.testing.assert_allclose(TrackClassifier.load(Path(d) / "c.pkl").predict_proba(X), p)

    def test_rules_floor_matches_doc12(self):
        base = {"ttc_s": None, "drone_prob": 0.0, "in_protected_zone": False, "remote_id_authorised": False}
        r = lambda **kw: rules_floor({**base, **kw})
        self.assertEqual(r(), "log_only")
        self.assertEqual(r(drone_prob=0.8), "slew_ptz_for_identification")
        self.assertEqual(r(drone_prob=0.8, in_protected_zone=True), "alert_operator")          # loitering in zone
        self.assertEqual(r(drone_prob=0.8, ttc_s=2.0), "request_engagement_decision")          # decide now
        # A spoofed "authorised" Remote ID must not relax an imminent track, inside or outside the zone.
        self.assertEqual(r(drone_prob=0.8, ttc_s=6.0, remote_id_authorised=True), "alert_operator")
        self.assertEqual(r(ttc_s=6.0, in_protected_zone=True, remote_id_authorised=True), "alert_operator")
        self.assertEqual(r(drone_prob=0.8, remote_id_authorised=True), "log_only")             # not imminent
        # Not yet classified is not "not a drone": an unknown track closing on the camera is looked at, not
        # logged and forgotten (16 #54). A TTC clamped to 0 (due now) is the most imminent case, not a non-case.
        self.assertEqual(r(drone_prob=None, ttc_s=2.0), "slew_ptz_for_identification")
        self.assertEqual(r(drone_prob=0.4, label="unknown", ttc_s=6.0), "slew_ptz_for_identification")
        self.assertEqual(r(drone_prob=0.9, label="unknown", ttc_s=2.0), "slew_ptz_for_identification")   # p not trusted
        self.assertEqual(r(drone_prob=0.9, label="drone", ttc_s=2.0), "request_engagement_decision")
        self.assertEqual(r(drone_prob=0.1, label="not_drone", ttc_s=2.0), "log_only")
        self.assertEqual(r(drone_prob=None), "log_only")                                        # unknown, not closing
        self.assertEqual(r(drone_prob=0.8, ttc_s=0.0), "request_engagement_decision")
        self.assertEqual(combine("alert_operator", {"suggested": "log_only", "confidence": 0.99}), "alert_operator")
        self.assertEqual(combine("log_only", {"suggested": "alert_operator", "confidence": 0.5}), "log_only")
        self.assertEqual(combine("log_only", {"suggested": "launch", "confidence": 0.99}), "log_only")

    def test_pipeline_end_to_end(self):
        class Always:                                     # stand-in classifier: fixed p(drone)
            def predict_proba(self, X):
                return np.array([0.9])

            def decide(self, p, X, d):
                return np.array(["drone"])
        gray = np.full((480, 640), 200, np.uint8)
        pipe = TrackPipeline(classifier=Always())
        states = []
        for k in range(60):                               # 2 s at 30 fps, target growing: closing fast
            t, w = k / 30, 10 * 5 / (5 - k / 30)          # TTC = 5 s at t = 0, 3 s at t = 2 s
            box = (320 - w / 2, 240 - w / 2, w, w)
            g = gray.copy()
            x0, y0 = int(box[0]), int(box[1])
            g[y0:y0 + int(w), x0:x0 + int(w)] = 40
            states = pipe.update(t, [(*box, 0, 0.9)], g)
        self.assertEqual(len(states), 1)
        s = states[0]
        self.assertEqual((s["label"], s["drone_prob"]), ("drone", 0.9))
        self.assertAlmostEqual(s["ttc_s"], 5 - 59 / 30, delta=0.15)
        self.assertEqual(s["response"], "alert_operator")     # imminent drone, TTC just above 3 s
        self.assertTrue(pipe._sil[s["track_id"]])             # silhouettes were measured

    def test_pipeline_is_bounded_and_merges_cross_class_duplicates(self):
        pipe = TrackPipeline()                                # no classifier: every track stays "unknown"
        for k in range(150):                                  # 5 s; the same object boxed as drone AND helicopter
            x = 100 + 2 * k
            states = pipe.update(k / 30, [(x, 100, 10, 10, 0, 0.6), (x + 1, 100, 10, 10, 3, 0.7)])
        self.assertEqual(len(pipe.tracker.tracks), 1)                       # one object, one track
        s, = states
        self.assertEqual((s["label"], s["drone_prob"], s["n_obs"]), ("unknown", None, 150))
        self.assertAlmostEqual(s["duration_s"], 149 / 30)
        self.assertLessEqual(len(pipe.tracker.tracks[0].history), 80)       # ~2.5 s kept, not 5
        pipe.update(60.0, [])                                               # the track times out
        self.assertEqual((pipe._cache, pipe._sil, pipe.tracker.finished), ({}, {}, []))

    def test_sequential_decision(self):
        # Evidence per second of flight, not per reading: read 6 times a second or once a second, the same 3 s of
        # p = 0.8 give the same evidence (1.5 windows' worth).
        fast, slow = SequentialDecision(prior=0.5), SequentialDecision(prior=0.5)
        for k in range(1, 19):
            fast.update("a", k / 6, 0.8, age=k / 6)
        for k in range(1, 4):
            slow.update("a", float(k), 0.8, age=float(k))
        self.assertAlmostEqual(fast.evidence("a"), slow.evidence("a"))
        self.assertAlmostEqual(slow.evidence("a"), 1.5 * np.log(4))
        # Declared only once enough has accumulated: p = 0.95 is 2.94 per window, short of 3.0 after one window
        d = SequentialDecision(accept=3.0, prior=0.5)
        self.assertEqual(d.update(1, 2.0, 0.95, age=2.0), "unknown")
        self.assertEqual(d.update(1, 2.5, 0.95), "drone")
        # A reading that cannot be trusted adds nothing; the training prior is taken out of p
        before = d.evidence(1)
        self.assertEqual(d.update(1, 3.0, 0.01, usable=False), "drone")
        self.assertEqual(d.evidence(1), before)
        p8 = SequentialDecision(prior=0.8)
        p8.update(2, 2.0, 0.8, age=2.0)
        self.assertAlmostEqual(p8.evidence(2), 0.0)
        # Clamped both ways, and calibrated 0 / 1 do not give infinite evidence
        c = SequentialDecision(accept=3.0, margin=2.0)
        for k in range(1, 30):
            c.update(3, k, 1.0, age=k)
        self.assertEqual(c.evidence(3), 5.0)
        for k in range(30, 60):
            c.update(3, k, 0.0)
        self.assertEqual((c.evidence(3), c.label(3)), (-3.0, "not_drone"))
        c.drop(3)
        self.assertEqual((c.evidence(3), c.label(3)), (0.0, "unknown"))

    def test_pipeline_with_sequential_decision(self):
        class Always:                                     # p(drone) 0.9 in every window: 2.2 of evidence per 2 s
            def predict_proba(self, X):
                return np.array([0.9])

            def decide(self, p, X, d):
                return np.array(["drone"])
        pipe = TrackPipeline(classifier=Always(), decision=SequentialDecision(accept=3.0, prior=0.5))
        for k in range(60):                               # 2 s: every window says drone, the evidence is not enough
            states = pipe.update(k / 30, [(100 + k, 100, 10, 10, 0, 0.9)])
        s, = states
        self.assertEqual((s["window_label"], s["label"]), ("drone", "unknown"))
        for k in range(60, 90):                           # 1 s more and it is
            states = pipe.update(k / 30, [(100 + k, 100, 10, 10, 0, 0.9)])
        s, = states
        self.assertEqual(s["label"], "drone")
        pipe.update(60.0, [])                             # the track ends: its evidence is dropped
        self.assertEqual(pipe.decision._llr, {})

    def test_centre_matching_for_tiny_targets(self):
        gt = [(100.0, 100.0, 6.0, 6.0, 0)]
        off = [(102.5, 102.5, 6.0, 6.0, 0, 0.9)]              # 2.5 px off both ways: IoU 0.20, centre still on the target
        r = evaluate([(gt, off)])
        self.assertEqual((r["n_hit"][1], r["false_drone_on_drone"]), (0, 1))          # IoU rule: a miss
        r = evaluate([(gt, off)], center_below=12)
        self.assertEqual((r["n_hit"][1], r["false_drone_on_drone"], r["false_drone_on_background"]), (1, 0, 0))
        far = [(112.0, 100.0, 6.0, 6.0, 0, 0.9)]              # 12 px away: a miss under either rule
        self.assertEqual(evaluate([(gt, far)], center_below=12)["n_hit"][1], 0)
        big = [(100.0, 100.0, 40.0, 40.0, 0)]                 # a 40-px drone keeps the IoU rule
        loose = [(118.0, 118.0, 8.0, 8.0, 0, 0.9)]            # a small box at its centre: IoU 0.04
        self.assertEqual(sum(evaluate([(big, loose)], center_below=12)["n_hit"]), 0)
        two = [(100.0, 100.0, 6.0, 6.0, 0), (108.0, 100.0, 6.0, 6.0, 0)]
        preds = [(100.0, 100.0, 6.0, 6.0, 0, 0.9), (108.0, 100.0, 6.0, 6.0, 0, 0.8)]
        self.assertEqual(evaluate([(two, preds)], center_below=12)["n_hit"][1], 2)    # each takes its nearest

    def test_cut_and_paste_a_drone(self):
        from cuas.paste import cut, paste
        rng = np.random.default_rng(0)
        sky = np.clip(180 + rng.normal(0, 2, (120, 160, 3)), 0, 255).astype(np.uint8)
        sky[50:56, 60:84] = 40                                # a dark drone: body ...
        sky[44:62, 69:75] = 40                                # ... and arms (a cross, 24 x 18 px)
        box = (60, 44, 24, 18)
        patch = cut(sky, box)
        rgb, alpha, (bx, by, bw, bh), background = patch
        self.assertEqual((bw, bh), (24, 18))
        self.assertTrue(np.allclose(background, 180, atol=2))
        self.assertGreater(alpha[by + 8, bx + 12], 0.95)      # on the drone
        self.assertLess(alpha[by + 1, bx + 1], 0.05)          # sky inside the box stays behind
        self.assertEqual(float(alpha[0].max()), 0.0)          # nothing from the margin comes along
        ground = rng.integers(120, 160, (200, 200, 3)).astype(np.uint8)         # textured, a little darker
        out, (x, y, w, h) = paste(ground, patch, 100, 80, scale=0.5)
        self.assertEqual((x, y, w, h), (100, 80, 12, 9))
        changed = np.argwhere((out != ground).any(axis=2))
        self.assertTrue(changed[:, 0].min() >= 78 and changed[:, 0].max() <= 91)  # only around the new box
        self.assertTrue(changed[:, 1].min() >= 98 and changed[:, 1].max() <= 114)
        self.assertLess(out[84, 106].mean(), 70)              # the drone is there, dark on grey
        ring = out[78:92, 98:114][(out[78:92, 98:114] != ground[78:92, 98:114]).any(axis=2)]
        self.assertLess(ring.max(), 165)                      # no halo of the old, brighter sky around it
        self.assertIsNone(paste(np.full((200, 200, 3), 42, np.uint8), patch, 100, 80))   # a day drone on a night scene
        self.assertIsNone(paste(np.full((200, 200, 3), 125, np.uint8), (rgb, alpha * 0.02, (bx, by, bw, bh), background), 100, 80))   # would not stand out
        self.assertIsNone(paste(ground, patch, 195, 80))      # would leave the image
        clutter = rng.integers(0, 255, (120, 160, 3)).astype(np.uint8)
        clutter[50:56, 60:84] = 40
        self.assertIsNone(cut(clutter, box))                  # no plain surroundings: no matte

    def test_training_degradation(self):
        import random
        from cuas.degrade import Degrade
        img = np.full((64, 64, 3), 128, np.uint8)
        img[28:36, 28:36] = 30                                # a dark 8-px target
        self.assertTrue(np.array_equal(Degrade(p_blur=0, p_noise=0, p_jpeg=0)(img), img))      # p = 0: untouched
        random.seed(0), np.random.seed(0)
        out = Degrade(p_blur=1, p_noise=1, p_jpeg=1)({"img": img.copy(), "cls": 7})
        self.assertEqual((out["img"].shape, out["img"].dtype, out["cls"]), (img.shape, np.uint8, 7))
        self.assertGreater(np.abs(out["img"].astype(int) - img).mean(), 0.5)                   # really degraded
        self.assertLess(out["img"][30:34, 30:34].mean(), out["img"][:16, :16].mean() - 30)     # target survives
        random.seed(0), np.random.seed(0)
        again = Degrade(p_blur=1, p_noise=1, p_jpeg=1)(img.copy())
        self.assertTrue(np.array_equal(again, out["img"]))                                     # seeded = repeatable
        clean = sum(np.array_equal(Degrade()(img), img) for _ in range(400)) / 400
        self.assertTrue(0.08 < clean < 0.30, clean)           # about 1 image in 5 stays clean (0.6 * 0.5 * 0.6)

    def test_dvb_annotations(self):
        ann = parse_annotations("0 1 10 20 5 4 drone\n1 0\n3 2 1 2 3 4 drone 50 60 7 8 Bird\n")
        self.assertEqual(ann[3][1], (50.0, 60.0, 7.0, 8.0, "bird"))
        stream = list(label_stream(ann, fps=25))
        self.assertEqual([b for _, b in stream][2], [])           # frame 2 has no line
        self.assertAlmostEqual(stream[3][0], 3 / 25)
        with self.assertRaises(ValueError):
            parse_annotations("0 2 1 2 3 4 drone")


class Temporal(unittest.TestCase):
    """E23 frame stacks (cuas/temporal.py)."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.sky = cv2.normalize(cv2.GaussianBlur((rng.random((540, 960)) * 255).astype(np.uint8), (0, 0), 3),
                                 None, 0, 255, cv2.NORM_MINMAX)

    def test_register_undoes_a_camera_shift(self):
        from cuas.temporal import register
        moved = cv2.warpAffine(self.sky, np.float32([[1, 0, 6], [0, 1, -3]]), self.sky.shape[::-1],
                               borderMode=cv2.BORDER_REPLICATE)
        out, (dx, dy) = register(self.sky, moved)
        self.assertAlmostEqual(dx, -6, delta=0.5)
        self.assertAlmostEqual(dy, 3, delta=0.5)
        inner = (slice(20, -20), slice(20, -20))
        self.assertLess(np.abs(out[inner].astype(int) - self.sky[inner]).mean(), 2.0)

    def test_stack_keeps_frame_t_and_shows_only_the_mover(self):
        from cuas.temporal import stack, static
        frames = []
        for x in (100, 104, 108):                       # a dark 3-px target moving 4 px per step on a still sky
            f = self.sky.copy()
            f[130:133, x:x + 3] = 0
            frames.append(f)
        st, shifts = stack(*frames)
        self.assertTrue(np.array_equal(st[:, :, 2], frames[2]))         # channel 2 is the current frame, untouched
        self.assertLess(max(abs(v) for s in shifts for v in s), 0.5)    # the target does not drag the registration
        diff = np.abs(st[:, :, 2].astype(int) - st[:, :, 0])
        self.assertGreater(diff[130:133, 100:111].max(), 100)           # the motion is there ...
        diff[128:135, 98:113] = 0
        self.assertLess(diff.max(), 30)                                 # ... and nowhere else
        s = static(st)
        self.assertTrue(np.array_equal(s[:, :, 0], frames[2]) and np.array_equal(s[:, :, 1], frames[2]))


class RFDETRWrapper(unittest.TestCase):
    """cuas.detector.RFDETRDetector around a stand-in model (rfdetr itself is not needed)."""

    def make(self):
        from types import SimpleNamespace
        from cuas.detector import RFDETRDetector
        seen = []

        def predict(images, threshold):
            seen.extend(images)
            return [SimpleNamespace(xyxy=np.array([[10., 20., 30., 50.], [0., 0., 5., 5.]]),
                                    class_id=np.array([0, 4]), confidence=np.array([0.9, 0.8])) for _ in images]
        det = object.__new__(RFDETRDetector)
        det.model, det.conf, det.res = SimpleNamespace(predict=predict), 0.05, 640
        det.names = dict(enumerate(["drone", "bird", "airplane", "helicopter"]))
        return det, seen

    def test_pads_without_scaling_and_drops_the_background_slot(self):
        det, seen = self.make()
        frame = np.zeros((512, 640, 3), np.uint8)
        frame[:, :, 0] = 200                                    # blue in BGR
        out = det([frame])
        self.assertEqual(out, [[(10.0, 20.0, 20.0, 30.0, 0, 0.9)]])     # x, y, w, h; class 4 = background, dropped
        self.assertEqual(seen[0].shape, (640, 640, 3))
        self.assertEqual(tuple(seen[0][0, 0]), (0, 0, 200))     # RGB for the model
        self.assertEqual(tuple(seen[0][600, 0]), (114, 114, 114))       # padded below, not stretched

    def test_refuses_an_image_it_would_have_to_shrink(self):
        det, _ = self.make()
        with self.assertRaises(ValueError):
            det([np.zeros((1080, 1920, 3), np.uint8)])


if __name__ == "__main__":
    unittest.main()
