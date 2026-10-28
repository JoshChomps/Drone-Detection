"""
Detector interface. Everything else in `cuas` sees only this call signature, so the detector can be
swapped (e.g. an Apache-2.0 model instead of AGPL Ultralytics) without touching the rest of the pipeline.

    detector(list_of_bgr_images) -> per image: [(x, y, w, h, cls, score), ...] in that image's pixels
"""
import numpy as np


class UltralyticsDetector:
    """Ultralytics YOLO (e.g. YOLO26s) behind the common interface. `ultralytics` is imported lazily."""

    def __init__(self, weights: str, conf: float = 0.05, imgsz: int = 640, device=None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.names = self.model.names
        self.kw = {"conf": conf, "imgsz": imgsz, "verbose": False, **({"device": device} if device is not None else {})}

    def __call__(self, images, batch: int = 8):
        # Fixed-size chunks: a frame can need dozens of crops, and every new batch size grew host memory (a 1,050-
        # frame cluttered clip reached 6.6 GB and ran the laptop out of memory; field test, 16 #93).
        out = []
        for i in range(0, len(images), batch):
            out += self._predict(images[i:i + batch])
        return out

    def _predict(self, images):
        out = []
        for r in self.model.predict(images, **self.kw):
            b = r.boxes
            xyxy, cls, score = b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy()
            # Plain Python numbers: numpy float32 is not JSON-serialisable (lost a Colab M2 run, 16 #50).
            out.append([(float(x0), float(y0), float(x1 - x0), float(y1 - y0), int(c), float(s))
                        for (x0, y0, x1, y1), c, s in zip(xyxy.tolist(), cls, score)])
        return out


class RFDETRDetector:
    """RF-DETR (E25: Apache-2.0 for sizes N-L, windowed DINOv2 backbone, patch 16) behind the common interface.
    `rfdetr` is imported lazily. RF-DETR resizes every input to a square, so each image is padded (grey 114,
    bottom/right) to resolution x resolution first: a 640 x 512 Halmstad frame is not stretched and no tile is scaled.
    Inputs larger than `resolution` must be tiled by the caller, as the evaluation scripts already do."""

    VARIANTS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}

    def __init__(self, weights: str, conf: float = 0.05, variant: str = "small", resolution: int = 640,
                 names=None, device=None):
        import rfdetr
        kw = {"pretrain_weights": weights, "resolution": resolution, **({"device": str(device)} if device is not None
                                                                         and str(device) != "0" else {})}
        self.model = getattr(rfdetr, self.VARIANTS[variant])(**kw)
        self.names = dict(enumerate(names)) if names else dict(enumerate(self.model.class_names))
        self.conf, self.res = conf, resolution

    def __call__(self, images):
        padded = []
        for im in images:
            h, w = im.shape[:2]
            if h > self.res or w > self.res:
                raise ValueError(f"RFDETRDetector: {w}x{h} input is larger than {self.res}; tile it first")
            c = np.full((self.res, self.res, 3), 114, np.uint8)
            c[:h, :w] = im
            padded.append(np.ascontiguousarray(c[:, :, ::-1]))      # BGR -> RGB
        res = self.model.predict(padded, threshold=self.conf)
        res = res if isinstance(res, list) else [res]
        n = len(self.names)       # the head has one more output, "background" (id n), which is not a detection
        return [[(float(x0), float(y0), float(x1 - x0), float(y1 - y0), int(c), float(s))
                 for (x0, y0, x1, y1), c, s in zip(d.xyxy.tolist(), d.class_id.tolist(), d.confidence.tolist())
                 if 0 <= int(c) < n]
                for d in res]


def load(weights: str, conf: float = 0.05, imgsz: int = 640, device=None):
    """The detector for a weights file: RF-DETR when its run folder holds cuas_rfdetr.json (written by
    experiments/train_rfdetr.py), else Ultralytics. Evaluation scripts call this, so an RF-DETR run is scored
    exactly like a YOLO run."""
    import json
    from pathlib import Path
    marker = Path(weights).parent.parent / "cuas_rfdetr.json"
    if marker.exists():
        m = json.loads(marker.read_text())
        return RFDETRDetector(weights, conf=conf, variant=m["variant"], resolution=m["resolution"],
                              names=m.get("names"), device=device)
    return UltralyticsDetector(weights, conf=conf, imgsz=imgsz, device=device)


def nms(dets, iou_thr: float = 0.5, class_aware: bool = True) -> list:
    """NMS over frame-coordinate detections, for duplicates from overlapping crops. Class-aware by default, so
    evaluation still sees a drone boxed as "airplane" (cuas.eval). The tracker wants class_aware=False: one
    object boxed under two classes must not start two tracks."""
    from cuas.eval import iou
    keep = []
    for d in sorted(dets, key=lambda d: -d[5]):
        if all((class_aware and k[4] != d[4]) or iou(k, d) < iou_thr for k in keep):
            keep.append(d)
    return keep


def detect_in_crops(frame, origins, detector, tile: int = 640) -> list:
    """Run the detector on tile-sized crops at `origins` and return deduplicated frame-coordinate boxes."""
    if not origins:
        return []
    crops = [np.ascontiguousarray(frame[y:y + tile, x:x + tile]) for x, y in origins]
    dets = []
    for (ox, oy), per_crop in zip(origins, detector(crops)):
        dets += [(x + ox, y + oy, w, h, c, s) for x, y, w, h, c, s in per_crop]
    return nms(dets)
