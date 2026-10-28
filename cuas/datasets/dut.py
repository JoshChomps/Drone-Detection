"""
DUT Anti-UAV detection subset -> Ultralytics split (all objects are class 0, drone). Used only as a HELD-OUT test (E15).

Facts from the paper (Zhao et al., "Vision-based Anti-UAV Detection and Tracking", arXiv 2205.10851): 10,000 images,
train 5,200 / val 2,600 / test 2,200; resolutions from 160x240 to 5616x3744; 35+ UAV types; day, night, dawn, dusk.
The annotation format is not stated in the paper; secondary sources describe one Pascal-VOC-style XML per image
(<size>, <object><bndbox>xmin ymin xmax ymax). convert() fails loudly if it finds no such annotations.

    python -m cuas.datasets.dut --src C:/cuas_data/dut_raw/test --dst C:/cuas_data/dut_test
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from cuas.tiling import yolo_lines

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def read_voc(text: str):
    """(width, height, [(x, y, w, h, 0), ...]) from one VOC XML annotation."""
    root = ET.fromstring(text)
    size = root.find("size")
    w, h = int(float(size.findtext("width"))), int(float(size.findtext("height")))
    boxes = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        x0, y0, x1, y1 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((x0, y0, x1 - x0, y1 - y0, 0))
    return w, h, boxes


def convert(src, dst) -> dict:
    """Pair every image under src with the XML of the same stem (anywhere under src); copy the image and write YOLO
    labels. Images without an XML are skipped and counted. Returns size statistics for the report."""
    src, dst = Path(src), Path(dst)
    xmls = {p.stem: p for p in src.rglob("*.xml")}
    images = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    if not xmls:
        raise SystemExit(f"No .xml annotations under {src}. Look at the download's layout and adapt read_voc/convert.")
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    widths, img_w, no_xml, n = [], [], 0, 0
    for img in images:
        if img.stem not in xmls:
            no_xml += 1
            continue
        w, h, boxes = read_voc(xmls[img.stem].read_text(encoding="utf-8", errors="replace"))
        shutil.copy2(img, dst / "images" / img.name)
        (dst / "labels" / f"{img.stem}.txt").write_text(yolo_lines(boxes, w, h))
        widths += [b[2] for b in boxes]
        img_w.append(w)
        n += 1
    q = lambda v: np.quantile(v, [0, 0.1, 0.5, 0.9, 1]).round(1).tolist() if v else []  # noqa: E731
    return {"images": n, "images_without_xml": no_xml, "drones": len(widths),
            "drone_width_px_min_p10_p50_p90_max": q(widths), "image_width_px_min_p10_p50_p90_max": q(img_w)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DUT Anti-UAV detection split (VOC XML) -> Ultralytics split")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    print(convert(a.src, a.dst))
