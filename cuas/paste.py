"""
Cut a real drone out of one frame and paste it into another (E22c).

Every experiment so far left the unseen dataset where it was (recall 0.12-0.16 on DUT). One reading: the drone
class is tied to the scenes it was filmed in. A Halmstad drone is always in Halmstad sky and an Anti-UAV drone in
an Anti-UAV scene, so the detector never has to separate the object from its surroundings. Pasting each dataset's
drones into the other's scenes (and later into our own target-free footage) breaks that tie with real pixels. The
2025 Drone-vs-Bird winner pasted drone patches the same way (19 section 9).

A rectangular patch would carry a rectangle of the old background with it, and the detector would learn the
rectangle. So the drone is cut with a soft matte, which is only possible where it was filmed against something
plain (sky): `cut` refuses the rest. `paste` refuses a spot where the drone would not be visible, and one whose
brightness is far from where the drone was filmed: a night drone (lights on black) pasted into a daylight sky is
not something a camera ever sees (the first contact sheet was full of them, 16 #69).

Edge pixels are a mix of drone and old background. They are un-mixed with the old background colour before
compositing (out = patch + (1 - alpha) * (new - old background)), so no halo of the old sky comes along.

    patch = cut(frame, box)                       # None if the surroundings are not plain enough for a matte
    out, new_box = paste(tile, patch, x, y, scale)   # None if it would not stand out there

Degradation (blur, noise, compression) is NOT applied here: train_local.py --degrade does it on the composed
image, so the pasted drone and its new background are degraded together and no edge betrays the paste.
"""
import cv2
import numpy as np


def cut(img, box, margin: float = 0.3, max_noise: float = 12.0, ramp: float = 15.0):
    """(patch BGR uint8, alpha float32 in [0, 1], box inside the patch, old background BGR) for the object in
    `box` = (x, y, w, h, ...), or None when no clean matte exists: surroundings not plain, or the object hardly
    differs from them."""
    x, y, w, h = box[:4]
    H, W = img.shape[:2]
    m = max(3.0, margin * max(w, h))
    x0, y0, x1, y1 = int(max(0, x - m)), int(max(0, y - m)), int(min(W, x + w + m)), int(min(H, y + h + m))
    bx0, by0, bx1, by1 = int(round(x)) - x0, int(round(y)) - y0, int(round(x + w)) - x0, int(round(y + h)) - y0
    patch = img[y0:y1, x0:x1]
    if bx1 - bx0 < 3 or by1 - by0 < 3 or patch.shape[0] <= by1 - by0 or patch.shape[1] <= bx1 - bx0:
        return None
    ring = np.ones(patch.shape[:2], bool)
    ring[max(by0, 0):by1, max(bx0, 0):bx1] = False
    if ring.sum() < 20:
        return None
    background = np.median(patch[ring].astype(np.float32), axis=0)
    dist = np.linalg.norm(patch.astype(np.float32) - background, axis=2)
    noise = float(np.percentile(dist[ring], 90))
    if noise > max_noise:                                   # clutter behind the drone: no matte
        return None
    alpha = np.clip((dist - noise) / ramp, 0, 1)
    inside = np.zeros_like(ring)
    inside[max(by0 - 2, 0):by1 + 2, max(bx0 - 2, 0):bx1 + 2] = True
    alpha[~inside] = 0                                      # nothing of the old background comes along
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 0.7)
    core = alpha[max(by0, 0):by1, max(bx0, 0):bx1]
    if core.max() < 0.8 or not 0.05 <= core.mean() <= 0.9:  # a ghost, or a "drone" that fills its whole box
        return None
    return patch.copy(), alpha, (bx0, by0, bx1 - bx0, by1 - by0), background


def paste(dst, patch, x: float, y: float, scale: float = 1.0, min_contrast: float = 10.0, max_shift: float = 60.0):
    """Composite `patch` (from cut) into a copy of dst with the object's box at (x, y), shrunk by `scale` <= 1.
    Returns (image, (x, y, w, h)), or None if the box leaves the image, the object would not stand out there, or
    the spot's brightness differs from the object's old background by more than `max_shift` grey levels."""
    rgb, alpha, (bx, by, bw, bh), background = patch
    if scale < 1:
        size = (max(1, int(round(rgb.shape[1] * scale))), max(1, int(round(rgb.shape[0] * scale))))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
        bx, by, bw, bh = bx * scale, by * scale, bw * scale, bh * scale
    px, py = int(round(x - bx)), int(round(y - by))
    ph, pw = alpha.shape
    H, W = dst.shape[:2]
    if px < 0 or py < 0 or px + pw > W or py + ph > H:
        return None
    region = dst[py:py + ph, px:px + pw].astype(np.float32)
    if abs(float(region.mean()) - float(background.mean())) > max_shift:
        return None                                         # e.g. lights filmed at night, pasted into a day sky
    a = alpha[..., None]
    out_region = np.where(a > 0.02, rgb.astype(np.float32) + (1 - a) * (region - background), region)
    solid = alpha > 0.5
    if solid.sum() < 2 or float(np.abs(out_region - region).mean(axis=2)[solid].mean()) < min_contrast:
        return None                                         # e.g. a dark drone on a dark roof: an unlabelled smudge
    out = dst.copy()
    out[py:py + ph, px:px + pw] = np.clip(out_region, 0, 255).astype(np.uint8)
    return out, (px + bx, py + by, bw, bh)
