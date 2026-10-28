"""
Motion cueing for a fixed camera (docs 07 section 4 and 13 section 3.1): propose where to look, cheaply,
so the CNN only runs on crops around movers plus a periodic full scan.

Two things make this harder than textbook background subtraction:
  * Mount jitter. At 16 mm on the Pi HQ camera 1 px is 0.0056 deg, so wind shakes the whole image by
    pixels. Each frame is registered to the background before comparison. Plain phase correlation is
    only ~1 px accurate on low-texture sky (flat regions dominate it), so it runs on gradient magnitude
    (~0.2 px on the synthetic test sky). `refine=True` adds ECC alignment (~0.03 px, ~4x slower).
    On a real site the static skyline strip (trees, roofs) carries most of the usable texture.
  * Tiny targets. Morphological opening would erase a 2-px drone, so blobs are only dilated (merged),
    never eroded, and filtered by area.
  * A scene that changes. The registration reference follows the scene (a slow blend of registered frames).
    Kept as the first frame, it stops resembling the sky once the clouds have changed: on a synthetic sky
    that cross-fades to new clouds, hit rate fell to 0.44 with 56 false blobs per frame (16 #54).
"""
import cv2
import numpy as np


def _grad(img):
    """Gradient magnitude, clipped at its 99th percentile. Without the clip, one high-contrast target
    (a dark drone on bright sky) outweighs every cloud edge, registration locks onto the drone instead of
    the scene, and the background tears (synthetic test: hit rate 0.42 and 3,078 false blobs vs 1.0 and 5)."""
    m = cv2.magnitude(cv2.Sobel(img, cv2.CV_32F, 1, 0), cv2.Sobel(img, cv2.CV_32F, 0, 1))
    return np.minimum(m, np.percentile(m, 99))


class MotionCue:
    def __init__(self, scale: float = 0.5, var_threshold: float = 16, history: int = 300,
                 min_area: int = 1, max_area: int = 4000, mask=None, stabilise: bool = True, refine: bool = False,
                 warmup: int = 10, ref_alpha: float = 0.02, ref_every: int = 10, max_shift: float = 16.0):
        """scale: work on a downscaled grey frame (0.5 halves a 10-px drone to 5 px, still visible).
        var_threshold: MOG2 squared-Mahalanobis threshold; lower finds fainter targets and more noise.
        mask: optional uint8 image, full frame size, 255 where to look (sky), 0 to ignore (ground, trees).
        ref_alpha: share of each registered frame blended into the registration reference (0 = keep the first
        frame for ever); its gradient image is refreshed every `ref_every` frames.
        max_shift: px at the working scale. Mount jitter is a few px; a larger shift means the reference has
        followed drifting cloud, so reference and background model restart from the current frame (`warmup`
        frames without proposals). Untuned: needs footage from the real mount."""
        self.scale, self.min_area, self.max_area = scale, min_area, max_area
        self.stabilise, self.refine, self.warmup = stabilise, refine, warmup
        self.ref_alpha, self.ref_every, self.max_shift = ref_alpha, ref_every, max_shift
        self._new_bg = lambda: cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold,
                                                                  detectShadows=False)
        self.bg = self._new_bg()
        self.mask_src, self.mask = mask, None
        self.ref, self.n = None, 0
        self.last_shift = (0.0, 0.0)

    def __call__(self, frame) -> list:
        """Proposal boxes (x, y, w, h) in full-frame pixels."""
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(grey, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        if self.mask is None and self.mask_src is not None:
            self.mask = cv2.resize(self.mask_src, small.shape[::-1], interpolation=cv2.INTER_NEAREST)

        dx = dy = 0.0
        if self.stabilise:
            f = small.astype(np.float32)
            if self.ref is None:
                self.ref, self.ref_grad = f, _grad(f)
            else:
                (dx, dy), _ = cv2.phaseCorrelate(self.ref_grad, _grad(f))
                if self.refine:
                    warp = np.float32([[1, 0, dx], [0, 1, dy]])
                    try:
                        _, warp = cv2.findTransformECC(self.ref, f, warp, cv2.MOTION_TRANSLATION,
                                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4),
                                                       None, 5)
                        dx, dy = float(warp[0, 2]), float(warp[1, 2])
                    except cv2.error:
                        pass                     # ECC did not converge: keep the phase-correlation estimate
                if max(abs(dx), abs(dy)) > self.max_shift:      # the reference has drifted with the scene: restart
                    self.ref, self.ref_grad, self.bg, self.n, dx, dy = f, _grad(f), self._new_bg(), 0, 0.0, 0.0
                else:
                    # Move the frame back onto the reference so the background model sees a still scene.
                    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
                    small = cv2.warpAffine(small, m, small.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
                    if self.ref_alpha:
                        cv2.accumulateWeighted(cv2.warpAffine(f, m, small.shape[::-1], borderMode=cv2.BORDER_REPLICATE),
                                               self.ref, self.ref_alpha)
                        if self.n % self.ref_every == 0:
                            self.ref_grad = _grad(self.ref)
        self.last_shift = (dx / self.scale, dy / self.scale)

        fg = self.bg.apply(small)
        self.n += 1
        if self.n <= self.warmup:
            return []
        if self.mask is not None:
            fg = cv2.bitwise_and(fg, self.mask)
        fg = cv2.dilate(fg, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        s = 1 / self.scale
        return [((x + dx) * s, (y + dy) * s, w * s, h * s)
                for x, y, w, h, area in stats[1:n] if self.min_area <= area <= self.max_area]
