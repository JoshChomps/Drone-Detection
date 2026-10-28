"""
Image degradation for training (E22): what a small sensor and a video link do to a distant drone.

E21 measured that a detector trained on clean frames loses its small targets under moderate blur, noise and
compression (~8.5 px: recall 0.93 -> 0.08; 19 section 3.2). The zoom-out mosaics average noise away, so the
detector had never seen a noisy small drone. `Degrade` is applied to each training image AFTER the geometric
augmentation, so the noise is at the final pixel scale: optics blur, then sensor noise, then compression.

    install()      adds Degrade to the Ultralytics training pipeline (train_local.py --degrade)

The evaluation keeps degradations the training never saw (e21_degradation.py: motion blur, signal-dependent noise,
WebP), so the gain is not measured only on the generator that produced the training data (the E2 lesson, 17 #5).
"""
import random

import cv2
import numpy as np


def gaussian_blur(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def gaussian_noise(img, sigma):
    rng = np.random.default_rng(np.random.randint(2 ** 31))     # seeded from numpy's state, which Ultralytics
    noise = rng.standard_normal(img.shape, dtype=np.float32)    # seeds per worker; float32 is ~3x faster
    return np.clip(img + sigma * noise, 0, 255).astype(np.uint8)


def jpeg(img, quality):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])[1], cv2.IMREAD_COLOR)


class Degrade:
    """Random blur / noise / JPEG on a uint8 BGR image. Each is applied independently with its probability, so
    about 1 image in 5 stays clean with the defaults. Callable on an image, or on an Ultralytics label dict."""

    def __init__(self, p_blur=0.4, blur=(0.3, 1.5), p_noise=0.5, noise=(1.0, 8.0), p_jpeg=0.4, quality=(30, 95)):
        self.p_blur, self.blur, self.p_noise, self.noise = p_blur, blur, p_noise, noise
        self.p_jpeg, self.quality = p_jpeg, quality

    def image(self, img):
        if random.random() < self.p_blur:
            img = gaussian_blur(img, random.uniform(*self.blur))
        if random.random() < self.p_noise:
            img = gaussian_noise(img, random.uniform(*self.noise))
        if random.random() < self.p_jpeg:
            img = jpeg(img, random.uniform(*self.quality))
        return img

    def __call__(self, labels):
        if isinstance(labels, dict):
            labels["img"] = self.image(labels["img"])
            return labels
        return self.image(labels)

    def __repr__(self):
        return (f"Degrade(blur p={self.p_blur} sigma={self.blur}, noise p={self.p_noise} sigma={self.noise}, "
                f"jpeg p={self.p_jpeg} quality={self.quality})")


def install(**kw):
    """Append Degrade to every training pipeline Ultralytics builds from now on in this process (also the one it
    rebuilds when mosaic closes). Validation images are untouched. Returns the transform."""
    import ultralytics.data.dataset as ds
    if getattr(ds.v8_transforms, "_cuas_degrade", None):
        return ds.v8_transforms._cuas_degrade
    original, degrade = ds.v8_transforms, Degrade(**kw)

    def v8_transforms(*args, **kwargs):
        pipeline = original(*args, **kwargs)
        pipeline.append(degrade)
        return pipeline

    v8_transforms._cuas_degrade = degrade
    ds.v8_transforms = v8_transforms
    print(f"training images degraded: {degrade}")
    return degrade
