"""
Temporal detector input (E23, 19 section 5 rank 2): three registered grey frames t-2k, t-k, t as the three image
channels, so the pretrained first layer still applies and the network can see pixel motion that one frame cannot
show (YOLOMG, arXiv 2503.07115: RGB + motion difference raised AP 0.33 -> 0.78 on ARD100).

Channel order follows OpenCV's BGR: B = t-2k, G = t-k, R = t (the current frame). Past frames only, so a live
camera can build the same stack. Past frames are aligned to frame t by phase correlation on clipped gradient
magnitude (cuas.motion's registration), translation only.

Storage: JPEG with 4:2:0 chroma subsampling halves the resolution of channel differences, which is where the motion
is, so stacks are written with 4:4:4 sampling (JPEG_444). Training must not use hue or saturation augmentation either
(it mixes channels): train_local.py --temporal.
"""
import cv2
import numpy as np

from cuas.motion import _grad

JPEG_444 = [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]


def shift_to(ref_grey, grey, scale: float = 0.5):
    """(dx, dy) that moves `grey` onto `ref_grey`, estimated at `scale` on clipped gradient magnitude."""
    a = cv2.resize(ref_grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(_grad(a), _grad(b))
    return -dx / scale, -dy / scale


def register(ref_grey, grey, scale: float = 0.5):
    """`grey` translated onto `ref_grey` (edges replicated). Returns (image, (dx, dy))."""
    dx, dy = shift_to(ref_grey, grey, scale)
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(grey, m, grey.shape[::-1], borderMode=cv2.BORDER_REPLICATE), (dx, dy)


def stack(past2, past1, now, scale: float = 0.5):
    """Three grey frames (t-2k, t-k, t) -> one 3-channel uint8 image with the past frames registered to t.
    Returns (image, [(dx, dy) of t-2k, (dx, dy) of t-k])."""
    r2, s2 = register(now, past2, scale)
    r1, s1 = register(now, past1, scale)
    return np.dstack([r2, r1, now]), [s2, s1]


def static(stacked):
    """The control input (E23ctl): the current frame (channel 2 of a stack) in all three channels, so a model sees
    the same pixels as E23 without the motion."""
    g = stacked[:, :, 2]
    return np.dstack([g, g, g])
