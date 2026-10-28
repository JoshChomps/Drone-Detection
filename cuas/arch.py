"""
E18: optional architecture add-ons for Ultralytics YOLO26, after YOLOBirDrone (arXiv 2601.08319, 01 §2.8).

The paper releases no code and leaves layer positions, channel counts, the attention formulas and the fusion
unspecified, so these modules are OUR reading of its text, not a reproduction:

* AELAN  -> `DeformConv`: the 3x3 convolution in each Bottleneck of the chosen C3k2 blocks samples at learned
            offsets (DCNv1, the paper's equation; torchvision `deform_conv2d`).
* MPDA   -> `MPDA(reverse=False)` after the P3 blocks: four progressive branches with receptive fields 3/5/7/9
            (X1 = f3(X), X2 = f3(X1), X3 = f5(X1), X4 = f5(X2), the paper's eqs.), spatial attention on X1/X2,
            channel attention on X3/X4, channel attention after concatenation (eq. 8).
* RMPDA  -> `MPDA(reverse=True)` after the P4 blocks: the same with the attention types swapped.
* ECA    -> a cheap control: channel attention alone after the P3 blocks.

Choices the paper does not make, made so that fine-tuning keeps the COCO-pretrained detector:
* Every add-on STARTS AS THE IDENTITY. The attention blocks are residual (x + fuse(...)) with the fuse BN scale
  at zero, and the deformable offsets start at zero. At step 0 the model computes exactly what yolo26s.pt does.
* Original parameter names are kept (new ones are added), so `model.load(yolo26s.pt)` transfers every weight.
* Spatial attention is CBAM-style (mean/max over channels -> 7x7 conv); channel attention is ECA. The paper names
  EPSANet and "EECA" without formulas.

Usage: `apply_arch(detection_model, "aelan+mpda+rmpda")`, or train with `ArchTrainer` and a model whose
`yaml["cuas_arch"]` names the add-ons (experiments/train_local.py --arch). The spec travels inside checkpoints, so
resume and evaluation rebuild the same model. Loading a checkpoint needs this module importable (`cuas` on the
path). Layer indices are for the YOLO26 detect layout (yolo26.yaml, all scales).
"""
import math

import torch
from torch import nn
from torchvision.ops import deform_conv2d
from ultralytics.nn.modules import C3k2, Conv
from ultralytics.nn.modules.block import Bottleneck

# yolo26.yaml: 4 = P3 backbone, 6 = P4 backbone, 16 = P3 head (small-object output), 19 = P4 head.
# P5 already carries self-attention (C2PSA at 10, attention C3k2 at 22), so nothing is added there.
LAYERS = {"aelan": (4, 6, 16, 19), "mpda": (4, 16), "rmpda": (6, 19), "eca": (4, 16)}
TOKENS = tuple(LAYERS)


class ECA(nn.Module):
    """Efficient channel attention: global average pool -> 1-D conv across channels -> sigmoid gate."""

    def __init__(self, c, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(c) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        y = x.mean((2, 3), keepdim=True)                                # B,C,1,1
        y = self.conv(y.squeeze(-1).transpose(-1, -2))                  # B,1,C
        return x * y.transpose(-1, -2).unsqueeze(-1).sigmoid()


class SpatialAttn(nn.Module):
    """Spatial attention (CBAM-style): channel mean and max -> k x k conv -> sigmoid gate per pixel."""

    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        return x * self.conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)).sigmoid()


class MPDA(nn.Module):
    """Multi-scale progressive dual attention (reverse=True: RMPDA). Residual, identity at initialisation."""

    def __init__(self, c, reverse=False):
        super().__init__()
        cb = max(c // 4, 8)
        self.b1, self.b2 = Conv(c, cb, 3), Conv(cb, cb, 3)             # receptive fields 3, 5
        self.b3, self.b4 = Conv(cb, cb, 5), Conv(cb, cb, 5)            # 7 (= 3 then 5), 9 (= 3, 3, 5)
        fine, coarse = (lambda: ECA(cb), SpatialAttn) if reverse else (SpatialAttn, lambda: ECA(cb))
        self.att = nn.ModuleList([fine(), fine(), coarse(), coarse()])
        self.agg = SpatialAttn() if reverse else ECA(4 * cb)
        self.fuse = Conv(4 * cb, c, 1, act=False)
        nn.init.zeros_(self.fuse.bn.weight)                            # add-on starts as the identity

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x1)
        y = torch.cat([a(t) for a, t in zip(self.att, (x1, x2, self.b3(x1), self.b4(x2)))], 1)
        return x + self.fuse(self.agg(y))


class ResidualECA(nn.Module):
    """ECA as a residual add-on: x + s * (ECA(x) - x), with the learned scale s starting at 0."""

    def __init__(self, c):
        super().__init__()
        self.eca = ECA(c)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.scale * (self.eca(x) - x)


class DeformConv(nn.Module):
    """Stand-in for an Ultralytics `Conv` (conv -> BN -> act) whose convolution samples at learned offsets (DCNv1).

    Keeps the parameter names `conv.weight` and `bn.*`, adds `offset.*`; offsets start at zero, so it starts
    identical to the Conv it replaces. Deliberately NOT a `Conv` subclass: `DetectionModel.fuse()` would otherwise
    fold it into a plain nn.Conv2d and silently drop the offsets.
    """

    def __init__(self, src: Conv):
        super().__init__()
        c = src.conv
        if c.groups != 1 or c.dilation != (1, 1):
            raise ValueError("DeformConv supports groups=1, dilation=1 only")
        self.stride, self.padding = c.stride, c.padding
        self.conv = nn.Module()                                          # holds `weight` under the original name
        self.conv.weight = nn.Parameter(c.weight.detach().clone())
        kh, kw = c.kernel_size
        self.offset = nn.Conv2d(c.in_channels, 2 * kh * kw, (kh, kw), c.stride, c.padding)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.bn, self.act = src.bn, src.act

    def forward(self, x):
        off = self.offset(x)
        if off.dtype != x.dtype:                                         # under autocast the conv may run in half
            off = off.to(x.dtype)
        y = deform_conv2d(x, off, self.conv.weight.to(x.dtype), stride=self.stride, padding=self.padding)
        return self.act(self.bn(y))


class C3k2Post(C3k2):
    """C3k2 followed by an add-on (`self.post`). Keeps every original parameter name."""

    def forward(self, x):
        return self.post(super().forward(x))

    def forward_split(self, x):                                          # the exporter swaps forward for this
        return self.post(super().forward_split(x))


def parse_spec(spec):
    toks = [t for t in (spec or "").lower().replace(",", "+").split("+") if t]
    bad = [t for t in toks if t not in TOKENS]
    if bad:
        raise ValueError(f"unknown add-on(s) {bad}; choose from {TOKENS}")
    if "eca" in toks and "mpda" in toks:
        raise ValueError("'eca' is the cheap alternative to 'mpda' at the same layers; use one")
    return tuple(t for t in TOKENS if t in toks)                        # canonical order


def _out_channels(block):
    return block.cv2.conv.out_channels


def apply_arch(model, spec):
    """Add the named add-ons to an Ultralytics DetectionModel (YOLO26 layout) in place; returns it.
    Idempotent per model: a model already carrying a spec is left alone if the spec matches."""
    toks = parse_spec(spec)
    canon = "+".join(toks)
    if any(isinstance(m, (C3k2Post, DeformConv)) for m in model.modules()):
        have = model.yaml.get("cuas_arch", "") if isinstance(getattr(model, "yaml", None), dict) else "?"
        if have != canon:
            raise ValueError(f"model already has add-ons '{have}', asked for '{canon}'")
        return model
    layers = model.model
    for t in toks:
        for i in LAYERS[t]:
            if not isinstance(layers[i], C3k2):
                raise TypeError(f"layer {i} is {type(layers[i]).__name__}, expected C3k2 (YOLO26 detect layout)")
    for t in toks:
        for i in LAYERS[t]:
            blk = layers[i]
            if t == "aelan":
                for b in blk.modules():
                    if isinstance(b, Bottleneck) and b.cv2.conv.kernel_size == (3, 3):
                        b.cv2 = DeformConv(b.cv2)
                continue
            post = {"mpda": lambda c: MPDA(c), "rmpda": lambda c: MPDA(c, reverse=True),
                    "eca": ResidualECA}[t](_out_channels(blk))
            blk.__class__ = C3k2Post
            blk.post = post
    if isinstance(getattr(model, "yaml", None), dict):
        model.yaml["cuas_arch"] = canon
    return model


def build_model(cfg, nc, ch=3, weights=None, verbose=False):
    """DetectionModel from `cfg`, with the add-ons named in cfg['cuas_arch'] applied BEFORE loading `weights`, so
    both a plain pretrained checkpoint (new parameters keep their identity init) and an E18 checkpoint (every
    parameter restored) load correctly."""
    from ultralytics.nn.tasks import DetectionModel
    model = DetectionModel(cfg, nc=nc, ch=ch, verbose=verbose)
    spec = cfg.get("cuas_arch", "") if isinstance(cfg, dict) else ""
    if spec:
        apply_arch(model, spec)
    if weights:
        model.load(weights, verbose=verbose)
    return model


def _make_trainer():
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import RANK

    class ArchTrainer(DetectionTrainer):
        """DetectionTrainer that rebuilds the E18 add-ons named in the model yaml (fresh start and resume)."""

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = build_model(cfg, nc=self.data["nc"], ch=self.data["channels"], weights=None,
                                verbose=verbose and RANK == -1)
            model = self.set_model_names_for_load(model)
            if weights:
                model.load(weights)
            return model

    return ArchTrainer


def __getattr__(name):                                                  # lazy: ultralytics' trainer imports are heavy
    if name == "ArchTrainer":
        globals()["ArchTrainer"] = _make_trainer()
        return globals()["ArchTrainer"]
    raise AttributeError(name)


def register_onnx_deform():
    """Map torchvision's deform_conv2d to the ONNX DeformConv operator (opset >= 19); torchvision ships no ONNX
    exporter for it. Call before torch.onnx.export(..., opset_version=19) of an 'aelan' model."""
    from torch.onnx import register_custom_op_symbolic
    from torch.onnx.symbolic_helper import _get_tensor_sizes, parse_args

    @parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "b")
    def sym(g, x, w, off, mask, bias, sh, sw, ph, pw, dh, dw, n_wg, n_og, use_mask):
        kh, kw = _get_tensor_sizes(w)[2:]
        ins = [x, w, off, bias] + ([mask] if use_mask else [])
        return g.op("DeformConv", *ins, dilations_i=[dh, dw], group_i=n_wg, kernel_shape_i=[kh, kw],
                    offset_group_i=n_og, pads_i=[ph, pw, ph, pw], strides_i=[sh, sw])

    register_custom_op_symbolic("torchvision::deform_conv2d", sym, 19)


def added_parameters(model):
    """Parameter names the add-ons introduced (absent from a plain YOLO26)."""
    return [n for n, _ in model.named_parameters() if ".post." in n or ".offset." in n]
