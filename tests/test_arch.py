"""
Tests for cuas.arch (E18 add-ons). They need torchvision + ultralytics, so they run in the GPU venv and are
skipped elsewhere:

    C:\\cuas_venv\\Scripts\\python -m unittest discover -s tests -p test_arch.py -v
"""
import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    import torchvision  # noqa: F401
    import ultralytics  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False

SPECS = ("eca", "mpda", "rmpda", "aelan", "mpda+rmpda", "aelan+mpda+rmpda")


def flat(o):
    if isinstance(o, torch.Tensor):
        return [o]
    if isinstance(o, dict):
        o = list(o.values())
    return [t for v in o for t in flat(v)]


def max_diff(a, b):
    return max((u - v).abs().max().item() for u, v in zip(flat(a), flat(b)))


@unittest.skipUnless(HAVE, "needs torchvision + ultralytics (GPU venv)")
class TestArch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ultralytics.nn.tasks import DetectionModel
        torch.manual_seed(0)
        cls.base = DetectionModel("yolo26n.yaml", nc=4, verbose=False).eval()
        cls.x = torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            cls.y0 = cls.base(cls.x)

    def patched(self, spec):
        from cuas.arch import apply_arch
        return apply_arch(copy.deepcopy(self.base), spec).eval()

    @staticmethod
    def perturb(m, s=0.3):
        from cuas.arch import added_parameters
        names = set(added_parameters(m))
        with torch.no_grad():
            for n, p in m.named_parameters():
                if n in names:
                    p.add_(torch.randn_like(p) * s)

    def taps(self, m, layers=(4, 6, 16, 19)):
        out = {}
        hs = [m.model[i].register_forward_hook(lambda mod, a, o, i=i: out.__setitem__(i, o)) for i in layers]
        with torch.no_grad():
            m(self.x)
        for h in hs:
            h.remove()
        return out

    def test_identity_at_init(self):
        """Fine-tuning starts from exactly the pretrained function: no add-on changes the output at step 0."""
        for spec in SPECS:
            with self.subTest(spec=spec):
                with torch.no_grad():
                    self.assertLess(max_diff(self.patched(spec)(self.x), self.y0), 1e-4)

    def test_addons_are_active(self):
        """Once their parameters move, the patched blocks' outputs change (the add-ons are in the forward path)."""
        from cuas.arch import LAYERS, parse_spec
        t0 = self.taps(self.base)
        for spec in SPECS:
            m = self.patched(spec)
            self.perturb(m)
            t1 = self.taps(m)
            for tok in parse_spec(spec):
                for i in LAYERS[tok]:
                    with self.subTest(spec=spec, layer=i):
                        rel = ((t1[i] - t0[i]).norm() / t0[i].norm()).item()
                        self.assertGreater(rel, 1e-3)

    def test_original_names_kept_and_pretrained_loads(self):
        """Every original parameter keeps its name and shape, so a plain checkpoint transfers completely."""
        from cuas.arch import build_model
        sd0 = self.base.state_dict()
        for spec in SPECS:
            with self.subTest(spec=spec):
                cfg = dict(copy.deepcopy(self.base.yaml), cuas_arch=spec)
                m = build_model(cfg, nc=4, weights=self.base)
                sd = m.state_dict()
                for k, v in sd0.items():
                    self.assertIn(k, sd)
                    self.assertTrue(torch.equal(sd[k], v), k)
                with torch.no_grad():
                    self.assertLess(max_diff(m.eval()(self.x), self.y0), 1e-4)

    def test_checkpoint_roundtrip_and_resume(self):
        """A trained E18 model survives torch.save/load (as Ultralytics checkpoints do) and rebuilds from its own
        yaml with every parameter, add-ons included, restored (the resume path)."""
        from cuas.arch import build_model
        for spec in ("mpda+rmpda", "aelan+mpda+rmpda"):
            with self.subTest(spec=spec):
                m = self.patched(spec)
                self.perturb(m)
                with torch.no_grad():
                    y = m(self.x)
                buf = io.BytesIO()
                torch.save({"model": m}, buf)
                buf.seek(0)
                loaded = torch.load(buf, weights_only=False)["model"].eval()
                self.assertEqual(loaded.yaml["cuas_arch"], spec)
                with torch.no_grad():
                    self.assertLess(max_diff(loaded(self.x), y), 1e-5)
                rebuilt = build_model(copy.deepcopy(loaded.yaml), nc=4, weights=loaded).eval()
                with torch.no_grad():
                    self.assertLess(max_diff(rebuilt(self.x), y), 1e-5)

    def test_fuse_keeps_deformable_offsets(self):
        """DetectionModel.fuse() (used for inference) must not fold DeformConv into a plain conv."""
        from cuas.arch import DeformConv
        m = self.patched("aelan+mpda+rmpda")
        self.perturb(m)
        n_dc = sum(isinstance(k, DeformConv) for k in m.modules())
        with torch.no_grad():
            y = m(self.x)
            m.fuse(verbose=False)
            self.assertEqual(sum(isinstance(k, DeformConv) for k in m.modules()), n_dc)
            self.assertLess(max_diff(m(self.x), y), 1e-3)

    def test_gradients_reach_addons(self):
        """In training mode the add-on parameters that gate everything else get gradients at step 0."""
        m = self.patched("aelan+mpda+rmpda").train()
        out = m(self.x)
        sum(t.float().pow(2).mean() for t in flat(out)).backward()
        grads = {n: p.grad for n, p in m.named_parameters()}
        # zero-initialised gates: residual MPDA scale (fuse BN weight) and deformable offsets
        for key in ("model.4.post.fuse.bn.weight", "model.19.post.fuse.bn.weight"):
            self.assertIsNotNone(grads[key])
            self.assertGreater(grads[key].abs().sum().item(), 0, key)
        off = [g for n, g in grads.items() if ".offset.weight" in n]
        self.assertTrue(off and all(g is not None for g in off))
        self.assertGreater(sum(g.abs().sum().item() for g in off), 0)
        e = self.patched("eca").train()
        sum(t.float().pow(2).mean() for t in flat(e(self.x))).backward()
        self.assertGreater(e.model[4].post.scale.grad.abs().item(), 0)

    def test_forward_split_matches(self):
        """The exporter swaps C2f.forward for forward_split; the add-on must stay in that path too."""
        m = self.patched("mpda")
        self.perturb(m)
        blk = m.model[4]
        t = torch.rand(1, blk.cv1.conv.in_channels, 32, 32)
        with torch.no_grad():
            self.assertLess((blk.forward(t) - blk.forward_split(t)).abs().max().item(), 1e-5)

    def test_spec_parsing(self):
        from cuas.arch import apply_arch, parse_spec
        self.assertEqual(parse_spec("rmpda+aelan+mpda"), ("aelan", "mpda", "rmpda"))
        self.assertEqual(parse_spec(""), ())
        with self.assertRaises(ValueError):
            parse_spec("mpda+cbam")
        with self.assertRaises(ValueError):
            parse_spec("eca+mpda")
        m = self.patched("mpda")
        self.assertIs(apply_arch(m, "mpda"), m)              # idempotent
        with self.assertRaises(ValueError):
            apply_arch(m, "aelan")                            # different add-ons on an already-patched model

    def test_trainer_get_model(self):
        """ArchTrainer.get_model rebuilds the add-ons from the yaml (fresh start from a plain checkpoint)."""
        from cuas.arch import ArchTrainer, C3k2Post, DeformConv
        fake = SimpleNamespace(data={"nc": 4, "channels": 3}, set_model_names_for_load=lambda m: m)
        cfg = dict(copy.deepcopy(self.base.yaml), cuas_arch="aelan+mpda")
        m = ArchTrainer.get_model(fake, cfg=cfg, weights=self.base, verbose=False)
        self.assertTrue(any(isinstance(k, C3k2Post) for k in m.modules()))
        self.assertTrue(any(isinstance(k, DeformConv) for k in m.modules()))
        plain = ArchTrainer.get_model(fake, cfg=copy.deepcopy(self.base.yaml), weights=self.base, verbose=False)
        self.assertFalse(any(isinstance(k, (C3k2Post, DeformConv)) for k in plain.modules()))

    def test_onnx_export(self):
        """Deployment check: every add-on exports to ONNX and ONNX Runtime (CPU, as on a Pi) matches PyTorch."""
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime not installed")
        import numpy as np

        from cuas.arch import register_onnx_deform
        register_onnx_deform()

        class Head(torch.nn.Module):                          # first output only: a fixed-shape tensor
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return flat(self.m(x))[0]

        for spec in ("mpda+rmpda", "aelan"):
            with self.subTest(spec=spec):
                m = self.patched(spec)
                self.perturb(m, 0.05)
                m.fuse(verbose=False)
                net = Head(m).eval()
                with torch.no_grad():
                    ref = net(self.x).numpy()
                with tempfile.TemporaryDirectory() as d:
                    f = str(Path(d) / "m.onnx")
                    torch.onnx.export(net, (self.x,), f, opset_version=19, dynamo=False)
                    sess = ort.InferenceSession(f, providers=["CPUExecutionProvider"])
                    got = sess.run(None, {sess.get_inputs()[0].name: self.x.numpy()})[0]
                self.assertEqual(got.shape, ref.shape)
                self.assertLess(float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-9)), 1e-3)


if __name__ == "__main__":
    unittest.main()
