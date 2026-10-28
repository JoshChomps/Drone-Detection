"""
E18 cost of the YOLOBirDrone-style add-ons (cuas/arch.py) on YOLO26s at one 640-px tile, before any training:
parameters, counted FLOPs, GPU latency (PyTorch fp16, batch 1, fused) and CPU latency through ONNX Runtime with 4
threads (a stand-in for the Pi 5's 4 cores: RELATIVE cost only, the Pi itself is slower). Random weights: latency
does not depend on the weights. Writes C:/cuas_data/runs/e18_bench.json.

    C:\\cuas_venv\\Scripts\\python experiments/e18_arch_bench.py
"""
import copy
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuas.arch import apply_arch, register_onnx_deform  # noqa: E402

D = Path(os.environ.get("CUAS_DATA", "C:/cuas_data"))
VARIANTS = ("", "eca", "mpda", "rmpda", "mpda+rmpda", "aelan", "aelan+mpda+rmpda")
IMG = 640
ROUNDS = 3


def first(o):
    return o if isinstance(o, torch.Tensor) else first(o[0] if isinstance(o, (list, tuple)) else list(o.values())[0])


class Head(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return first(self.m(x))


def gflops(model):
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(torch.zeros(1, 3, IMG, IMG))
    return fc.get_total_flops() / 1e9


def gpu_ms(model, n=200):
    if not torch.cuda.is_available():
        return None
    m = copy.deepcopy(model).cuda().half()
    x = torch.rand(1, 3, IMG, IMG, device="cuda", dtype=torch.half)
    with torch.no_grad():
        for _ in range(30):
            m(x)
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t = time.perf_counter()
            m(x)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
    return 1000 * statistics.median(ts)


def ort_ms(model, n=30, threads=4):
    import onnxruntime as ort
    x = torch.rand(1, 3, IMG, IMG)
    with tempfile.TemporaryDirectory() as d:
        f = str(Path(d) / "m.onnx")
        torch.onnx.export(Head(model), (x,), f, opset_version=19, dynamo=False)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        s = ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])
        name, xa = s.get_inputs()[0].name, x.numpy()
        for _ in range(5):
            s.run(None, {name: xa})
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            s.run(None, {name: xa})
            ts.append(time.perf_counter() - t)
    return 1000 * statistics.median(ts)


def main():
    import warnings
    from ultralytics.nn.tasks import DetectionModel
    warnings.filterwarnings("ignore")
    register_onnx_deform()
    torch.manual_seed(0)
    base = DetectionModel("yolo26s.yaml", nc=4, verbose=False).eval()
    models, rows = {}, []
    for spec in VARIANTS:
        m = apply_arch(copy.deepcopy(base), spec).eval() if spec else copy.deepcopy(base)
        m.fuse(verbose=False)
        models[spec] = m
        rows.append({"variant": spec or "base (YOLO26s)", "params_M": sum(p.numel() for p in m.parameters()) / 1e6,
                     "gflops_counted": gflops(m), "gpu_fp16_ms": None, "ort_cpu4_ms": None})
    # The laptop GPU/CPU are shared (a local LLM, browsers): interleave the variants over ROUNDS rounds and keep
    # each one's best round, so a burst of outside load does not land on one variant only.
    for k in range(ROUNDS):
        for spec, r in zip(VARIANTS, rows):
            g, c = gpu_ms(models[spec]), ort_ms(models[spec])
            r["gpu_fp16_ms"] = g if r["gpu_fp16_ms"] is None or (g and g < r["gpu_fp16_ms"]) else r["gpu_fp16_ms"]
            r["ort_cpu4_ms"] = c if r["ort_cpu4_ms"] is None or c < r["ort_cpu4_ms"] else r["ort_cpu4_ms"]
        print(f"round {k + 1}/{ROUNDS} done", flush=True)
    for r in rows:
        print(f"{r['variant']:18s} {r['params_M']:6.2f} M  {r['gflops_counted']:6.1f} GFLOP  "
              f"GPU {r['gpu_fp16_ms'] or float('nan'):5.2f} ms  ORT-CPU(4 thr) {r['ort_cpu4_ms']:6.1f} ms", flush=True)
    b = rows[0]
    print("\nrelative to base: " + ", ".join(
        f"{r['variant']} GPU x{r['gpu_fp16_ms'] / b['gpu_fp16_ms']:.2f} CPU x{r['ort_cpu4_ms'] / b['ort_cpu4_ms']:.2f}"
        for r in rows[1:] if r["gpu_fp16_ms"]))
    print("note: counted FLOPs miss deform_conv2d itself (its offset convs are counted); latency includes it.")
    meta = {"rounds": ROUNDS, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "img": IMG, "rows": rows}
    out = D / "runs" / "e18_bench.json"
    out.write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
