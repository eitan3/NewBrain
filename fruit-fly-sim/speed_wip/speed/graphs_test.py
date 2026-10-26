"""Task 2c: CUDA graphs without Triton, on the full-size training step (K fixed, so shapes are fixed).

    python graphs_test.py --mode eager|mixer_cg|core_cg|norm_cg|explain_mixer|explain_core [--nohomeo]

  mixer_cg  torch.compile(SSDMixer.forward, backend="cudagraphs")   (Dynamo + AOTAutograd + CUDA-graph trees)
  core_cg   torch.compile(Core.forward, backend="cudagraphs")       (graph breaks at the routing .tolist())
  norm_cg   torch.compile(RMSNorm.forward, backend="cudagraphs")
  explain_* torch._dynamo.explain: graph count / break reasons (no timing)
Timing: median of 5 steps after 3 warm-up steps; the loss of the first 3 steps from the same seed vs eager.
"""
import argparse
import json
import time

import numpy as np
import torch

import sp_common as sc
from sp_common import vc
import step_lib

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="eager")
ap.add_argument("--K", type=int, default=6)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--nohomeo", action="store_true")
ap.add_argument("--variant", default="")
cli = ap.parse_args()

sc.guard()
if "cg" in cli.mode:  # (Dynamo reads the TF32 flags through the new API; the legacy setter breaks it: use the new one)
    import contextlib
    import memory
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"

    @contextlib.contextmanager
    def _fp32(device_type):
        old = torch.backends.cuda.matmul.fp32_precision, torch.backends.cudnn.conv.fp32_precision
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        try:
            with torch.autocast(device_type, enabled=False):
                yield
        finally:
            torch.backends.cuda.matmul.fp32_precision, torch.backends.cudnn.conv.fp32_precision = old
    memory.fp32_math = _fp32
torch.manual_seed(0)
S = vc.load(full=True)
np.random.seed(1234)
if cli.nohomeo:
    S.model.set_homeo(0.0, S.args.target, S.args.homeo_mode)
from loop_brain import Core  # noqa: E402
from spiking_lm import RMSNorm, SSDMixer  # noqa: E402

import torch._dynamo  # noqa: E402
torch._dynamo.config.cache_size_limit = 64
res = dict(mode=cli.mode, K=cli.K, batch=cli.batch, accum=cli.accum, nohomeo=cli.nohomeo)
if cli.mode.startswith("explain"):
    core = S.model.blocks[0]
    x = torch.randn(cli.batch, 512, 384, device="cuda")
    S.model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if cli.mode == "explain_mixer":
            ex = torch._dynamo.explain(core.mix.forward)(x, group=1)
        else:
            s = core.init_state(x)
            ex = torch._dynamo.explain(core.forward)(x, s, x, group=1)
    res.update(graphs=ex.graph_count, breaks=ex.graph_break_count,
               reasons=[str(b.reason)[:300] for b in ex.break_reasons][:20], ops=ex.op_count)
    print(res)
else:
    import speed_variants
    speed_variants.apply(S, cli.variant.split(","))
    step_lib.train_step(S, cli.K, step=6990, batch=cli.batch, accum=cli.accum)   # eager warm-up: host caches filled
    torch.manual_seed(0)
    np.random.seed(1234)
    def cg(f):
        c = torch.compile(f, backend="cudagraphs")

        def g(*a, **k):
            torch.compiler.cudagraph_mark_step_begin()
            return c(*a, **k)
        return g
    if "contig" in cli.mode:
        from spiking_lm import CausalConv
        _cc = CausalConv.forward
        CausalConv.forward = lambda self, x, ch=None: _cc(self, x, ch).contiguous()
    if cli.mode.startswith("mixer_cg"):
        SSDMixer.forward = cg(SSDMixer.forward) if "mark" in cli.mode else torch.compile(SSDMixer.forward, backend="cudagraphs")
    elif cli.mode == "scan_cg_mark":
        import spiking_lm
        spiking_lm.ssd_scan = cg(spiking_lm.ssd_scan)
    elif cli.mode == "core_cg_mark":
        Core.forward = cg(Core.forward)
    elif cli.mode == "core_cg":
        Core.forward = torch.compile(Core.forward, backend="cudagraphs")
    elif cli.mode == "norm_cg":
        RMSNorm.forward = torch.compile(RMSNorm.forward, backend="cudagraphs")
    losses, times = [], []
    try:
        for i in range(8):
            torch.cuda.synchronize()
            t0 = time.time()
            li = step_lib.train_step(S, cli.K, step=7000 + i, batch=cli.batch, accum=cli.accum)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
            losses.append(li)
            print(f"step {i} loss {li:.5f} {times[-1]:.3f} s", flush=True)
        res.update(losses=losses, times=times, median_s=float(np.median(times[3:])),
                   tok_s=cli.batch * cli.accum * 512 / float(np.median(times[3:])),
                   peak_alloc_gb=torch.cuda.max_memory_allocated() / 1e9)
    except Exception as ex:  # noqa: BLE001
        res.update(error=repr(ex)[:1500], losses=losses, times=times)
        print("ERROR", repr(ex)[:1500], flush=True)
(sc.HERE / f"graphs_{cli.mode}{'_nohomeo' if cli.nohomeo else ''}{'_' + cli.variant.replace(',', '+') if cli.variant else ''}.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k not in ("losses", "times")}, indent=1))
