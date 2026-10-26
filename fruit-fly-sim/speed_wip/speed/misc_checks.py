"""Task 2 misc checks: versions, jiterator availability / speed on the spike chain, allocator syncs per step, and
set_sync_debug_mode('error') on the sync-free variant (only the per-step write_counts read should remain)."""
import json
import sys
import time

import torch

import sp_common as sc
from sp_common import vc
import step_lib
import speed_variants

sc.guard()
out = dict(torch=torch.__version__, cuda=torch.version.cuda, python=sys.version.split()[0],
           gpu=torch.cuda.get_device_name(), cudnn=torch.backends.cudnn.version())
try:
    import importlib.util
    out["triton_importable"] = importlib.util.find_spec("triton") is not None
except Exception as ex:  # noqa: BLE001
    out["triton_importable"] = repr(ex)
# jiterator: the graded-spike forward chain s = round(clamp(u / a, lo, hi)) * a as one runtime-compiled kernel
try:
    code = "template <typename T> T spike(T u, T a) { T v = u / a; v = v < T(-4) ? T(-4) : (v > T(4) ? T(4) : v); return nearbyint(v) * a; }"
    jf = torch.cuda.jiterator._create_jit_fn(code)
    u = torch.randn(4096, 704, device="cuda")
    a = torch.rand(704, device="cuda") + 0.3
    y = jf(u, a)
    ref = torch.clamp(u / a, -4, 4).round() * a
    def bench(f, n=200):
        for _ in range(10):
            f()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(n):
            f()
        torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
    out["jiterator"] = dict(ok=True, max_diff=float((y - ref).abs().max()), jit_ms=bench(lambda: jf(u, a)),
                            eager_chain_ms=bench(lambda: torch.clamp(u / a, -4, 4).round() * a))
except Exception as ex:  # noqa: BLE001
    out["jiterator"] = dict(ok=False, error=repr(ex)[:400])
print(out, flush=True)

torch.manual_seed(0)
S = vc.load(full=True)
speed_variants.apply(S, ["topk_nosync2", "route_static"])
kw = dict(batch=8, accum=4)
step_lib.train_step(S, 6, **kw)
step_lib.train_step(S, 6, **kw)
st0 = torch.cuda.memory_stats()
torch.cuda.synchronize()
# 'error' mode: the first sync raises; the write_counts read is outside the forward/backward
import traceback  # noqa: E402
torch.cuda.set_sync_debug_mode("error")
try:
    step_lib.train_step(S, 6, sync=False, **kw)
    out["sync_error_mode"] = "no sync in the whole step"
except RuntimeError as ex:
    tb = traceback.extract_tb(ex.__traceback__)
    out["sync_error_mode"] = f"first sync at {tb[-1].filename.split(chr(92))[-1]}:{tb[-1].lineno} {tb[-1].line}"
torch.cuda.set_sync_debug_mode("default")
torch.cuda.synchronize()
for i in range(3):
    step_lib.train_step(S, 6, **kw)
st1 = torch.cuda.memory_stats()
out["allocator_per_step"] = {k: (st1.get(k, 0) - st0.get(k, 0)) / 4 for k in
                             ("num_alloc_retries", "num_sync_all_streams", "num_device_alloc", "num_device_free")}
print(out, flush=True)
(sc.HERE / "misc_checks.json").write_text(json.dumps(out, indent=1))
