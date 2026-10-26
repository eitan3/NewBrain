"""Task 2a: where one full-size training step spends its GPU time (torch.profiler): top kernels, op groups (forward and
backward together), and component regions (record_function, forward; checkpoint recomputes are inside 'backward').

    python prof_ops.py --batch 8 --accum 4 --K 6 --out prof_8x4_K6.json
"""
import argparse
import json
import re
import time
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

import sp_common as sc
from sp_common import vc
import step_lib

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--K", type=int, default=6)
ap.add_argument("--variant", default="")
ap.add_argument("--no-regions", action="store_true")
ap.add_argument("--out", default="prof.json")
cli = ap.parse_args()

sc.guard()
torch.manual_seed(0)
S = vc.load(full=True)
if cli.variant:
    import variants
    variants.apply(S, cli.variant.split(","))
if not cli.no_regions:
    step_lib.install_regions()
kw = dict(batch=cli.batch, accum=cli.accum)
for _ in range(2):
    step_lib.train_step(S, cli.K, **kw)
torch.cuda.synchronize()
t0 = time.time()
step_lib.train_step(S, cli.K, **kw)
wall_noprof = time.time() - t0
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    t0 = time.time()
    step_lib.train_step(S, cli.K, regions=not cli.no_regions, **kw)
    wall = time.time() - t0
ka = prof.key_averages()

GROUPS = [
    ("gemm", r"^aten::(mm|bmm|addmm|matmul|baddbmm|linear|_grouped_mm|einsum)$"),
    ("conv", r"^aten::(cudnn_convolution|convolution|_convolution|conv1d|convolution_backward)"),
    ("topk_sort", r"^aten::(topk|sort|argsort|_sort|kthvalue|unique|_unique2)"),
    ("index_scatter_gather", r"^aten::(index|index_put_|_index_put_impl_|index_add_|index_select|gather|scatter|"
                             r"scatter_|scatter_add_|index_copy_|index_fill_|embedding_bag|_embedding_bag|"
                             r"embedding_dense_backward|_embedding_bag_backward|_embedding_bag_dense_backward|"
                             r"embedding|masked_fill_|masked_fill|nonzero|nonzero_static|bincount|"
                             r"_embedding_bag_per_sample_weights_backward)"),
    ("softmax", r"^aten::(_softmax|_log_softmax|_softmax_backward_data|_log_softmax_backward_data|logsumexp)"),
    ("reduction", r"^aten::(sum|mean|norm|linalg_vector_norm|amax|max|min|any|all|cumsum|_cumsum|cummin|prod|std|var)"),
    ("copy_cast", r"^aten::(copy_|_to_copy|to|clone|contiguous|cat|stack|fill_|zero_|zeros|ones|full|new_zeros|"
                  r"_foreach_copy_)"),
    ("elementwise", r"^aten::"),
]


def group_of(name):
    for g, rx in GROUPS:
        if re.match(rx, name):
            return g
    return "other"


ops = [e for e in ka if e.key.startswith("aten::")]
gsum = defaultdict(float)
for e in ops:
    gsum[group_of(e.key)] += e.self_device_time_total / 1e6
kern = [e for e in ka if not e.key.startswith(("aten::", "R:", "forward", "backward", "clip", "opt_step",
                                               "apply_balance", "mem_after_step", "cuda", "Optimizer",
                                               "autograd::", "Activity", "ProfilerStep", "torch::", "<"))
        and e.self_device_time_total > 0]
gpu_total = sum(e.self_device_time_total for e in kern) / 1e6
n_kernels = sum(e.count for e in kern)
top_k = sorted(kern, key=lambda e: -e.self_device_time_total)[:30]
top_ops = sorted(ops, key=lambda e: -e.self_device_time_total)[:30]
regions = {e.key: dict(device_s=e.device_time_total / 1e6, cpu_s=e.cpu_time_total / 1e6, count=e.count)
           for e in ka if e.key.startswith("R:") or e.key in ("forward", "backward", "clip", "opt_step",
                                                               "apply_balance", "mem_after_step")}
syncs = {e.key: dict(count=e.count, cpu_ms=e.cpu_time_total / 1e3) for e in ka
         if e.key in ("aten::item", "aten::_local_scalar_dense", "cudaStreamSynchronize", "cudaDeviceSynchronize",
                      "cudaMemcpyAsync", "aten::nonzero", "aten::tolist")}
# GPU idle gaps on the kernel timeline, and the part of them that starts at a host sync
from torch.autograd import DeviceType  # noqa: E402
evs = prof.events()
kev = sorted([(e.time_range.start, e.time_range.end) for e in evs if e.device_type == DeviceType.CUDA
              and e.time_range.end > e.time_range.start], key=lambda t: t[0])
sync_end = sorted(e.time_range.end for e in evs if e.name in ("cudaStreamSynchronize", "cudaDeviceSynchronize"))
gaps, after_sync, big = 0.0, 0.0, 0.0
end = kev[0][1] if kev else 0
import bisect  # noqa: E402
for s, e in kev[1:]:
    if s > end:
        g = s - end
        gaps += g
        if g > 100:
            big += g
        i = bisect.bisect_left(sync_end, end)
        if i < len(sync_end) and sync_end[i] <= s:
            after_sync += g
    end = max(end, e)
span = (kev[-1][1] - kev[0][0]) / 1e6 if kev else 0
gap_info = dict(kernel_span_s=span, idle_s=gaps / 1e6, idle_after_sync_s=after_sync / 1e6, idle_gaps_over_100us_s=big / 1e6,
                n_syncs=len(sync_end))
print("gaps:", gap_info)
res = dict(gaps=gap_info, batch=cli.batch, accum=cli.accum, K=cli.K, variant=cli.variant, wall_s=wall, wall_noprof_s=wall_noprof,
           gpu_kernel_s=gpu_total, gpu_busy_frac=gpu_total / wall, n_kernels=n_kernels,
           tok_s_this_K=cli.batch * cli.accum * 512 / wall_noprof,
           op_groups={k: round(v, 4) for k, v in sorted(gsum.items(), key=lambda kv: -kv[1])},
           top_kernels=[dict(name=e.key[:140], gpu_ms=round(e.self_device_time_total / 1e3, 2), count=e.count)
                        for e in top_k],
           top_ops=[dict(name=e.key, gpu_ms=round(e.self_device_time_total / 1e3, 2), count=e.count)
                    for e in top_ops],
           regions=dict(sorted(regions.items(), key=lambda kv: -kv[1]["device_s"])), syncs=syncs)
(sc.HERE / cli.out).write_text(json.dumps(res, indent=1))
print(f"wall {wall:.3f} s (no profiler {wall_noprof:.3f} s), GPU kernels {gpu_total:.3f} s "
      f"({100 * gpu_total / wall:.0f}%), {n_kernels} kernels")
print("op groups (s):", res["op_groups"])
print("regions (device s, inclusive):")
for k, v in res["regions"].items():
    print(f"  {k:24s} {v['device_s']:8.3f}  x{v['count']}")
print("top kernels:")
for e in res["top_kernels"][:20]:
    print(f"  {e['gpu_ms']:8.1f} ms x{e['count']:5d}  {e['name'][:110]}")
print("syncs:", syncs)
