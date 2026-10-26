"""Profile the evaluation side of a training run at full size: evaluate() (eval_batch 32, K = 6), the loop report
(loop_report: K 1..16 sweeps, halt_eval, think_diag, knock-outs, tick losses), the drift probes (memory_metrics
with MemoryProbes), each with its peak allocated / reserved, and the reserved VRAM of the training steps after it
(fragmentation), exactly in the order train_lm runs them at a report step that is also a probe eval.

    python profile_report.py [--free 1] [--variant chunk_ce] [--probe-batch 8] [--eval-batch 32]
"""
import argparse
import json
import time

import torch

import vram_common as vc
import train_lm

ap = argparse.ArgumentParser()
ap.add_argument("--free", type=int, default=1, help="1 = free_cache around the report / probes as train_lm does")
ap.add_argument("--extra-free", type=int, default=0, help="1 = also empty_cache after evaluate() and before the next step")
ap.add_argument("--variant", default="")
ap.add_argument("--probe-batch", type=int, default=0)
ap.add_argument("--eval-batch", type=int, default=0)
ap.add_argument("--report", type=int, default=1)
ap.add_argument("--out", default="report_profile.json")
cli = ap.parse_args()

vc.gpu_guard()
torch.manual_seed(0)
S = vc.load(full=True)
if cli.variant:
    import variants
    variants.apply(S, cli.variant.split(","))
if cli.probe_batch:
    S.args.mem_probe_batch = cli.probe_batch
    S.probes.batch = cli.probe_batch
if cli.eval_batch:
    S.args.eval_batch = cli.eval_batch
rows = []


def mark(tag, t0=None):
    torch.cuda.synchronize()
    st = vc.mstat(tag)
    if t0 is not None:
        st["seconds"] = time.time() - t0
    rows.append(st)
    print(f"{tag:28s} alloc {st['alloc']:6.2f}  reserved {st['reserved']:6.2f}  peak alloc {st['peak_alloc']:6.2f}  "
          f"peak reserved {st['peak_reserved']:6.2f}" + (f"  {st['seconds']:.1f}s" if t0 else ""), flush=True)
    torch.cuda.reset_peak_memory_stats()


free = (lambda: torch.cuda.empty_cache()) if cli.free else (lambda: None)
for i in range(3):
    vc.train_step(S, [6, 10, 16][i])
mark("train x3 (K 6,10,16)")
a = S.args
t0 = time.time()
val, sp = train_lm.evaluate(S.model, S.data, a, track=True)
mark("evaluate", t0)
if cli.extra_free:
    torch.cuda.empty_cache()
if cli.report:
    free()
    mark("free before report")
    t0 = time.time()
    rep = train_lm.loop_report(S.model, S.data, a)
    mark("loop_report", t0)
    free()
    mark("free after report")
res0 = torch.cuda.memory_reserved()
free()
t0 = time.time()
mrec, alarm = train_lm.memory_metrics(S.model, S.probes)
mark("memory probes", t0)
free()
mark("free after probes")
for K in (6, 10, 16, 6):
    vc.train_step(S, K)
    mark(f"train K={K} after eval")
out = dict(cli=vars(cli), rows=rows, val=val, stats=torch.cuda.memory_stats().get("num_alloc_retries"))
(vc.HERE / cli.out).write_text(json.dumps(out, indent=1, default=str))
print("wrote", vc.HERE / cli.out)
