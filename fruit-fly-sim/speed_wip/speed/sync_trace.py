"""Task 2b: every host-device sync of one training step, with the Python stack that caused it
(torch.cuda.set_sync_debug_mode('warn')), grouped by the innermost model-code frame; plus the time each costs
(host wall time blocked in the sync, measured with a wrapper around the step).

    python sync_trace.py --batch 16 --accum 2 --K 6
"""
import argparse
import json
import time
import traceback
import warnings
from collections import Counter, defaultdict

import torch

import sp_common as sc
from sp_common import vc
import step_lib

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--K", type=int, default=6)
ap.add_argument("--variant", default="")
ap.add_argument("--out", default="")
cli = ap.parse_args()

sc.guard()
torch.manual_seed(0)
S = vc.load(full=True)
if cli.variant:
    import variants
    import speed_variants
    for v in cli.variant.split(","):
        (variants.apply if v in ("chunk_ce", "ckpt_pass", "bf16_mom", "nohomeo") else speed_variants.apply)(S, [v])
kw = dict(batch=cli.batch, accum=cli.accum)
step_lib.train_step(S, cli.K, **kw)
step_lib.train_step(S, cli.K, **kw)
torch.cuda.synchronize()

recs = []
CODE = ("fruit-fly-sim",)


def show(message, category, filename, lineno, file=None, line=None):
    if "synchroniz" not in str(message):
        return
    st = traceback.extract_stack()[:-1]
    frames = [f for f in st if any(c in f.filename for c in CODE) and "warnings" not in f.filename]
    model = [f for f in frames if "speed_wip" not in f.filename]
    inner = model[-1] if model else (frames[-1] if frames else st[-1])
    chain = " <- ".join(f"{f.filename.split(chr(92))[-1]}:{f.lineno}({f.name})" for f in reversed(model[-4:]))
    recs.append(dict(site=f"{inner.filename.split(chr(92))[-1]}:{inner.lineno} {inner.name}",
                     code=(inner.line or "").strip()[:120], chain=chain, msg=str(message)[:80]))


old = warnings.showwarning
warnings.showwarning = show
warnings.simplefilter("always")
torch.cuda.set_sync_debug_mode("warn")
t0 = time.time()
step_lib.train_step(S, cli.K, sync=False, **kw)
torch.cuda.set_sync_debug_mode("default")
torch.cuda.synchronize()
wall = time.time() - t0
warnings.showwarning = old

by = Counter(r["site"] for r in recs)
code = {r["site"]: r["code"] for r in recs}
chains = defaultdict(set)
for r in recs:
    chains[r["site"]].add(r["chain"])
print(f"{len(recs)} syncs in one step (batch {cli.batch} x accum {cli.accum}, K {cli.K}); wall {wall:.2f} s")
for s, n in by.most_common():
    print(f"{n:4d}  {s:55s} | {code[s]}")
    for c in list(chains[s])[:2]:
        print(f"        {c}")
out = dict(batch=cli.batch, accum=cli.accum, K=cli.K, n=len(recs),
           sites=[dict(site=s, count=n, code=code[s], chains=sorted(chains[s])[:3]) for s, n in by.most_common()])
(sc.HERE / (cli.out or f"syncs_{cli.batch}x{cli.accum}_K{cli.K}.json")).write_text(json.dumps(out, indent=1))
