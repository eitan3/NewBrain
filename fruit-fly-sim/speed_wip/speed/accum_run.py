"""Task 1: N optimizer steps of the full-size model at batch x accum from the same seed and the same data.

    python accum_run.py --batch 16 --accum 2 --steps 40 --tag A
    python accum_run.py --batch 8 --accum 4 --fix all --tag D

Logs per step: K, loss, pre-clip grad norm, capacity drop fraction, per-expert routed counts (the step's pending
counts), touched memory rows. Saves (scratchpad) the step-1 gradients and the final parameters / key buffers for
compare_runs.py. chunk_ce (same math, from ../vram/variants.py) is on by default so 16x2 fits under the 0.78 cap.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import sp_common as sc
from sp_common import vc

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--fix", default="")
ap.add_argument("--tag", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cuda-seed", type=int, default=0)
ap.add_argument("--chunk-ce", type=int, default=1)
ap.add_argument("--dropless", type=int, default=0)
cli = ap.parse_args()

sc.guard()
torch.manual_seed(cli.seed)
S = vc.load(full=True)
import variants  # noqa: E402  (../vram)
if cli.chunk_ce:
    variants.apply(S, ["chunk_ce"])
import accum_fixes as fx  # noqa: E402
fx.install(S, [n for n in cli.fix.split(",") if n])
mlp = S.model.blocks[0].mlp
if cli.dropless:
    mlp.cap = 0
# the same random streams for every run (after the load, whose growth consumed some)
torch.manual_seed(1000 + cli.seed)
torch.cuda.manual_seed_all(1000 + cli.cuda_seed)
np.random.seed(1234)
kg = torch.Generator().manual_seed(123)


def sample_K():
    if torch.rand((), generator=kg).item() < 0.05:
        return int(torch.randint(12, 17, (), generator=kg).item())
    return int(torch.randint(2, 11, (), generator=kg).item())


Ks = [sample_K() for _ in range(cli.steps)]

# record capacity drops per routed forward
_drops = []
_fwd = type(mlp).forward


def rec_fwd(self, *a, **kw):
    out = _fwd(self, *a, **kw)
    if self.training:
        _drops.append(self.stats[1].float() / self.stats[2])
    return out


type(mlp).forward = rec_fwd


def step_fn(K, step, first):
    m, opt, a, data, mem = S.model, S.opt, S.args, S.data, S.mem
    B, A = cli.batch, cli.accum
    lr = vc.set_step_state(S, step)
    m.train()
    m.K = K
    m.aux_k_fixed = m.sample_aux_k(K)
    m.group_offset = m.sample_group_offset()
    opt.zero_grad(set_to_none=True)
    xs = [data.batch(B) for _ in range(A)]
    fx.begin_step(S, step, xs, A)
    micro = []
    _drops.clear()
    for i, (x, y) in enumerate(xs):
        fx.begin_micro(S, i)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, loss = m(x, y)
            total = loss
            aux = m.aux_loss()
            if aux is not None:
                total = total + aux
            if m.concept_loss is not None:
                total = total + a.concept_w * m.concept_loss
            if m.aux_x is not None:
                total = total + 0.2 * F.cross_entropy(m.heads["lm"](m.aux_x).float().flatten(0, 1), y.flatten())
        (total / A).backward()
        micro.append(loss.detach())
        del logits, total
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    grads = None
    if first:
        grads = {n: p.grad.detach().float().cpu().clone() for n, p in m.named_parameters() if p.grad is not None}
        grads["mem.G"] = mem.G.float().cpu().clone()
    opt.step(lr / a.lr)
    fx.end_step(S)
    counts = mlp.pending.clone()
    mlp.apply_balance()
    lt = sum(micro) / len(micro)
    cnt = mem.write_counts()
    drop = torch.stack(_drops).mean()
    tok = torch.cat([x for x, _ in xs]).flatten()
    chk = (tok.double() * torch.arange(tok.numel(), device=tok.device).double().remainder(9973)).sum()
    vals = torch.cat([lt.double().view(1), gn.double().view(1), drop.double().view(1), chk.view(1),
                      cnt.double()]).tolist()
    li, gnv, dr, ck, c = vals[0], vals[1], vals[2], vals[3], [int(v) for v in vals[4:]]
    mem.after_step(lr, c)
    return dict(K=K, loss=li, grad_norm=gnv, drop=dr, tok_chk=ck, touched=c), counts.cpu(), grads


out = dict(batch=cli.batch, accum=cli.accum, fix=cli.fix, seed=cli.seed, cuda_seed=cli.cuda_seed,
           chunk_ce=cli.chunk_ce, dropless=cli.dropless, steps=[])
loads = []
t0 = time.time()
for i, K in enumerate(Ks):
    rec, counts, grads = step_fn(K, S.step0 + i, i == 0)
    if grads is not None:
        torch.save(grads, sc.SCRATCH / f"t1_grads_{cli.tag}.pt")
        del grads
    loads.append(counts)
    out["steps"].append(rec)
    print(f"step {i:2d} K={K:2d} loss {rec['loss']:.4f} gn {rec['grad_norm']:.3f} drop {rec['drop']:.4f}", flush=True)
torch.cuda.synchronize()
out["seconds"] = time.time() - t0
out["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
out["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
m = S.model
final = {n: p.detach().float().cpu().clone() for n, p in m.named_parameters()}
for s_name, s in [(n, mod) for n, mod in m.named_modules() if hasattr(mod, "ema_rate")]:
    final[f"{s_name}.ema_rate"] = s.ema_rate.float().cpu().clone()
final["mlp.bias"] = mlp.bias.float().cpu().clone()
final["mlp.load_ema"] = mlp.load_ema.float().cpu().clone()
final["mem.norm_m"] = S.mem.norm_m.cpu().clone()
final["mem.norm_v"] = S.mem.norm_v.cpu().clone()
final["mem.V_sample"] = S.mem.V[::97].float().cpu().clone()
final["loads"] = torch.stack(loads)
torch.save(final, sc.SCRATCH / f"t1_final_{cli.tag}.pt")
(sc.HERE / f"t1_{cli.tag}.json").write_text(json.dumps(out, indent=1))
print(f"done {out['seconds']:.1f} s, peak alloc {out['peak_alloc_gb']:.2f} reserved {out['peak_reserved_gb']:.2f}",
      flush=True)
