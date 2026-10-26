"""One training step exactly as train_lm (= vram_common.train_step), with optional profiler regions (record_function)
around the model's components, installed by monkeypatching (the model code is not edited)."""
import functools

import torch
import torch.nn.functional as F
from torch.profiler import record_function

from sp_common import vc


def train_step(S, K, step=7000, batch=None, accum=None, regions=False, sync=True):
    m, opt, a, data, mem = S.model, S.opt, S.args, S.data, S.mem
    B, A = batch or a.batch, accum or a.accum
    R = (lambda n: record_function(n)) if regions else (lambda n: _Null())
    lr = vc.set_step_state(S, step)
    m.train()
    m.K = K
    if A > 1:
        m.aux_k_fixed = m.sample_aux_k(m.K)
    m.group_offset = m.sample_group_offset()
    opt.zero_grad(set_to_none=True)
    micro = []
    for i in range(A):
        x, y = data.batch(B)
        with R("forward"):
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
        with R("backward"):
            (total / A if A > 1 else total).backward()
        micro.append(loss.detach())
        del logits, total
    with R("clip"):
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    with R("opt_step"):
        opt.step(lr / a.lr)
    with R("apply_balance"):
        m.blocks[0].mlp.apply_balance()
    lt = sum(micro) / len(micro)
    with R("mem_after_step"):
        cnt = mem.write_counts()
        li = None
        if cnt is not None:
            vals = torch.cat([lt.double().view(1), cnt.double()]).tolist()
            li, cnt = vals[0], [int(v) for v in vals[1:]]
        mem.after_step(lr, cnt)
    if sync:
        torch.cuda.synchronize()
    return li


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _wrap(obj, name, label):
    f = getattr(obj, name)
    if isinstance(f, staticmethod):
        f = f.__func__

    @functools.wraps(f)
    def g(*a, **kw):
        with record_function(label):
            return f(*a, **kw)
    setattr(obj, name, g)


def install_regions():
    """record_function labels on the components (forward; checkpoint recomputes show up inside 'backward')."""
    import loop_brain
    import memory
    import spiking_lm
    from loop_brain import Core, LoopBrain
    from memory import ConceptHier, Memory
    from spiking_lm import MLP, ExpertSpike, RMSNorm, RoutedMLP, Spike, SSDMixer
    _wrap(RMSNorm, "forward", "R:rmsnorm")
    _wrap(SSDMixer, "forward", "R:ssd_mixer")
    orig_scan = spiking_lm.ssd_scan

    def scan(*a, **kw):
        with record_function("R:ssd_scan"):
            return orig_scan(*a, **kw)
    spiking_lm.ssd_scan = scan
    _wrap(Spike, "forward", "R:spike")
    _wrap(ExpertSpike, "forward", "R:expert_spike")
    _wrap(RoutedMLP, "forward", "R:routed_mlp")
    _wrap(RoutedMLP, "route", "R:router_scores")
    _wrap(MLP, "forward", "R:shared_mlp")
    _wrap(Core, "read", "R:ws_read")
    _wrap(Core, "write", "R:ws_write")
    _wrap(Core, "forward", "R:core_pass")
    _wrap(LoopBrain, "_hier_step", "R:concept_mem_block")
    _wrap(ConceptHier, "forward", "R:concept_heads")
    _wrap(Memory, "select", "R:mem_select")
    _wrap(Memory, "online", "R:mem_online")
    _wrap(Memory, "read_shared", "R:mem_read_shared")
    _wrap(Memory, "private", "R:mem_private")
    _wrap(Memory, "prep", "R:mem_prep")
    _wrap(LoopBrain, "pass_readout_ce", "R:pass_readout")
    _wrap(LoopBrain, "halt_lambda", "R:halt_head")
    orig_ct, orig_cl = loop_brain.concept_targets, loop_brain.concept_loss

    def ct(*a, **kw):
        with record_function("R:concept_targets"):
            return orig_ct(*a, **kw)

    def cl(*a, **kw):
        with record_function("R:concept_loss"):
            return orig_cl(*a, **kw)
    loop_brain.concept_targets, loop_brain.concept_loss = ct, cl
    _wrap(LoopBrain, "features", "R:run_total")
