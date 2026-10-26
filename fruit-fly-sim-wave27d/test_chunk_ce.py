"""--chunk-ce (loop_brain.chunked_readout): same loss / halting features / gradients as the plain read-outs (CPU,
fp32, tiny model). Run: python test_chunk_ce.py"""
import copy
import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from loop_brain import LoopBrain, chunked_readout

torch.set_num_threads(2)


def _close(a, b, rtol=1e-4, atol=1e-6):
    assert torch.allclose(a, b, rtol=rtol, atol=atol), (a - b).abs().max().item()


def test_helper_matches_reference():
    torch.manual_seed(0)
    N, d, V = 37, 16, 50
    head = nn.Linear(d, V, bias=False)
    h = torch.randn(N, d, requires_grad=True)
    t = torch.randint(0, V, (N,))
    prev = torch.randn(N, V)
    g = torch.rand(N)
    lp = F.log_softmax(head(h).float(), -1)
    ce0 = -lp.gather(1, t[:, None]).squeeze(1)
    (ce0 * g).sum().backward()
    dh0, dW0 = h.grad.clone(), head.weight.grad.clone()
    h.grad = head.weight.grad = None
    p = lp.exp().detach()
    H0 = -(p * lp.detach()).sum(-1)
    kl0 = torch.log1p((p * (lp.detach() - F.log_softmax(prev, -1))).sum(-1).clamp(min=0))
    for chunk in (5, 37, 64):
        ce1, lg1, H1, kl1 = chunked_readout(head, h, t, chunk, prev, True, True, True)
        (ce1 * g).sum().backward()
        _close(ce1, ce0, rtol=0)
        _close(H1, H0, rtol=0)
        _close(kl1, kl0, rtol=0)
        _close(lg1, head(h).detach(), rtol=0)
        _close(h.grad, dh0)
        _close(head.weight.grad, dW0)
        h.grad = head.weight.grad = None


def _loop(**kw):
    cfg = dict(vocab=50, d=32, d_s=8, heads=4, head_dim=8, spike="graded", chunk=16, k_mean=4, k_min=2, k_max=6,
               bptt=3, state="", refractory=1, slots=4, waves=True, ws_direct=True, head_group=2, head_groups=2,
               deep_sup=0.3, halt=True, think_w=0.1, think_fb=True, think_use=True, think_target="latent",
               moe=dict(n_exp=8, k=2, hidden=16, shared=16, capacity=1.5, birth_mode="split", birth_delta=0.1))
    cfg.update(kw)
    torch.manual_seed(0)
    m = LoopBrain(**cfg).float()
    for s in m.spikes():
        s.calibrated.fill_(True)
    return m


def _step(m, x, y, chunk):
    m.chunk_ce = chunk
    m.train()
    m.K = 5
    torch.manual_seed(1)
    _, loss = m(x, y)
    extra = m.aux_loss()
    total = loss + (extra if extra is not None else 0)
    total.backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    return loss.detach(), total.detach(), grads, m.halt_stats


def _compare_model(**kw):
    base = _loop(**kw)
    x = torch.randint(1, 50, (2, 17), generator=torch.Generator().manual_seed(3))
    y = torch.randint(1, 50, (2, 17), generator=torch.Generator().manual_seed(4))
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    l0, t0, g0, hs0 = _step(a, x, y, 0)
    l1, t1, g1, hs1 = _step(b, x, y, 7)
    _close(l1, l0, rtol=0)
    _close(t1, t0, rtol=0)
    assert set(g0) == set(g1)
    for n in g0:
        _close(g1[n], g0[n])
    _close(hs1["expected_passes"], hs0["expected_passes"], rtol=0)


def test_loop_brain_deep_sup_halt():
    _compare_model()


def test_loop_brain_readout_frac():
    _compare_model(readout_frac=0.5)


def test_select_writes_ce_h():
    """blob_write.select_writes from (ce, H, V) == from the logits."""
    import blob_write
    torch.manual_seed(5)
    B, T, V = 2, 20, 50
    logits = torch.randn(B, T, V)
    y = torch.randint(0, V, (B, T))
    lp = F.log_softmax(logits, -1)
    ce = -lp.gather(-1, y[..., None]).squeeze(-1)
    H = -(lp.exp() * lp).sum(-1)

    def bm():
        return types.SimpleNamespace(ch_mean=torch.zeros(32), ch_n=torch.zeros(32, dtype=torch.long), eot=None,
                                     cand_rate=0.1, priv=types.SimpleNamespace(occ=torch.zeros(3, 4, dtype=torch.bool)),
                                     _kept_pairs=[], _wstats=[], _wsel=None)
    b0, b1 = bm(), bm()
    w0 = blob_write.select_writes(b0, logits, y)
    w1 = blob_write.select_writes(b1, None, y, ce_h=(ce, H, V))
    for k in ("cand_tok", "valid", "tok2cand", "chosen"):
        assert torch.equal(w0[k], w1[k]), k
    _close(w1["sig"], w0["sig"], rtol=0)
    _close(b1.ch_mean, b0.ch_mean, rtol=0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
