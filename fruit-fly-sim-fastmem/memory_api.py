"""The memory-module interface of the loop brain (README_memory_api.md).

A memory module is what LoopBrain keeps in `self.bm` and what RoutedMLP calls inside the routed experts. Today there
are three choices (train_lm.py --memory):

    blob  blob_memory.BlobMemory (the --gm ... flags; the default, bit-identical to fruit-fly-sim-next)
    wave  wave_memory: CodedWaveMemory (--wave-mode on) or the stub WaveMemory (--wave-mode stub, reads zeros)
    fastmem  fastmem.FastMem: per-expert + global fast-weight matrices (SPEC_fastmem.md)
    none  no module at all (model.bm is None, exactly the --gm 0 code path)

Every module subclasses MemoryBase. The base class registers NOTHING (no parameter, no buffer), so subclassing it does
not change a module's state_dict.

The read (one call per pass, inside RoutedMLP.forward, after the capacity layout of the routed experts is filled):

    module(xf, buf, x, side, e_k, p_k, t_k, slot_k, counts, N, E, C, n_keep, call=k) -> (buf, x_shared)

      xf (N, D)       the experts' input rows (the "thought" of each token at this pass); N = B * T
      buf (E, C, D)   the routed experts' capacity layout (buf[e_k, p_k] = xf[t_k])
      x (B, T, D)     the shared expert's input
      side            the router's side input (workspace summary) or None
      e_k, p_k, t_k   the kept (expert, capacity slot, token) pair list, each (n_pairs,)
      slot_k          the top-k slot of each pair; counts (E,) tokens per expert; N, E, C, n_keep sizes
      call            the pass index (a cache id: activation checkpointing re-runs the forward)

  MemoryBase.__call__ calls read(), which returns the injected vectors
      (r_pairs (n_pairs, D) or None, r_shared (N, D) or None)
  and adds them: buf[e_k, p_k] += r_pairs, x_shared = x + r_shared. A module overrides read() (the simple way) or
  __call__ (full control: the blob memory does, it has its own masked / compacted / gated layout).

The trainer hooks (train_lm.py calls them in this order each optimizer step):

    step_begin(step, args) -> [log lines]     before the step's evaluation (the blob: step stamp, fit window, fit)
    begin_forward()                           LoopBrain.run, once per model forward
    select_writes(logits, y, x, ce_h=None)    between a micro-batch's loss and its backward
    after_backward()                          after each micro-batch's backward (memory-side losses go here; they
                                              must not reach the brain: tests_memory_api.py (f))
    after_step(step, refit_every=0)           after the brain's optimizer step (the memory's own update)
    on_grow(src, E0, split)                   experts E0 .. E0 + len(src) - 1 were born from parents src

  plus logging: describe(args), log_record(), health(), diag_snapshot(), metrics(step).

State (swap support):

    memory_state_dict() / load_memory_state_dict(sd)   the module's state alone (the blob: its state_dict)
    memory_cfg()                                       what is needed to rebuild it (LoopBrain.make_memory)
    save_memory_file(path, model, step) / load_memory_file(path, model)   memory.pt files next to checkpoints

Runtime switch: module.enabled = False (or `with memory_off(model):`) makes LoopBrain run exactly the no-memory code
path (the module is not called at all); nothing in the memory changes.

Train-only mode (--memory-train-only experts=<ids>): set_train_only(ids) restricts the memory's learning to those
experts' own memory; freeze_for_train_only(model, opt, ids) freezes the brain (requires_grad False, removed from the
optimizer) and every memory parameter not returned by trainable_parameters(ids).
"""
import contextlib
import os
from pathlib import Path

import torch
import torch.nn as nn

MEMORY_KINDS = ("blob", "wave", "fastmem", "none")
MEM_PREFIX = "bm."          # the memory module's key prefix in the model's state_dict (LoopBrain.bm)


class MemoryBase(nn.Module):
    """The interface (module doc). Registers nothing: a subclass's state_dict is exactly its own."""

    kind = "base"
    enabled = True          # (class default: a plain attribute, never saved; LoopBrain skips the module when False)
    train_only = None       # (None, or the expert ids whose memory alone learns: set_train_only)

    # ------------------------------------------------------------------ the read
    def read(self, xf, e_k, t_k, p_k=None, slot_k=None, side=None, call=None):
        """-> (r_pairs (n_pairs, D) or None, r_shared (N, D) or None): what is added to each kept pair's expert input
        and to each token's shared-expert input."""
        raise NotImplementedError(f"{type(self).__name__}.read")

    def __call__(self, xf, buf, x, side, e_k, p_k, t_k, slot_k, counts, N, E, C, n_keep, call=None):
        """The RoutedMLP hook: read(), then the injection buf[e_k, p_k] += r_pairs, x_shared = x + r_shared. Override
        read() for the simple case, or __call__ itself for full control (the blob memory does)."""
        r_p, r_s = self.read(xf, e_k, t_k, p_k=p_k, slot_k=slot_k, side=side, call=call)
        if r_p is not None:
            buf = buf.index_put((e_k, p_k), r_p.to(buf.dtype), accumulate=True)
        if r_s is not None:
            x = x + r_s.to(x.dtype).view(x.shape)
        return buf, x

    # ------------------------------------------------------------------ trainer hooks (defaults: nothing)
    @property
    def ready(self):
        """True once the module reads (the blob: after its address fit)."""
        return True

    def trainer_setup(self, args):
        pass

    def describe(self, args=None):
        return f"memory: {self.kind} ({self.nbytes() / 1e6:.2f} MB of state)"

    def step_begin(self, step, args=None):
        return []

    def begin_forward(self):
        pass

    def select_writes(self, logits, targets, idx=None, ce_h=None):
        return None

    def after_backward(self):
        return None

    def after_step(self, step, refit_every=0):
        pass

    def on_grow(self, src, E0, split=True):
        pass

    def log_record(self, step=None):
        return {}

    def health(self):
        return {}

    def diag_snapshot(self):
        return {}

    def metrics(self, step=None):
        return {}

    # ------------------------------------------------------------------ state
    def memory_cfg(self):
        return {}

    def memory_state_dict(self):
        return self.state_dict()

    def load_memory_state_dict(self, sd, strict=True):
        return self.load_state_dict(sd, strict=strict)

    def nbytes(self):
        return sum(b.numel() * b.element_size() for b in self.buffers()) + \
            sum(p.numel() * p.element_size() for p in self.parameters())

    # ------------------------------------------------------------------ train-only mode
    def set_train_only(self, expert_ids):
        """Restrict the memory's learning to these experts' own memory (None = off). -1 = the shared expert."""
        self.train_only = None if expert_ids is None else [int(e) for e in expert_ids]

    def trainable_parameters(self, expert_ids):
        """The module's nn.Parameters that stay trainable in train-only mode (default: none)."""
        return []


# ---------------------------------------------------------------------- helpers used by the trainer / tests
def memory_of(model):
    return getattr(model, "bm", None)


@contextlib.contextmanager
def memory_off(model):
    """Runtime memory-off switch (evaluation): LoopBrain runs the no-memory code path while inside."""
    m = memory_of(model)
    if m is None:
        yield
        return
    old = m.enabled
    m.enabled = False
    try:
        yield
    finally:
        m.enabled = old


def split_state(state, prefix=MEM_PREFIX):
    """A model state_dict -> (brain part, memory part with the prefix stripped)."""
    brain = {k: v for k, v in state.items() if not k.startswith(prefix)}
    mem = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    return brain, mem


def memory_path(ckpt_path):
    """The memory file next to a checkpoint: runs/x/last.pt -> runs/x/last.memory.pt."""
    p = Path(ckpt_path)
    return p.with_name(p.stem + ".memory.pt")


def save_memory_file(path, model, step=None):
    """The memory module's state alone (atomic write). Returns the path (None when the model has no memory)."""
    m = memory_of(model)
    if m is None:
        return None
    obj = dict(kind=m.kind, cfg=m.memory_cfg(), state=m.memory_state_dict(), step=step,
               n_experts=int(model.blocks[0].mlp.E), d=int(model.cfg["d"]))
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return Path(path)


def load_memory_file(path, model, strict=True):
    """Load a memory.pt into the model's memory module (the kinds must match; sizes must fit the configured module)."""
    m = memory_of(model)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if m is None:
        raise ValueError(f"--memory-load {path}: the model has no memory module (--memory none / --gm 0)")
    if obj["kind"] != m.kind:
        raise ValueError(f"--memory-load {path}: a {obj['kind']} memory, the model's is {m.kind}")
    if obj.get("d") is not None and obj["d"] != model.cfg["d"]:
        raise ValueError(f"--memory-load {path}: width {obj['d']}, the model's d is {model.cfg['d']}")
    dev = next((b.device for b in model.buffers()), torch.device("cpu"))
    state = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in obj["state"].items()}
    m.load_memory_state_dict(state, strict=strict)
    return obj


def parse_train_only(spec):
    """'experts=1,3' / 'experts=shared,2' -> [1, 3] / [-1, 2]; '' -> None."""
    if not spec:
        return None
    key, _, val = spec.partition("=")
    if key.strip() != "experts" or not val.strip():
        raise ValueError(f"--memory-train-only {spec!r}: expected experts=<id>[,<id>...] (id: int or 'shared')")
    out = []
    for v in val.split(","):
        v = v.strip()
        out.append(-1 if v in ("shared", "-1") else int(v))
    if any(e < -1 for e in out):
        raise ValueError(f"--memory-train-only {spec!r}: negative expert id")
    return out


def freeze_for_train_only(model, opt, expert_ids):
    """Train-only mode: every model parameter except the memory's trainable_parameters(ids) gets requires_grad False
    and leaves the optimizer (BrainOpt: its groups; torch AdamW: its param groups). Returns the trainable names."""
    m = memory_of(model)
    keep = {id(p) for p in (m.trainable_parameters(expert_ids) if m is not None else [])}
    names = []
    for n, p in model.named_parameters():
        on = id(p) in keep
        p.requires_grad_(on)
        p.grad = None
        if on:
            names.append(n)
    if hasattr(opt, "groups"):          # BrainOpt
        opt.groups = {n: g for n, g in opt.groups.items() if g[0].requires_grad}
        opt.state = {n: s for n, s in opt.state.items() if n in opt.groups}
    elif opt is not None:               # torch optimizer
        for g in opt.param_groups:
            for p in g["params"]:
                if not p.requires_grad:
                    opt.state.pop(p, None)
            g["params"] = [p for p in g["params"] if p.requires_grad]
    return names


class BufferFreeze:
    """Train-only mode: the brain's buffers (homeostasis thresholds, router balance bias, load EMAs ...) restored after
    every optimizer step, so the frozen brain stays exactly as loaded (its parameters are frozen by requires_grad)."""

    def __init__(self, model, prefix=MEM_PREFIX):
        self.items = [(n, b, b.detach().clone()) for n, b in model.named_buffers() if not n.startswith(prefix)]

    @torch.no_grad()
    def restore(self):
        for _, b, s in self.items:
            b.copy_(s)


def byte_budget(model_or_module, cap_bytes=0):
    """The memory state's bytes vs a cap: persistent (what memory.pt holds), derived (rebuilt at load), parameters,
    by top-level group. cap_bytes 0 = no cap."""
    m = memory_of(model_or_module) if not isinstance(model_or_module, MemoryBase) else model_or_module
    rep = dict(kind=None if m is None else m.kind, persistent=0, derived=0, params=0, groups={})
    if m is None:
        rep.update(total=0, cap=int(cap_bytes), ok=True, frac=0.0)
        return rep
    pers = set(m.state_dict().keys())
    for n, b in m.named_buffers():
        nb = b.numel() * b.element_size()
        kind = "persistent" if n in pers else "derived"
        rep[kind] += nb
        g = rep["groups"].setdefault(n.split(".")[0] if "." in n else n, [0, 0])
        g[0 if kind == "persistent" else 1] += nb
    for n, p in m.named_parameters():
        rep["params"] += p.numel() * p.element_size()
    extra = 0
    for k, v in m.state_dict().items():          # (_extra_state: e.g. the blob gate's weights / optimizer state)
        if k.endswith("_extra_state"):
            extra += _nbytes(v)
    rep["extra_state"] = extra
    rep["total"] = rep["persistent"] + rep["params"] + extra
    rep["cap"] = int(cap_bytes)
    rep["ok"] = not cap_bytes or rep["total"] <= cap_bytes
    rep["frac"] = rep["total"] / cap_bytes if cap_bytes else 0.0
    return rep


def _nbytes(v):
    if torch.is_tensor(v):
        return v.numel() * v.element_size()
    if isinstance(v, dict):
        return sum(_nbytes(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return sum(_nbytes(x) for x in v)
    return 0


def budget_line(rep):
    mb = lambda b: f"{b / 1e6:.2f} MB"
    s = (f"memory budget ({rep['kind']}): state {mb(rep['total'])} (persistent buffers {mb(rep['persistent'])}, "
         f"params {mb(rep['params'])}, extra {mb(rep.get('extra_state', 0))}; derived, not saved {mb(rep['derived'])})")
    if rep["cap"]:
        s += f" vs cap {mb(rep['cap'])}: {100 * rep['frac']:.1f}% {'OK' if rep['ok'] else 'OVER THE CAP'}"
    return s
