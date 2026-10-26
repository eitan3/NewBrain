"""BrainOpt: the v2 weight-update rule (DESIGN_v2.md section 5).

- Hidden weight matrices (2-D, or 3-D stacks of expert matrices): Muon = momentum, then
  orthogonalised by 5 Newton-Schulz steps, scaled to AdamW-like size (0.2 * sqrt(max(m, n))).
  No second moment, so newborn weights need no guessed optimizer state.
- Everything else (embeddings, head, norms, gains, decays, thresholds): AdamW.
- lazy: a parameter (or expert / embedding row) that got no gradient this step is left completely
  untouched: no momentum decay, no weight decay. Sleeping parts of the brain don't fade.
  fine_lazy: the membranes of pass-routed head groups (SSDMixer with n_groups) are lazy per row, column
  and vector element, so a head group that sat idle this step is not touched at all (its output columns
  and its per-head vectors included). Every other parameter keeps the plain rule above.
- maturity: per-element learning-rate multiplier 1 + (k - 1) * exp(-age / tau); every weight's age
  counts from its birth (start of training, or the growth step that created it). Young neurons
  learn fast, mature ones settle.
Optimizer state is stored per parameter name, so growth just pads it (carry_brain_state). Memory: the birth
times exist only with maturity on, and the AdamW step count is kept at the lazy mask's resolution (count_shape).
"""
import math

import torch


def newton_schulz(G, steps=5, eps=1e-7):
    """Approximate orthogonalisation of the last two dims (Muon's quintic iteration)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16 if G.is_cuda else torch.float32)
    tr = X.size(-2) > X.size(-1)
    if tr:
        X = X.transpose(-1, -2)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        X = a * X + (b * A + c * A @ A) @ X
    if tr:
        X = X.transpose(-1, -2)
    return X.to(G.dtype)


class BrainOpt:
    def __init__(self, model, lr=2e-3, wd=0.1, muon=True, lazy=True, maturity=0.0, tau=1000,
                 betas=(0.9, 0.95), momentum=0.95, alpha_lr=0.1, fine_lazy=True):
        self.lr, self.wd, self.muon, self.lazy = lr, wd, muon, lazy
        self.k_mat, self.tau, self.betas, self.momentum, self.alpha_lr = maturity, tau, betas, momentum, alpha_lr
        self.t = 0
        self.groups = {}  # name -> (param, kind)
        for name, p in model.named_parameters():
            hidden = p.dim() >= 2 and not name.startswith(("emb.", "heads.")) and "conv" not in name and \
                ".cx." not in name and ".cq." not in name and ".ck." not in name and \
                not name.endswith("norm_w") and "router" not in name  # routers on AdamW: Muon's big normalised
            # steps moved the shared loop router faster than the balancer could follow (26% dropped tokens)
            # memory (plan_v2/12): the translators' W_q / sub-keys ("mem.keys.") and the gates ("mem.gates.") on
            # AdamW (Muon's orthogonalised equal-size steps are wrong for keys and gates); the tables are buffers
            kind = "muon" if (muon and hidden and not name.startswith(("mem.keys.", "mem.gates."))) else "adam"
            self.groups[name] = (p, kind)
        self.state = {}  # name -> dict of tensors
        self.fine = set()  # parameters lazy per row / column / element: the head-group-sliced membranes
        if fine_lazy:
            for mn, mod in model.named_modules():
                if getattr(mod, "n_groups", 0) and hasattr(mod, "hgroup"):
                    self.fine |= {f"{mn}.{pn}" if mn else pn for pn, _ in mod.named_parameters()}

    def lr_mult(self, name, p, st):
        if not self.k_mat:
            return 1.0
        if "born" not in st:  # (state carried from an optimizer without maturity: born now)
            st["born"] = torch.full_like(p, float(self.t - 1), dtype=torch.float32)
        age = (self.t - st["born"]).clamp(min=0)
        return 1 + (self.k_mat - 1) * torch.exp(-age / self.tau)

    def count_shape(self, name, p):
        """Shape of an AdamW parameter's step count "n". n only advances where the lazy mask is active, so it
        needs the mask's resolution: per element for the fine (head-group) parameters, per row for other
        matrices (one count per row instead of per weight), one per tensor for vectors / scalars."""
        if name in self.fine:
            return tuple(p.shape)
        if self.lazy and p.dim() >= 2:
            return (p.size(0),) + (1,) * (p.dim() - 1)
        return (1,) * p.dim()

    def _active_rows(self, g, fine=False):
        """Mask of rows (dim 0) that received any gradient, broadcastable to g. fine (head-group membranes):
        matrices also per column (the output columns of idle heads), vectors per element."""
        if g.dim() <= 1:  # vectors and scalars (e.g. the wave strengths) are handled as one unit
            return g.ne(0) if fine else None
        nz = g.flatten(1).abs().sum(1) != 0
        mask = nz.view(-1, *([1] * (g.dim() - 1)))
        if fine and g.dim() == 2:
            mask = mask & (g.abs().sum(0) != 0)[None, :]
        return mask

    @torch.no_grad()
    def step(self, lr_scale=1.0):
        """No host-device syncs: "lazy" is applied with masks instead of Python-side checks, and Muon
        orthogonalises all hidden matrices of the same shape in one batched Newton-Schulz call."""
        self.t += 1
        lr = self.lr * lr_scale
        b1, b2 = self.betas
        muon_by_shape = {}
        for name, (p, kind) in self.groups.items():
            g = p.grad
            if g is None:
                continue
            st = self.state.get(name)
            if st is None:  # (no per-element birth times without maturity: lr_mult is 1 then)
                st = self.state[name] = ({"born": torch.full_like(p, float(self.t - 1), dtype=torch.float32)}
                                         if self.k_mat else {})
            fine = name in self.fine
            mask = self._active_rows(g, fine) if self.lazy else None
            if mask is None and self.lazy:  # 1-D parameter: one mask for the whole tensor
                mask = g.ne(0).any().reshape([1] * g.dim())  # works for scalars too
            mult = self.lr_mult(name, p, st)
            plr = lr * (self.alpha_lr if name.endswith(".alpha") else 1.0)
            decay = self.wd if (p.dim() >= 2 and not name.startswith(("emb.", "mem.keys.")) and
                                "router_conn" not in name) else 0.0  # (normalised keys: no weight decay)
            if kind == "muon":
                buf = st.setdefault("mom", torch.zeros_like(p))
                new_buf = buf * self.momentum + g
                upd_in = g + self.momentum * new_buf  # nesterov
                if fine and mask is not None:  # idle heads' stale momentum must not shape the others' step
                    upd_in = torch.where(mask, upd_in, torch.zeros_like(upd_in))
                key = tuple(p.shape[-2:])
                muon_by_shape.setdefault(key, []).append((p, buf, new_buf, upd_in, mask, mult, plr, decay))
            else:
                m = st.setdefault("m", torch.zeros_like(p))
                v = st.setdefault("v", torch.zeros_like(p))
                n = st.get("n")  # step count at the lazy mask's resolution (count_shape)
                if n is None:
                    n = st["n"] = torch.zeros(self.count_shape(name, p), dtype=torch.float32, device=p.device)
                act = mask if mask is not None else torch.ones((), dtype=torch.bool, device=p.device)
                n.add_(act.float())
                m.copy_(torch.where(act, b1 * m + (1 - b1) * g, m))
                v.copy_(torch.where(act, b2 * v + (1 - b2) * g * g, v))
                # (a full-size temporary, as before: CPU pow differs in the last bit between its vectorised
                # and scalar paths, so the bias correction is computed on exactly the old layout)
                nn_ = n.clamp(min=1).expand(p.shape).contiguous()
                mh = m / (1 - b1 ** nn_)
                vh = v / (1 - b2 ** nn_)
                upd = mh / (vh.sqrt() + 1e-8) * mult + decay * p
                p.sub_(torch.where(act, plr * upd, torch.zeros_like(upd)))
        for (m_rows, n_cols), items in muon_by_shape.items():
            # one batched orthogonalisation per matrix shape (expert stacks are flattened into the batch)
            mats = [it[3].reshape(-1, m_rows, n_cols) for it in items]
            O_all = newton_schulz(torch.cat(mats)) * 0.2 * math.sqrt(max(m_rows, n_cols))
            O_parts = torch.split(O_all, [x.size(0) for x in mats])
            for (p, buf, new_buf, _, mask, mult, plr, decay), O in zip(items, O_parts):
                upd = O.view_as(p) * mult + decay * p
                if mask is not None:  # lazy: experts / rows without gradient keep everything as is
                    buf.copy_(torch.where(mask, new_buf, buf))
                    p.sub_(torch.where(mask, plr * upd, torch.zeros_like(upd)))
                else:
                    buf.copy_(new_buf)
                    p.sub_(plr * upd)

    def zero_grad(self, set_to_none=True):
        for p, _ in self.groups.values():
            p.grad = None


def carry_brain_state(old_opt, model, new_opt):
    """Move state to an optimizer built on the grown model. New slices: zero momentum, median second
    moment, step count 0, born = now (so they get the young-neuron learning rate)."""
    new_opt.t = old_opt.t
    for name, (p, _) in new_opt.groups.items():
        st = old_opt.state.get(name)
        if not st:
            continue
        ns = {}
        for key, old in st.items():
            shape = p.shape
            if key == "n":  # the step count lives at the lazy mask's resolution (per row / per tensor)
                shape, old_shape = new_opt.count_shape(name, p), st["m"].shape  # (m: the old parameter's shape)
                # a count shared along a dimension stays shared only while that dimension does not grow (new
                # columns of old rows start at 0, as with a per-element count); a finer count stays per element
                if any(c == 1 and (o != 1 or a != b) for c, o, a, b in zip(shape, old.shape, p.shape, old_shape)):
                    old, shape = old.expand(old_shape), p.shape
            if old.shape == shape:
                ns[key] = old.clone()
                continue
            fill = {"v": old.median().item(), "born": float(old_opt.t)}.get(key, 0.0)
            buf = torch.full(shape, fill, dtype=old.dtype, device=old.device)
            buf[tuple(slice(0, s) for s in old.shape)] = old
            ns[key] = buf
        new_opt.state[name] = ns
    return new_opt
