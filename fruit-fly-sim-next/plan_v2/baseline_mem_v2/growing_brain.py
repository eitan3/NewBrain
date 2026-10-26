"""A trainable spiking "brain" that grows while it learns. Prototype for NewBrain.

- Neurons: the same leaky integrate-and-fire family as the fly model
  (voltage v, synaptic current g with its own decay, threshold, reset).
- Synapses: a recurrent weight matrix with a connectivity mask (most pairs are
  not connected, like a real brain). Weights are trained on GPU with
  backprop-through-time using a "surrogate gradient" for the spike.
- Growth: when learning stalls, `grow()` adds neurons and synapses. Existing
  weights are kept and the new neurons start with zero output weight, so the
  network computes exactly the same thing right after growing and training
  simply continues (optimizer state is carried over too).
- Rewiring: `rewire()` prunes the weakest synapses and sprouts new random ones,
  keeping the synapse count fixed.
- Curriculum: the task (recognise temporal spike patterns) gets harder, with
  more classes, each time the brain masters the current level. New classes add
  output neurons the same way.

Run:  python growing_brain.py --device cuda    (or cpu)
      python growing_brain.py --resume brain.pt   to continue a saved brain
"""
import argparse
import math
import time

import torch
import torch.nn as nn


class SpikeFn(torch.autograd.Function):
    """Heaviside spike forward, fast-sigmoid derivative backward."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1 + 10 * x.abs()) ** 2


spike = SpikeFn.apply


class GrowingSNN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, p_conn=0.1, tau_mem=20.0, tau_syn=5.0, dt=1.0):
        super().__init__()
        self.n_in, self.n_hidden, self.n_out, self.p_conn = n_in, n_hidden, n_out, p_conn
        self.a_v, self.a_g = math.exp(-dt / tau_mem), math.exp(-dt / tau_syn)
        self.w_in = nn.Parameter(torch.randn(n_hidden, n_in) / math.sqrt(n_in))
        self.w_rec = nn.Parameter(torch.randn(n_hidden, n_hidden) / math.sqrt(p_conn * n_hidden))
        self.w_out = nn.Parameter(torch.randn(n_out, n_hidden) / math.sqrt(n_hidden))
        self.register_buffer("mask", (torch.rand(n_hidden, n_hidden) < p_conn).float().fill_diagonal_(0))

    def n_synapses(self):
        return int(self.mask.sum().item()) + self.w_in.numel()

    def forward(self, x):
        """x: (time, batch, n_in) input spikes -> (batch, n_out) logits, mean firing rate."""
        T, B, _ = x.shape
        v = x.new_zeros(B, self.n_hidden)
        g = torch.zeros_like(v)
        s = torch.zeros_like(v)
        out = x.new_zeros(B, self.n_out)
        w_rec = self.w_rec * self.mask
        rate = 0.0
        for t in range(T):
            g = self.a_g * g + x[t] @ self.w_in.T + s @ w_rec.T
            v = self.a_v * v + (1 - self.a_v) * g
            s = spike(v - 1.0)
            v = v - s.detach() * v  # reset to 0 after a spike
            out = out + s @ self.w_out.T
            rate = rate + s.mean()
        return out / T, rate / T

    # ---------------- growth ----------------
    @torch.no_grad()
    def grow(self, k, opt=None):
        """Add k hidden neurons without changing what the network currently computes."""
        n, dev = self.n_hidden, self.w_rec.device
        new_in = torch.randn(k, self.n_in, device=dev) / math.sqrt(self.n_in)
        scale = 1 / math.sqrt(self.p_conn * (n + k))
        w_rec = torch.randn(n + k, n + k, device=dev) * scale
        w_rec[:n, :n] = self.w_rec
        w_rec[:n, n:] = 0  # old neurons don't hear the new ones yet...
        mask = (torch.rand(n + k, n + k, device=dev) < self.p_conn).float().fill_diagonal_(0)
        mask[:n, :n] = self.mask
        w_out = torch.cat([self.w_out, torch.zeros(self.n_out, k, device=dev)], 1)  # ...and they don't vote yet
        old = {"w_in": self.w_in, "w_rec": self.w_rec, "w_out": self.w_out}
        self.w_in = nn.Parameter(torch.cat([self.w_in, new_in]))
        self.w_rec = nn.Parameter(w_rec)
        self.w_out = nn.Parameter(w_out)
        self.mask = mask
        self.n_hidden = n + k
        return self._carry_optimizer(opt, old)

    @torch.no_grad()
    def add_outputs(self, k, opt=None):
        """Add k output units (new classes) with zero weights."""
        old = {"w_out": self.w_out}
        self.w_out = nn.Parameter(torch.cat([self.w_out, torch.zeros(k, self.n_hidden, device=self.w_out.device)]))
        self.n_out += k
        return self._carry_optimizer(opt, old)

    @torch.no_grad()
    def rewire(self, frac=0.05):
        """Prune the weakest `frac` of recurrent synapses and grow as many new random ones."""
        live = self.mask.bool()
        k = int(frac * live.sum())
        if k == 0:
            return
        strength = (self.w_rec.abs() + (~live) * 1e9).flatten()
        self.mask.view(-1)[strength.topk(k, largest=False).indices] = 0
        empty = (self.mask == 0) & ~torch.eye(self.n_hidden, dtype=torch.bool, device=self.mask.device)
        cand = empty.flatten().nonzero().squeeze(1)
        pick = cand[torch.randperm(len(cand), device=cand.device)[:k]]
        self.mask.view(-1)[pick] = 1
        self.w_rec.view(-1)[pick] = 0  # new synapses start silent and are learned

    def _carry_optimizer(self, opt, old):
        """Build a new Adam whose moment estimates keep the old values for old weights."""
        if opt is None:
            return None
        new_opt = torch.optim.Adam(self.parameters(), lr=opt.param_groups[0]["lr"])
        for name, p_new in self.named_parameters():
            p_old = old.get(name)
            if p_old is None:
                p_old = p_new  # parameter was not replaced
                st = opt.state.get(p_new)
            else:
                st = opt.state.get(p_old)
            if not st:
                continue
            ns = {"step": st["step"].clone()}
            for key in ("exp_avg", "exp_avg_sq"):
                buf = torch.zeros_like(p_new)
                sl = tuple(slice(0, d) for d in st[key].shape)
                buf[sl] = st[key]
                ns[key] = buf
            new_opt.state[p_new] = ns
        return new_opt

    # ---------------- save / load ----------------
    def save(self, path, extra=None):
        torch.save({"cfg": dict(n_in=self.n_in, n_hidden=self.n_hidden, n_out=self.n_out, p_conn=self.p_conn),
                    "state": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device)
        m = cls(**ck["cfg"]).to(device)
        m.load_state_dict(ck["state"])
        return m, ck["extra"]


class PatternTask:
    """Each class is a fixed random spatio-temporal firing pattern; samples are noisy, jittered copies."""

    def __init__(self, n_in=20, T=60, max_classes=1024, device="cpu", seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.T, self.n_in, self.dev = T, n_in, device
        # smooth random rate profiles in [0, 0.3] spikes/step
        base = torch.rand(max_classes, T // 10, n_in, generator=gen)
        self.rates = (base.repeat_interleave(10, 1) ** 3 * 0.3).to(device)

    def batch(self, n_classes, B):
        y = torch.randint(0, n_classes, (B,), device=self.dev)
        r = self.rates[y]                                     # (B, T, n_in)
        shift = torch.randint(-5, 6, (B,), device=self.dev)   # timing jitter
        r = torch.stack([torch.roll(r[i], int(shift[i]), 0) for i in range(B)])
        x = (torch.rand_like(r) < r + 0.02).float()           # Poisson spikes + background noise
        return x.transpose(0, 1), y                           # (T, B, n_in)


def accuracy(model, task, n_classes, B=512):
    with torch.no_grad():
        x, y = task.batch(n_classes, B)
        logits, _ = model(x)
        return (logits.argmax(1) == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--max-hidden", type=int, default=4096)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--save", default="brain.pt")
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(0)

    task = PatternTask(device=dev)
    if a.resume:
        model, extra = GrowingSNN.load(a.resume, dev)
        n_classes = extra.get("n_classes", model.n_out)
        print(f"resumed {a.resume}: {model.n_hidden} neurons, {n_classes} classes")
    else:
        model, n_classes = GrowingSNN(task.n_in, a.hidden, a.classes).to(dev), a.classes
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    best, stall, t0 = 0.0, 0, time.time()

    for step in range(1, a.steps + 1):
        x, y = task.batch(n_classes, 128)
        logits, rate = model(x)
        loss = nn.functional.cross_entropy(logits * 5, y) + 0.1 * (rate - 0.05) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            acc = accuracy(model, task, n_classes)
            print(f"step {step:5d}  classes {n_classes:3d}  neurons {model.n_hidden:5d}  "
                  f"synapses {model.n_synapses():8,d}  rate {rate.item():.3f}  acc {acc:.2f}  "
                  f"({time.time()-t0:.0f}s)")
            model.rewire(0.02)
            if acc >= 0.9 and n_classes * 2 <= task.rates.shape[0]:
                opt = model.add_outputs(n_classes, opt)   # harder level: twice the classes
                n_classes *= 2
                best, stall = 0.0, 0
                print(f"  -> mastered; now {n_classes} classes")
            elif acc > best + 0.02:
                best, stall = acc, 0
            else:
                stall += 1
                if stall >= 3 and model.n_hidden < a.max_hidden:
                    k = model.n_hidden // 2
                    xc, _ = task.batch(n_classes, 256)
                    with torch.no_grad():
                        before = model(xc)[0]
                    opt = model.grow(k, opt)
                    with torch.no_grad():
                        change = (model(xc)[0] - before).abs().max().item()
                    print(f"  -> stalled; grew +{k} neurons (output change on same inputs: {change:.1e})")
                    stall = 0
    model.save(a.save, {"n_classes": n_classes})
    print("saved", a.save)


if __name__ == "__main__":
    main()
